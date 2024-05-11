# Copyright 2022 Applied BioComputation Group, Stony Brook University
# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, List
from functools import  partial, reduce
from operator import add
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tuned_abs.common import residue_constants as rc
from . import structure_multimer, all_atom_multimer, rigid
from .tensor_utils import masked_mean
from .chunk_utils import chunk_layer
from .primitives import Attention, AxisWiseDropout

def gumbel_noise(shape: Sequence[int]) -> torch.tensor:
    """Generate Gumbel Noise of given Shape.

    This generates samples from Gumbel(0, 1).

    Args:
        key: Jax random number key.
        shape: Shape of noise to return.

    Returns:
        Gumbel noise of given shape.
    """
    epsilon = 1e-6
    uniform_noise = torch.rand(shape)
    gumbel = -torch.log(-torch.log(uniform_noise + epsilon) + epsilon)
    return gumbel

def make_msa_profile(batch):
    """Compute the MSA profile."""

    # Compute the profile for every residue (over all MSA sequences).
    return masked_mean(
        batch['msa_mask'][...,:, :, None], 
        torch.nn.functional.one_hot(batch['msa'].long(), 22), dim=1
    )

def sample_msa(batch, max_seq):
    logits = (torch.clip(torch.sum(batch['msa_mask'], -1), 0., 1.) - 1.) * 1e6

    rand_ind = torch.randperm(logits.shape[-1] - 1) + 1
    index_order = torch.cat((torch.tensor([0]), rand_ind))
    
    sel_idx = index_order[:max_seq]
    extra_idx = index_order[max_seq:]
    batch_sp = {
        k: v.clone()
        for k, v in batch.items()
    }
    for k in ['msa', 'deletion_matrix', 'msa_mask', 'bert_mask']:
        if k in batch_sp:
            batch_sp['extra_' + k] = batch_sp[k][:, extra_idx]
            batch_sp[k] = batch_sp[k][:, sel_idx]
    return batch_sp

def shaped_categorical(probs, epsilon=1e-10):
    ds = probs.shape
    num_classes = ds[-1]
    distribution = torch.distributions.categorical.Categorical(
        torch.reshape(probs + epsilon, [-1, num_classes])
    )
    counts = distribution.sample()
    return torch.reshape(counts, ds[:-1])

def make_masked_msa(batch, config):
  """Create data for BERT on raw MSA."""
  # Add a random amino acid uniformly.
  random_aa = torch.tensor([0.05] * 20 + [0., 0.], dtype=torch.float32, 
                           device=batch['aatype'].device)

  categorical_probs = (
      config['uniform_prob'] * random_aa +
      config['profile_prob'] * batch['msa_profile'] +
      config['same_prob'] * nn.functional.one_hot(batch['msa'].long(), 22))

  # Put all remaining probability on [MASK] which is a new column.
  pad_shapes = list(reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))]))
  pad_shapes[1] = 1
  mask_prob = 1. - config['profile_prob'] - config['same_prob'] - config['uniform_prob']
  assert mask_prob >= 0.
  categorical_probs = torch.nn.functional.pad(
      categorical_probs, pad_shapes, value=mask_prob)
  sh = batch['msa'].shape
  mask_position = torch.rand(sh, device=batch['aatype'].device) < config['replace_fraction']

  bert_msa = shaped_categorical(categorical_probs)
  bert_msa = torch.where(mask_position, bert_msa.type(torch.int32), batch['msa'])
  bert_msa = bert_msa * batch['msa_mask'].type(torch.int32)

  # Mix real and masked MSA.
  batch['bert_mask'] = mask_position.type(torch.float32)
  batch['true_msa'] = batch['msa']
  batch['msa'] = bert_msa

  return batch

def make_masked_msa_idx(batch, index, bert_indices_prefix):
    batch['true_msa'] = batch['msa']
    with np.load(bert_indices_prefix+str(index)+'.npz') as data:
        batch['bert_mask'] = torch.unsqueeze(torch.tensor(data['bert_mask'],device=batch['msa'].device),0)
        batch['msa'] = torch.unsqueeze(torch.tensor(data['bert_msa'],device=batch['msa'].device),0)
    return batch

def nearest_neighbor_clusters(batch, gap_agreement_weight=0.):
  """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

  # Determine how much weight we assign to each agreement.  In theory, we could
  # use a full blosum matrix here, but right now let's just down-weight gap
  # agreement because it could be spurious.
  # Never put weight on agreeing on BERT mask.

  weights = torch.tensor(
      [1.] * 21 + [gap_agreement_weight] + [0.], dtype=torch.float32, device=batch['aatype'].device)

  msa_mask = batch['msa_mask']
  msa_one_hot = torch.nn.functional.one_hot(batch['msa'].long(), 23)

  extra_mask = batch['extra_msa_mask']
  extra_one_hot = torch.nn.functional.one_hot(batch['extra_msa'].long(), 23)

  msa_one_hot_masked = msa_mask[..., None] * msa_one_hot
  extra_one_hot_masked = extra_mask[..., None] * extra_one_hot

  agreement = torch.einsum('...mrc, ...nrc->...nm', extra_one_hot_masked,
                         weights * msa_one_hot_masked)

  cluster_assignment = torch.nn.functional.softmax(1e3 * agreement, 1)
  cluster_assignment *= torch.einsum('...mr, ...nr->...mn', msa_mask, extra_mask)

  cluster_count = torch.sum(cluster_assignment, dim=-1)
  cluster_count += 1.  # We always include the sequence itself.

  msa_sum = torch.einsum('...nm, ...mrc->...nrc', cluster_assignment, extra_one_hot_masked)
  msa_sum += msa_one_hot_masked

  cluster_profile = msa_sum / cluster_count[..., None, None]

  extra_deletion_matrix = batch['extra_deletion_matrix']
  deletion_matrix = batch['deletion_matrix']

  del_sum = torch.einsum('...nm, ...mc->...nc', cluster_assignment,
                       extra_mask * extra_deletion_matrix)
  del_sum += deletion_matrix  # Original sequence.
  cluster_deletion_mean = del_sum / cluster_count[..., None]

  return cluster_profile, cluster_deletion_mean

def create_msa_feat(batch):
  """Create and concatenate MSA features."""
  msa_1hot = torch.nn.functional.one_hot(batch['msa'].long(), 23)
  deletion_matrix = batch['deletion_matrix']
  has_deletion = torch.clip(deletion_matrix, 0., 1.)[..., None]
  deletion_value = (torch.arctan(deletion_matrix / 3.) * (2. / torch.pi))[..., None]

  deletion_mean_value = (torch.arctan(batch['cluster_deletion_mean'] / 3.) *
                         (2. / torch.pi))[..., None]

  msa_feat = [
      msa_1hot,
      has_deletion,
      deletion_value,
      batch['cluster_profile'],
      deletion_mean_value
  ]

  return torch.cat(msa_feat, -1)

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""
    is_gly = torch.eq(aatype, rc.restype_order["G"])
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = torch.where(
            is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta

def create_extra_msa_feature(batch, num_extra_msa):
    extra_msa = batch['extra_msa'][:, :num_extra_msa]
    deletion_matrix = batch['extra_deletion_matrix'][:, :num_extra_msa]
    msa_1hot = torch.nn.functional.one_hot(extra_msa.long(), 23)
    has_deletion = torch.clip(deletion_matrix, 0., 1.)[..., None]
    deletion_value = (torch.arctan(deletion_matrix / 3.) * (2. / torch.pi))[..., None]
    extra_msa_mask = batch['extra_msa_mask'][:, :num_extra_msa]
    return torch.cat([msa_1hot, has_deletion, deletion_value], -1), extra_msa_mask

class OuterProductMean(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_c = config['norm_channel']
        out_c = config['num_output_channel']
        mid_c = config['num_outer_channel']
        self.layer_norm_input = nn.LayerNorm(in_c)
        self.left_projection = nn.Linear(in_c, mid_c)
        self.right_projection = nn.Linear(in_c, mid_c)
        self.output = nn.Linear(mid_c * mid_c, out_c)
        self.mid_c = mid_c
        self.out_c = out_c

    def forward(self, act, mask):
        act = self.layer_norm_input(act)
        mask = mask[..., None]
        left_act = mask*self.left_projection(act)
        right_act = mask*self.right_projection(act)
        x2d = torch.einsum('bmix,bmjy->bjixy', left_act, right_act)
        out = self.output(x2d.flatten(start_dim=-2)).transpose(-2, -3)
        norm = torch.einsum('...abc,...adc->...bdc', mask, mask)
        out = out/(norm +1e-3)
        return out

class RowAttentionWithPairBias(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        attn_num_c = config['attention_channel']
        num_heads = config['num_head']
        in_num_c = config['norm_channel']
        pair_rep_num_c = global_config['pair_channel']
        dropout_rate = config['dropout_rate']
        orientation = config['orientation']

        self.query_norm = nn.LayerNorm(in_num_c)
        self.feat_2d_norm = nn.LayerNorm(pair_rep_num_c)
        self.feat_2d_weights = nn.Linear(pair_rep_num_c, num_heads, bias=False)
        self.attn_num_c = attn_num_c
        self.num_heads = num_heads
        self.mha = Attention(in_num_c, in_num_c, in_num_c, attn_num_c, num_heads)
        if np.isclose(dropout_rate, 0):
            self.dropout = nn.Identity()
        else:
            self.dropout = AxisWiseDropout(dropout_rate, orientation)

    @torch.jit.ignore
    def _chunk(self,
        m: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        use_memory_efficient_kernel: bool,
        use_lma: bool,
    ) -> torch.Tensor:
        def fn(m, biases):
            m = self.query_norm(m)
            return self.mha(
                q_x=m,
                kv_x=m,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma,
            )

        return chunk_layer(
            fn,
            {
                "m": m,
                "biases": biases,
            },
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    
    def forward(self, msa_act, pair_act, msa_mask):
        chunks = []

        for i in range(0, pair_act.shape[-3], 256):
            z_chunk = pair_act[..., i: i + 256, :, :]

            # [*, N_res, N_res, C_z]
            z_chunk = self.feat_2d_norm(z_chunk)

            # [*, N_res, N_res, no_heads]
            z_chunk = self.feat_2d_weights(z_chunk)

            chunks.append(z_chunk)

        z = torch.cat(chunks, dim=-3)

        # [*, 1, no_heads, N_res, N_res]
        z = z.permute(0,3, 1,2).unsqueeze(-4)
        bias = (1e9 * (msa_mask - 1.))[...,:, None, None, :]
        biases = [bias, z]
        out_1d = self._chunk(
                msa_act, 
                biases, 
                256,
                use_memory_efficient_kernel=False, 
                use_lma=False,
            )

        return self.dropout(out_1d)

class ExtraColumnGlobalAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.attn_num_c = config['attention_channel']
        self.num_heads = config['num_head']

        self.query_norm = nn.LayerNorm(global_config['extra_msa_channel'])
        self.q = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c*self.num_heads, bias=False)
        self.k = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c, bias=False)
        self.v = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c, bias=False)
        self.gate = nn.Linear(global_config['extra_msa_channel'], self.attn_num_c * self.num_heads)
        self.output = nn.Linear(self.attn_num_c * self.num_heads, global_config['extra_msa_channel'])

    @torch.jit.ignore
    def _chunk(self,
                m: torch.Tensor,
                mask: torch.Tensor,
                chunk_size: int,
                use_lma: bool,
                ) -> torch.Tensor:
        def fn(m, mask):
            m = self.query_norm(m)
            return self.mha(
                m,
                mask,
                use_lma=use_lma,
            )

        return chunk_layer(
            fn,
            {
                "m": m,
                "mask": mask,
            },
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    def forward(self, msa_act, msa_mask):
        msa_act = msa_act.transpose(-2,-3)
        msa_mask = msa_mask.transpose(-1,-2)
        msa_act = self.query_norm(msa_act)
        q_avg = torch.sum(msa_act, dim=-2)/msa_act.shape[-2]
        q = self.q(q_avg).view(*q_avg.shape[:-1], self.num_heads, self.attn_num_c)
        q = q*(self.attn_num_c ** (-0.5))
        k = self.k(msa_act)
        v = self.v(msa_act)
        gate =  torch.sigmoid(self.gate(msa_act).view(*msa_act.shape[:-1], self.num_heads, self.attn_num_c))
        w = torch.softmax(torch.einsum('bihc,bikc->bihk', q, k), dim=-1)
        out_1d = torch.einsum('bmhk,bmkc->bmhc', w, v)
        out_1d = out_1d.unsqueeze(-3) * gate
        out = self.output(out_1d.view(*out_1d.shape[:-2], self.attn_num_c * self.num_heads))
        return out.transpose(-2,-3)

class LigColumnAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        attn_num_c = config['attention_channel']
        num_heads = config['num_head']
        in_num_c = global_config['msa_channel']

        self.query_norm = nn.LayerNorm(in_num_c)

        self.attn_num_c = attn_num_c
        self.num_heads = num_heads
        self.mha = Attention(in_num_c, in_num_c, in_num_c, attn_num_c, num_heads)
    
    @torch.jit.ignore
    def _chunk(self,
        m: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        use_memory_efficient_kernel: bool,
        use_lma: bool,
    ) -> torch.Tensor:
        def fn(m, biases):
            m = self.query_norm(m)
            return self.mha(
                q_x=m,
                kv_x=m,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma,
            )

        return chunk_layer(
            fn,
            {
                "m": m,
                "biases": biases,
            },
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )


    def forward(self, msa_act, msa_mask):
        msa_act = msa_act.transpose(-2,-3)
        msa_mask = msa_mask.transpose(-1,-2)
        bias = (1e9 * (msa_mask - 1.))[...,:, None, None, :]
        biases=[bias]
        out_1d = self._chunk(
                msa_act,
                biases,
                256,
                use_memory_efficient_kernel=False,
                use_lma=False,
            )

        out_1d = out_1d.transpose(-2,-3)

        return out_1d

class Transition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layer_norm = nn.LayerNorm(config['norm_channel'])
        self.transition1 = nn.Linear(config['norm_channel'], 
                                     config['norm_channel'] * config['num_intermediate_factor'])
        self.transition2 = nn.Linear(config['norm_channel'] * config['num_intermediate_factor'], 
                                     config['norm_channel'])

    def forward(self, act):
        act = self.input_layer_norm(act)
        act = self.transition1(act).relu_()
        act = self.transition2(act)
        return act

class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_c = config['norm_channel']
        mid_c = config['num_intermediate_channel']
        dropout_rate = config['dropout_rate']
        orientation = config['orientation']
        self.fused_projection = config['fuse_projection_weights']
        
        if config['fuse_projection_weights']:
            self.left_norm_input = nn.LayerNorm(in_c)
            self.projection = nn.Linear(in_c, 2*mid_c)
            self.gate = nn.Linear(in_c, 2*mid_c)
        else:
            self.layer_norm_input = nn.LayerNorm(in_c)
            self.left_projection = nn.Linear(in_c, mid_c)
            self.right_projection = nn.Linear(in_c, mid_c)
            self.left_gate = nn.Linear(in_c, mid_c)
            self.right_gate = nn.Linear(in_c, mid_c)
        self.center_layer_norm = nn.LayerNorm(mid_c)
        self.output_projection = nn.Linear(mid_c, in_c)
        self.gating_linear = nn.Linear(in_c, in_c)
        if np.isclose(dropout_rate, 0):
            self.dropout = nn.Identity()
        else:
            self.dropout = AxisWiseDropout(dropout_rate, orientation)

    def forward(self, act, mask):
        if self.fused_projection:
            left_act = self.left_norm_input(act)
            mask = mask[..., None]
            proj_act = mask*self.projection(left_act) * torch.sigmoid(self.gate(left_act))
            left_proj = proj_act[...,:self.mid_c]
            right_proj = proj_act[...,self.mid_c:]
            gate_values = self.gating_linear(left_act)
        else:
            act = self.layer_norm_input(act)
            mask = mask[..., None]
            left_proj = mask*self.left_projection(act) * torch.sigmoid(self.left_gate(act))
            right_proj = mask*self.right_projection(act) * torch.sigmoid(self.right_gate(act))
            gate_values = self.gating_linear(act)
        out = torch.einsum('bikc,bjkc->bijc', left_proj, right_proj)
        out = self.center_layer_norm(out)
        out = self.output_projection(out)
        out = out * torch.sigmoid(gate_values)
        return self.dropout(out)

class TriangleMultiplicationIngoing(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_c = config['norm_channel']
        mid_c = config['num_intermediate_channel']
        dropout_rate = config['dropout_rate']
        orientation = config['orientation']
        self.fused_projection = config['fuse_projection_weights']

        if config['fuse_projection_weights']:
            self.left_norm_input = nn.LayerNorm(in_c)
            self.projection = nn.Linear(in_c, 2*mid_c)
            self.gate = nn.Linear(in_c, 2*mid_c)
        else:
            self.layer_norm_input = nn.LayerNorm(in_c)
            self.left_projection = nn.Linear(in_c, mid_c)
            self.right_projection = nn.Linear(in_c, mid_c)
            self.left_gate = nn.Linear(in_c, mid_c)
            self.right_gate = nn.Linear(in_c, mid_c)
        self.center_layer_norm = nn.LayerNorm(mid_c)
        self.output_projection = nn.Linear(mid_c, in_c)
        self.gating_linear = nn.Linear(in_c, in_c)
        if np.isclose(dropout_rate, 0):
            self.dropout = nn.Identity()
        else:
            self.dropout = AxisWiseDropout(dropout_rate, orientation)

    def forward(self, act, mask): 
        if self.fused_projection:
            left_act = self.left_norm_input(act)
            mask = mask[..., None]
            proj_act = mask*self.projection(left_act) * torch.sigmoid(self.gate(left_act))
            left_proj = proj_act[...,:self.mid_c]
            right_proj = proj_act[...,self.mid_c:]
            gate_values = self.gating_linear(left_act)
        else:
            act = self.layer_norm_input(act)
            mask = mask[..., None]
            left_proj = mask*self.left_projection(act) * torch.sigmoid(self.left_gate(act))
            right_proj = mask*self.right_projection(act) * torch.sigmoid(self.right_gate(act))
            gate_values = self.gating_linear(act)
        out = torch.einsum('bkjc,bkic->bijc', left_proj, right_proj)
        out = self.center_layer_norm(out)
        out = self.output_projection(out)
        out = out * torch.sigmoid(gate_values)
        return self.dropout(out)

class TriangleAttentionStartingNode(nn.Module):
    def __init__(self, config):
        super().__init__()
        attn_num_c = config['attention_channel']
        num_heads = config['num_head']
        num_in_c = config['norm_channel']
        dropout_rate = config['dropout_rate']
        orientation = config['orientation']
        self.attn_num_c = attn_num_c
        self.num_heads = num_heads

        self.query_norm = nn.LayerNorm(num_in_c)
        self.feat_2d_weights = nn.Linear(num_in_c, num_heads, bias=False)

        self.mha = Attention(
            num_in_c, num_in_c, num_in_c, attn_num_c, num_heads
        )

        if np.isclose(dropout_rate, 0):
            self.dropout = nn.Identity()
        else:
            self.dropout = AxisWiseDropout(dropout_rate, orientation)
    
    @torch.jit.ignore
    def _chunk(self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(self, act, mask):
        act = self.query_norm(act)
        bias = (1e9 * (mask - 1.))[...,:, None, None, :]
        nonbatched_bias = self.feat_2d_weights(act)
        nonbatched_bias = nonbatched_bias.permute(0, 3, 1, 2)
        nonbatched_bias = nonbatched_bias.unsqueeze(-4)
        biases = [bias, nonbatched_bias]
        out = self._chunk(
                act,
                biases,
                256,
                use_memory_efficient_kernel=False,
                use_lma=False,
                inplace_safe=False,
            )

        return self.dropout(out)

class TriangleAttentionEndingNode(nn.Module):
    def __init__(self, config):
        super().__init__()
        attention_num_c = config['attention_channel']
        num_heads = config['num_head']
        num_in_c = config['norm_channel']
        dropout_rate = config['dropout_rate']
        orientation = config['orientation']

        self.attention_num_c = attention_num_c
        self.num_heads = num_heads

        self.query_norm = nn.LayerNorm(num_in_c)
        self.feat_2d_weights = nn.Linear(num_in_c, num_heads, bias=False)

        self.mha = Attention(
            num_in_c, num_in_c, num_in_c, attention_num_c, num_heads
        )

        if np.isclose(dropout_rate, 0):
            self.dropout = nn.Identity()
        else:
            self.dropout = AxisWiseDropout(dropout_rate, orientation)

    @torch.jit.ignore
    def _chunk(self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(self, act, mask):
        act = act.transpose(-2,-3)
        act = self.query_norm(act)
        mask = mask.transpose(-1,-2)
        bias = (1e9 * (mask - 1.))[...,:, None, None, :]
        nonbatched_bias = self.feat_2d_weights(act)
        nonbatched_bias = nonbatched_bias.permute(0, 3, 1, 2)
        nonbatched_bias = nonbatched_bias.unsqueeze(-4)
        biases = [bias, nonbatched_bias]
        out = self._chunk(
                act,
                biases,
                256,
                use_memory_efficient_kernel=False,
                use_lma=False,
                inplace_safe=False,
            )

        out = out.transpose(-2,-3)

        return self.dropout(out)

class RecyclingEmbedder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prev_pos_linear = nn.Linear(config['model']['embeddings_and_evoformer']['prev_pos']['num_bins'], 
                                         config['model']['embeddings_and_evoformer']['pair_channel'])
        self.max_bin = config['model']['embeddings_and_evoformer']['prev_pos']['max_bin']
        self.min_bin = config['model']['embeddings_and_evoformer']['prev_pos']['min_bin']
        self.num_bins = config['model']['embeddings_and_evoformer']['prev_pos']['num_bins']
        self.config = config
        self.prev_pair_norm = nn.LayerNorm(config['model']['embeddings_and_evoformer']['pair_channel'])
        self.prev_msa_first_row_norm = nn.LayerNorm(config['model']['embeddings_and_evoformer']['msa_channel'])
        self.position_activations = nn.Linear(config['rel_feat'], config['model']['embeddings_and_evoformer']['pair_channel'])

    def _relative_encoding(self, batch):
        c = self.config['model']['embeddings_and_evoformer']
        rel_feats = []
        pos = batch['residue_index']
        asym_id = batch['asym_id']
        asym_id_same = torch.eq(asym_id[..., None], asym_id[...,None, :])
        offset = pos[..., None] - pos[...,None, :]

        clipped_offset = torch.clip(offset + c['max_relative_idx'], min=0, max=2 * c['max_relative_idx'])

        if c['use_chain_relative']:
            final_offset = torch.where(asym_id_same, clipped_offset,
                               (2 * c['max_relative_idx'] + 1) *
                               torch.ones_like(clipped_offset))

            rel_pos = torch.nn.functional.one_hot(final_offset.long(), 2 * c['max_relative_idx'] + 2)

            rel_feats.append(rel_pos)

            entity_id = batch['entity_id']
            entity_id_same = torch.eq(entity_id[..., None], entity_id[...,None, :])
            rel_feats.append(entity_id_same.type(rel_pos.dtype)[..., None])

            sym_id = batch['sym_id']
            rel_sym_id = sym_id[..., None] - sym_id[...,None, :]

            max_rel_chain = c['max_relative_chain']

            clipped_rel_chain = torch.clip(rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain)

            final_rel_chain = torch.where(entity_id_same, clipped_rel_chain,
                                  (2 * max_rel_chain + 1) *
                                  torch.ones_like(clipped_rel_chain))
            rel_chain = torch.nn.functional.one_hot(final_rel_chain.long(), 2 * c['max_relative_chain'] + 2)

            rel_feats.append(rel_chain)

        else:
            rel_pos = torch.nn.functional.one_hot(clipped_offset.long(), 2 * c['max_relative_idx'] + 1)
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, -1)
        return rel_feat

    def forward(self, batch, recycle):
        prev_pseudo_beta = pseudo_beta_fn(batch['aatype'], recycle['prev_pos'], None)
        dgram = torch.sum((prev_pseudo_beta[..., None, :] - prev_pseudo_beta[..., None, :, :]) ** 2, dim=-1, keepdim=True)
        lower = torch.linspace(self.min_bin, self.max_bin, self.num_bins, device=prev_pseudo_beta.device) ** 2
        upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
        dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
        prev_pos_linear = self.prev_pos_linear(dgram)
        pair_activation_update = prev_pos_linear + self.prev_pair_norm(recycle['prev_pair'])
        rel_feat = self._relative_encoding(batch)
        pair_activation_update = pair_activation_update +  self.position_activations(rel_feat.float())
        prev_msa_first_row = self.prev_msa_first_row_norm(recycle['prev_msa_first_row'])
        del dgram, prev_pseudo_beta

        return prev_msa_first_row, pair_activation_update

class FragExtraStackIteration(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(config['msa_row_attention_with_pair_bias'], global_config)
        self.ExtraColumnGlobalAttention = ExtraColumnGlobalAttention(config['msa_column_attention'], global_config)
        self.RecTransition = Transition(config['msa_transition'])
        self.OuterProductMean = OuterProductMean(config['outer_product_mean'])
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['triangle_multiplication_outgoing'])
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['triangle_multiplication_incoming'])
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['triangle_attention_starting_node'])
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['triangle_attention_ending_node'])
        self.PairTransition = Transition(config['pair_transition'])

    def forward(self, msa_act, pair_act, msa_mask, pair_mask):
        pair_act = pair_act + self.OuterProductMean(msa_act, msa_mask)
        msa_act = msa_act + self.RowAttentionWithPairBias(msa_act, pair_act, msa_mask)
        msa_act = msa_act + self.ExtraColumnGlobalAttention(msa_act, msa_mask)
        msa_act = msa_act + self.RecTransition(msa_act)
        pair_act = pair_act + self.TriangleMultiplicationOutgoing(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleMultiplicationIngoing(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleAttentionStartingNode(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleAttentionEndingNode(pair_act, pair_mask)
        pair_act = pair_act + self.PairTransition(pair_act)
        return msa_act, pair_act


class FragExtraStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([FragExtraStackIteration(
            config['model']['embeddings_and_evoformer']['extra_msa'], config['model']['embeddings_and_evoformer']) \
            for _ in range(config['model']['embeddings_and_evoformer']['extra_msa_stack_num_block']
        )])
    
    def forward(self, msa_act, pair_act, extra_mask_msa, extra_mask_pair):
        for l in self.layers:
            msa_act, pair_act = checkpoint(l, msa_act, pair_act, extra_mask_msa, extra_mask_pair, use_reentrant=False)
        return pair_act

class EvoformerIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(config['msa_row_attention_with_pair_bias'], global_config)
        self.LigColumnAttention = LigColumnAttention(config['msa_column_attention'], global_config)
        self.RecTransition = Transition(config['msa_transition'])
        self.OuterProductMean = OuterProductMean(config['outer_product_mean'])
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['triangle_multiplication_outgoing'])
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['triangle_multiplication_incoming'])
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['triangle_attention_starting_node'])
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['triangle_attention_ending_node'])
        self.PairTransition = Transition(config['pair_transition'])

    def forward(self, msa_act, pair_act, msa_mask, pair_mask):
        pair_act = pair_act + self.OuterProductMean(msa_act, msa_mask)
        msa_act = msa_act + self.RowAttentionWithPairBias(msa_act, pair_act, msa_mask)
        msa_act = msa_act + self.LigColumnAttention(msa_act, msa_mask)
        msa_act = msa_act + self.RecTransition(msa_act)
        pair_act = pair_act + self.TriangleMultiplicationOutgoing(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleMultiplicationIngoing(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleAttentionStartingNode(pair_act, pair_mask)
        pair_act = pair_act + self.TriangleAttentionEndingNode(pair_act, pair_mask)
        pair_act = pair_act + self.PairTransition(pair_act)
        return msa_act, pair_act

class TemplateEmbeddingIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['triangle_multiplication_outgoing'])
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['triangle_multiplication_incoming'])
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['triangle_attention_starting_node'])
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['triangle_attention_ending_node'])
        self.PairTransition = Transition(config['pair_transition'])

    def forward(self, act, pair_mask):
        act = act + self.TriangleMultiplicationOutgoing(act, pair_mask)
        act = act + self.TriangleMultiplicationIngoing(act, pair_mask)
        act = act + self.TriangleAttentionStartingNode(act, pair_mask)
        act = act + self.TriangleAttentionEndingNode(act, pair_mask)
        act = act + self.PairTransition(act)
        return act


class SingleTemplateEmbedding(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.query_embedding_norm = nn.LayerNorm(global_config['model']['embeddings_and_evoformer']['pair_channel'])
        self.TemplateEmbeddingIteration = nn.ModuleList([TemplateEmbeddingIteration(
            config['template_pair_stack'], global_config) for _ in range(config['template_pair_stack']['num_block']
        )])
        self.output_layer_norm = nn.LayerNorm(config['num_channels'])
        self.template_pair_emb_0 = nn.Linear(config['dgram_features']['num_bins'], config['num_channels'])
        self.template_pair_emb_1 = nn.Linear(1, config['num_channels'])
        self.template_pair_emb_2 = nn.Linear(22, config['num_channels'])
        self.template_pair_emb_3 = nn.Linear(22, config['num_channels'])
        self.template_pair_emb_4 = nn.Linear(1, config['num_channels'])
        self.template_pair_emb_5 = nn.Linear(1, config['num_channels'])
        self.template_pair_emb_6 = nn.Linear(1, config['num_channels'])
        self.template_pair_emb_7 = nn.Linear(1, config['num_channels'])
        self.template_pair_emb_8 = nn.Linear(global_config['model']['embeddings_and_evoformer']['pair_channel'], 
                                             config['num_channels'])


        self.max_bin = config['dgram_features']['max_bin']
        self.min_bin = config['dgram_features']['min_bin']
        self.num_bins = config['dgram_features']['num_bins']

    def forward(self, query_embedding, template_aatype,
               template_all_atom_positions, template_all_atom_mask, padding_mask_2d, multichain_mask_2d):
        template_positions, pseudo_beta_mask = pseudo_beta_fn(template_aatype, template_all_atom_positions, 
                                                              template_all_atom_mask)
        pseudo_beta_mask_2d = (pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :])
        pseudo_beta_mask_2d *= multichain_mask_2d

        dgram = torch.sum((template_positions[..., None, :] - template_positions[..., None, :, :]) ** 2, dim=-1, keepdim=True)
        lower = torch.linspace(self.min_bin, self.max_bin, self.num_bins, device=template_positions.device) ** 2
        upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
        template_dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
        template_dgram *= pseudo_beta_mask_2d[..., None]
        template_dgram = template_dgram.type(query_embedding.dtype)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.type(query_embedding.dtype)

        aatype = nn.functional.one_hot(template_aatype,22).type(query_embedding.dtype)

        raw_atom_pos = template_all_atom_positions
        n, ca, c = [rc.atom_order[a] for a in ["N", "CA", "C"]]
        rigids = rigid.Rigid.make_transform_from_reference(
            n_xyz=raw_atom_pos[..., n, :],
            ca_xyz=raw_atom_pos[..., ca, :],
            c_xyz=raw_atom_pos[..., c, :],
            eps=1e-20,
            )
        backbone_mask = (template_all_atom_mask[:, n] * template_all_atom_mask[:, ca] * template_all_atom_mask[:, c]).float()
        points = rigids.get_trans()[..., None, :, :]
        rigid_vec = rigids[..., None].invert_apply(points)
        inv_distance_scalar = torch.rsqrt(1e-20 + torch.sum(rigid_vec ** 2, dim=-1))
        backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
        backbone_mask_2d *= multichain_mask_2d
        unit_vector = rigid_vec * inv_distance_scalar[..., None]
        unit_vector = unit_vector * backbone_mask_2d[..., None]
        unbind_unit_vector = torch.unbind(unit_vector[..., None, :], dim=-1)
        
        query_embedding = self.query_embedding_norm(query_embedding)
        
        act = self.template_pair_emb_0(template_dgram)
        act = act + self.template_pair_emb_1(pseudo_beta_mask_2d[..., None])
        act = act + self.template_pair_emb_2(aatype[None, :, :])
        act = act + self.template_pair_emb_3(aatype[:, None, :])
        act = act + self.template_pair_emb_4(unbind_unit_vector[0])
        act = act + self.template_pair_emb_5(unbind_unit_vector[1])
        act = act + self.template_pair_emb_6(unbind_unit_vector[2])
        act = act + self.template_pair_emb_7(backbone_mask_2d[..., None])
        act = act + self.template_pair_emb_8(query_embedding)
        act = torch.unsqueeze(act, dim=0)
        for iter_temp in self.TemplateEmbeddingIteration:
            act = checkpoint(iter_temp, act, torch.unsqueeze(padding_mask_2d, dim=0), use_reentrant=False)
        act = torch.squeeze(act)
        act = self.output_layer_norm(act)
        return act

class TemplateEmbedding(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.SingleTemplateEmbedding = SingleTemplateEmbedding(config, global_config)
        self.relu = nn.ReLU()
        self.output_linear = nn.Linear(config['num_channels'], global_config['model']['embeddings_and_evoformer']['pair_channel'])
        self.num_channels = config['num_channels']

    def forward(self, query_embedding, template_batch, padding_mask_2d,multichain_mask_2d):
        num_templates = template_batch['template_aatype'].shape[0]
        num_res, _, query_num_channels = query_embedding.shape
        scan_init = torch.zeros((num_res, num_res, self.num_channels), device=query_embedding.device, dtype=query_embedding.dtype)
        for i in range(num_templates):
            partial_emb = self.SingleTemplateEmbedding(
                query_embedding, template_batch['template_aatype'][i],
                template_batch['template_all_atom_positions'][i],
                template_batch['template_all_atom_mask'][i],
                padding_mask_2d, multichain_mask_2d
            )
            scan_init = scan_init + partial_emb
        embedding = scan_init / num_templates
        embedding = self.relu(embedding)
        embedding = self.output_linear(embedding)
        embedding = torch.unsqueeze(embedding, dim=0)
        return embedding

class TemplateEmbedding1D(torch.nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.template_single_embedding = nn.Linear(34, global_config['model']['embeddings_and_evoformer']['msa_channel'])
        self.template_projection = nn.Linear(global_config['model']['embeddings_and_evoformer']['msa_channel'], 
                                             global_config['model']['embeddings_and_evoformer']['msa_channel'])
        self.relu = nn.ReLU()

    def forward(self, batch):
        aatype_one_hot = nn.functional.one_hot(batch['template_aatype'], 22)
        
        num_templates = batch['template_aatype'].shape[1]
        all_chi_angles = []
        all_chi_masks = []
        for i in range(num_templates):
            template_chi_angles, template_chi_mask = all_atom_multimer.compute_chi_angles(
            batch['template_all_atom_positions'][0][i, :, :, :],
            batch['template_all_atom_mask'][0][i, :, :],
            batch['template_aatype'][0][i, :])
            all_chi_angles.append(template_chi_angles)
            all_chi_masks.append(template_chi_mask)
        chi_angles = torch.stack(all_chi_angles, dim=0)
        chi_angles = torch.unsqueeze(chi_angles, dim=0)
        chi_mask = torch.stack(all_chi_masks, dim=0)
        chi_mask = torch.unsqueeze(chi_mask, dim=0)
        
        template_features = torch.cat((aatype_one_hot, torch.sin(chi_angles)*chi_mask, 
                                       torch.cos(chi_angles)*chi_mask, chi_mask), dim=-1).type(torch.float32)
        template_mask = chi_mask[...,0]

        template_activations = self.template_single_embedding(template_features)
        template_activations = self.relu(template_activations)
        template_activations = self.template_projection(template_activations)

        return template_activations, template_mask

class InputEmbedding(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.preprocessing_1d = nn.Linear(
            global_config['aatype'], global_config['model']['embeddings_and_evoformer']['msa_channel']
        )
        self.left_single = nn.Linear(
            global_config['aatype'], global_config['model']['embeddings_and_evoformer']['pair_channel']
        )
        self.right_single = nn.Linear(
            global_config['aatype'], global_config['model']['embeddings_and_evoformer']['pair_channel']
        )
        self.preprocess_msa = nn.Linear(
            global_config['msa'], global_config['model']['embeddings_and_evoformer']['msa_channel']
            )
        self.max_seq = global_config['model']['embeddings_and_evoformer']['num_msa']
        self.msa_channel = global_config['model']['embeddings_and_evoformer']['msa_channel']
        self.pair_channel = global_config['model']['embeddings_and_evoformer']['pair_channel']
        self.num_extra_msa = global_config['model']['embeddings_and_evoformer']['num_extra_msa']
        self.global_config = global_config
        
        self.TemplateEmbedding = TemplateEmbedding(
            global_config['model']['embeddings_and_evoformer']['template'], global_config
        )
        self.RecyclingEmbedder = RecyclingEmbedder(global_config)
        self.extra_msa_activations = nn.Linear(
            global_config['extra_msa_act'], global_config['model']['embeddings_and_evoformer']['extra_msa_channel']
        )
        self.FragExtraStack = FragExtraStack(global_config)

    def forward(self, batch, recycle):
        num_batch, num_res = batch['aatype'].shape[0], batch['aatype'].shape[1]
        target_feat = nn.functional.one_hot(batch['aatype'].long(), 21).float()
        preprocessed_1d = self.preprocessing_1d(target_feat)
        left_single = self.left_single(target_feat)
        right_single = self.right_single(target_feat)
        pair_activations = left_single.unsqueeze(2) + right_single.unsqueeze(1)
        preprocess_msa = self.preprocess_msa(batch['msa_feat'])
        msa_activations = preprocess_msa + preprocessed_1d
        mask_2d = batch['seq_mask'][..., None] * batch['seq_mask'][...,None, :]
        mask_2d = mask_2d.type(torch.float32)
        if self.global_config['recycle'] and recycle == None:
            recycle = {
                'prev_pos': torch.zeros(num_batch, num_res, 37, 3).to(
                    batch['aatype'].device
                ),
                'prev_msa_first_row': torch.zeros(
                    num_batch, num_res, self.msa_channel
                ).to(batch['aatype'].device),
                'prev_pair': torch.zeros(
                    num_batch, num_res, num_res, self.pair_channel
                ).to(batch['aatype'].device)
            }

        if(recycle is not None):
            prev_msa_first_row, pair_activation_update = self.RecyclingEmbedder(batch, recycle)
            pair_activations = pair_activations + pair_activation_update
            msa_activations[:,0] += prev_msa_first_row
            del recycle
        
        if(self.global_config['model']['embeddings_and_evoformer']['template']['enabled']):
            template_batch = {
                'template_aatype': batch['template_aatype'][0],
                'template_all_atom_positions': batch['template_all_atom_positions'][0],
                'template_all_atom_mask': batch['template_all_atom_mask'][0]
            }
            multichain_mask = batch['asym_id'][..., None] == batch['asym_id'][:, None, ...]
            template_act = self.TemplateEmbedding(pair_activations[0], template_batch, mask_2d[0], multichain_mask[0])
            pair_activations = pair_activations + template_act
            del template_batch

        extra_msa_activations = self.extra_msa_activations(batch['extra_msa_feat'])
        if self.num_extra_msa > 0:
            pair_activations = self.FragExtraStack(extra_msa_activations, pair_activations, 
                                                   batch['extra_msa_mask'].type(torch.float32), mask_2d)
        msa_mask = batch['msa_mask']
        del target_feat
        return msa_activations, pair_activations, msa_mask, mask_2d

class MaskedMsaHead(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.logits = nn.Linear(global_config['model']['embeddings_and_evoformer']['msa_channel'], 22)

    def forward(self, representations):
        return self.logits(representations['msa'])

class ExperimentallyResolvedHead(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.logits = nn.Linear(global_config['model']['embeddings_and_evoformer']['seq_channel'], 37)

    def forward(self, representations):
        return self.logits(representations['single'])


class PredictedAlignedError(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.logits = nn.Linear(global_config['model']['embeddings_and_evoformer']['pair_channel'], 
                                global_config['model']['heads']['predicted_aligned_error']['num_bins'])
        self.max_error_bin = global_config['model']['heads']['predicted_aligned_error']['max_error_bin']
        self.num_bins = global_config['model']['heads']['predicted_aligned_error']['num_bins']

    def forward(self, representations):
        act = representations['pair']
        logits = self.logits(act)
        breaks = torch.linspace(0., self.max_error_bin, self.num_bins - 1, device=act.device)
        return logits, breaks

class PredictedLddt(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.input_layer_norm = nn.LayerNorm(global_config['model']['embeddings_and_evoformer']['seq_channel'])
        self.act_0 = nn.Linear(global_config['model']['embeddings_and_evoformer']['seq_channel'], 
                               global_config['model']['heads']['predicted_lddt']['num_channels'])
        self.act_1 = nn.Linear(global_config['model']['heads']['predicted_lddt']['num_channels'], 
                               global_config['model']['heads']['predicted_lddt']['num_channels'])
        self.logits = nn.Linear(global_config['model']['heads']['predicted_lddt']['num_channels'], 
                                global_config['model']['heads']['predicted_lddt']['num_bins'])

    def forward(self, representations):
        act = representations['structure_module']
        act = self.input_layer_norm(act)
        act = self.act_0(act).relu_()
        act = self.act_1(act).relu_()
        logits = self.logits(act)
        return logits

class Distogram(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.half_logits = nn.Linear(global_config['model']['embeddings_and_evoformer']['pair_channel'], 
                                     global_config['model']['heads']['distogram']['num_bins'])
        self.first_break = global_config['model']['heads']['distogram']['first_break']
        self.last_break = global_config['model']['heads']['distogram']['last_break']
        self.num_bins = global_config['model']['heads']['distogram']['num_bins']
    
    def forward(self, representations):
        pair = representations['pair']
        half_logits = self.half_logits(pair)
        logits = half_logits + half_logits.transpose(-2, -3)
        breaks = torch.linspace(self.first_break, self.last_break, self.num_bins - 1, device=pair.device)
        return logits, breaks


def preprocess_batch_msa(batch, config):
    config = config['model']['embeddings_and_evoformer']
    batch['msa_profile'] = make_msa_profile(batch)
    batch = sample_msa(batch, config['num_msa'])
    batch = make_masked_msa(batch, config['masked_msa'])
    batch['cluster_profile'], batch['cluster_deletion_mean'] = nearest_neighbor_clusters(batch)
    batch['msa_feat'] = create_msa_feat(batch)
    batch['extra_msa_feat'], batch['extra_msa_mask'] = create_extra_msa_feature(
        batch, config['num_extra_msa']
    )
    return batch


class AlphaFoldIteration(nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.InputEmbedder = InputEmbedding(global_config)
        self.TemplateEmbedding1D = TemplateEmbedding1D(global_config)
        self.Evoformer = nn.ModuleList([EvoformerIteration(
            global_config['model']['embeddings_and_evoformer']['evoformer'], global_config['model']['embeddings_and_evoformer']) \
            for _ in range(global_config['model']['embeddings_and_evoformer']['evoformer_num_block']
        )])
        self.EvoformerExtractSingleRec = nn.Linear(global_config['model']['embeddings_and_evoformer']['msa_channel'], 
                                                   global_config['model']['embeddings_and_evoformer']['seq_channel'])
        self.StructureModule = structure_multimer.StructureModule(global_config['model']['heads']['structure_module'], 
                                                                  global_config['model']['embeddings_and_evoformer'])
        self.Distogram = Distogram(global_config)
        self.PredictedLddt = PredictedLddt(global_config)
        self.PredictedAlignedError = PredictedAlignedError(global_config)
        self.ExperimentallyResolvedHead = ExperimentallyResolvedHead(global_config)
        self.MaskedMsaHead = MaskedMsaHead(global_config)

        self.global_config = global_config

    def forward(self, batch, recycles):
        out, m_1_prev, z_prev, x_prev = self.iteration(batch, recycles)
        
        recycles = {
            'prev_msa_first_row': m_1_prev,
            'prev_pair': z_prev,
            'prev_pos': x_prev
        }

        out = self.run_heads(batch, out)

        return out, recycles

    def iteration(self, batch, recycle=None):
        # Embed features into MSA stack and pair representation
        msa_activations, pair_activations, msa_mask, pair_mask = self.InputEmbedder(batch, recycle=recycle)
        num_msa_seq = msa_activations.shape[1]
        if(self.global_config['model']['embeddings_and_evoformer']['template']['enabled']):
            template_features, template_masks = self.TemplateEmbedding1D(batch)
            msa_activations = torch.cat((msa_activations, template_features), dim=1).type(torch.float32)
            msa_mask = torch.cat((msa_mask, template_masks), dim=1).type(torch.float32)
            del template_features

        # Evoformer blocks
        msa_activations, pair_activations = self.run_evoblocks(
            msa_activations, pair_activations, msa_mask, pair_mask, batch
        )

        representations = {
            'msa': msa_activations[:, :num_msa_seq],
            'pair': pair_activations
        }

        # Compute single representation from first MSA row
        single_activations = self.EvoformerExtractSingleRec(msa_activations[:,0])
        representations['single'] = single_activations

        # Structure module
        struct_out = self.StructureModule(
            single_activations, pair_activations, batch
        )
        representations['structure_module'] = struct_out['act']

        # Final coordinates for current iteration
        atom14_pred_positions = struct_out['atom_pos'][-1]
        atom37_pred_positions = all_atom_multimer.atom14_to_atom37(
            atom14_pred_positions.squeeze(0), batch['aatype'].squeeze(0).long()
        )
        atom37_mask = all_atom_multimer.atom_37_mask(batch['aatype'][0].long())

        # Form output
        out = {}
        out['representations'] = representations
        out['structure_module'] = struct_out
        out['final_atom14_positions'] = atom14_pred_positions.squeeze(0)
        out['final_atom_positions'] = atom37_pred_positions
        out['final_atom_mask'] = atom37_mask

        # Form representaions and coordinates for recycling
        m_1_prev = msa_activations[:,0]
        z_prev = pair_activations
        x_prev = atom37_pred_positions.unsqueeze(0)

        return out, m_1_prev, z_prev, x_prev

    def run_evoblocks(self, msa_activations, pair_activations, msa_mask, 
                      pair_mask, batch):
        for evo_iter in self.Evoformer:
            # Run individual Evoformer blocks
            msa_activations, pair_activations = evo_iter(
                msa_activations, pair_activations, msa_mask, pair_mask
            )

        return msa_activations, pair_activations

    def run_heads(self, batch, out):
        distogram_logits, distogram_bin_edges = self.Distogram(out['representations'])
        pred_lddt = self.PredictedLddt(out['representations'])
        pae_logits, pae_breaks = self.PredictedAlignedError(out['representations'])
        resolved_logits = self.ExperimentallyResolvedHead(out['representations'])
        masked_msa_logits = self.MaskedMsaHead(out['representations'])

        out['distogram'] = {}
        out['distogram']['logits'] = distogram_logits
        out['distogram']['bin_edges'] = distogram_bin_edges
        out['predicted_lddt'] = {}
        out['predicted_lddt']['logits'] = pred_lddt
        out['predicted_aligned_error'] = {}
        out['predicted_aligned_error']['logits'] = pae_logits
        out['predicted_aligned_error']['breaks'] = pae_breaks
        out['predicted_aligned_error']['asym_id'] = batch['asym_id'][0].clone()
        out['experimentally_resolved'] = resolved_logits
        out['masked_msa'] = masked_msa_logits

        return out

class AlphaFold(nn.Module):
    def __init__(self, net_iteration: nn.Module, global_config: dict):
        super().__init__()
        self.net_iteration = net_iteration
        self.global_config = global_config

    def forward(self, init_batch, num_recycle=None):
        if num_recycle is None:
            num_recycle = self.global_config['model']['num_recycle']
        
        out = self.run_recycles(init_batch, num_recycle)

        return out

    def run_recycles(self, init_batch, num_recycle):
        recycles = None
        
        batch = preprocess_batch_msa(
            init_batch, 
            self.global_config
        )
        for recycle_iter in range(num_recycle):
            out, recycles = self.net_iteration(batch, recycles)
            if recycle_iter < (num_recycle - 1):
                if self.global_config['model']['resample_msa_in_recycling']:
                    batch = preprocess_batch_msa(
                        init_batch, 
                        self.global_config
                    )

        return out

def create_alphafold(config):
    net_iteration = AlphaFoldIteration(config)
    model = AlphaFold(net_iteration, config)
    return model

if __name__ == "__main__":
    from copy import deepcopy
    from tuned_abs.configs import AF_M_V2_CONFIG
    from data.features import generate_random_example
    config = deepcopy(AF_M_V2_CONFIG)
    from .tensor_utils import to_device

    example = generate_random_example(config)
    batch = torch.utils.data.default_collate([example])
    batch = to_device(batch, device='cuda')

    net_iteration = AlphaFoldIteration(config)
    model = AlphaFold(net_iteration, config)
    model.to('cuda')

    with torch.no_grad():
        out = model(batch, num_recycle=1)
        print(out.keys())
