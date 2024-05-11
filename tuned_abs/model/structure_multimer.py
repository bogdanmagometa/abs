# Copyright 2022 Applied BioComputation Group, Stony Brook University
# Copyright 2021 DeepMind Technologies Limited
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

import numpy as np
import torch
from torch import nn
import math
from torch.utils.checkpoint import checkpoint
from . import all_atom_multimer
from .rigid import Rigid, Rotation

class InvariantPointAttention(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        self.num_head = config['num_head']
        self.num_scalar_qk = config['num_scalar_qk']
        self.num_scalar_v = config['num_scalar_v']
        self.num_point_qk = config['num_point_qk']
        self.num_point_v = config['num_point_v']

        self.num_output_c = config['num_channel']
        self.rep_1d_num_c = config['num_channel']
        self.rep_2d_num_c = global_config['pair_channel']

        self.q = nn.Linear(self.rep_1d_num_c, self.num_scalar_qk* self.num_head, bias=False)
        self.k = nn.Linear(self.rep_1d_num_c, (self.num_scalar_qk) * self.num_head, bias=False)
        self.v = nn.Linear(self.rep_1d_num_c, (self.num_scalar_v) * self.num_head, bias=False)
        self.q_points = nn.Linear(self.rep_1d_num_c, self.num_point_qk * self.num_head * 3)
        self.k_points = nn.Linear(self.rep_1d_num_c, (self.num_point_qk) * self.num_head * 3)
        self.v_points = nn.Linear(self.rep_1d_num_c, (self.num_point_v) * self.num_head * 3)
        self.rr_kqv_2d = nn.Linear(self.rep_2d_num_c, self.num_head)
        self.final_r = nn.Linear(self.num_head * (self.rep_2d_num_c + self.num_scalar_v + 4 * self.num_point_v), 
                                 self.num_output_c)
        self.trainable_w = nn.Parameter(torch.zeros((self.num_head)))
        self.softplus = nn.Softplus()

    def forward(self, act, act_2d, sequence_mask, rigid):

        q_scalar = checkpoint(self.q, act, use_reentrant=False)
        q_scalar = q_scalar.view(*q_scalar.shape[:-1], self.num_head, -1)
        k_scalar = checkpoint(self.k, act, use_reentrant=False)
        k_scalar = k_scalar.view(*k_scalar.shape[:-1], self.num_head, -1)
        v_scalar = checkpoint(self.v, act, use_reentrant=False)
        v_scalar = v_scalar.view(*v_scalar.shape[:-1], self.num_head, -1)

        q_point = checkpoint(self.q_points, act, use_reentrant=False)
        q_point = q_point.view(*q_point.shape[:-1], self.num_head, -1)
        q_point = torch.split(q_point, q_point.shape[-1]//3, dim=-1)
        q_point = torch.stack(q_point, dim=-1)
        q_point = q_point.view(*q_point.shape[:-3], -1, q_point.shape[-1])
        q_point_global = rigid[..., None].apply(q_point)

        k_point = checkpoint(self.k_points, act, use_reentrant=False)
        k_point = k_point.view(*k_point.shape[:-1], self.num_head, -1)
        k_point = torch.split(k_point, k_point.shape[-1]//3, dim=-1)
        k_point = torch.stack(k_point, dim=-1)
        k_point = k_point.view(*k_point.shape[:-3], -1, k_point.shape[-1])
        k_point_global = rigid[..., None].apply(k_point)

        v_point = checkpoint(self.v_points, act, use_reentrant=False)
        v_point = v_point.view(*v_point.shape[:-1], self.num_head, -1)
        v_point = torch.split(v_point, v_point.shape[-1]//3, dim=-1)
        v_point = torch.stack(v_point, dim=-1)
        v_point = v_point.view(*v_point.shape[:-3], -1, v_point.shape[-1])
        v_point_global = rigid[..., None].apply(v_point)

        attn_logits = 0.
        num_point_qk = self.num_point_qk
        point_variance = max(num_point_qk, 1) * 9. / 2
        point_weights = math.sqrt(1.0 / point_variance)
        trainable_point_weights = self.softplus(self.trainable_w)
        point_weights = point_weights * torch.unsqueeze(trainable_point_weights, dim=-1)
        dist2 = torch.sum(torch.square(torch.unsqueeze(q_point_global, -3) - torch.unsqueeze(k_point_global, -4)), -1)
        dist2 = dist2.view(*dist2.shape[:-1], self.num_head, -1)
        attn_qk_point = -0.5* torch.sum(point_weights[None, None, None,...]* dist2, -1)
        attn_logits += attn_qk_point
        num_scalar_qk = self.num_scalar_qk
        scalar_variance = max(num_scalar_qk, 1) * 1.
        scalar_weights = math.sqrt(1.0 / scalar_variance)
        q_scalar *= scalar_weights

        attn_logits += torch.einsum('...qhc,...khc->...qkh', q_scalar, k_scalar)
        
        attention_2d = checkpoint(self.rr_kqv_2d, act_2d, use_reentrant=False)
        attn_logits += attention_2d

        mask_2d = sequence_mask * torch.transpose(sequence_mask, -1, -2)
        attn_logits -= 1e5 * (1. - mask_2d[..., None])
        attn_logits *= math.sqrt(1. / 3)
        attn = torch.softmax(attn_logits, -2)
        result_scalar = torch.einsum('...qkh, ...khc->...qhc', attn, v_scalar)
        v_point_global = v_point_global.view(*v_point_global.shape[:-2], self.num_head, -1, 3)
        result_point_global = torch.sum(attn[..., None, None] * v_point_global[:, None, ...], -4)
        output_features = []
        result_scalar = result_scalar.reshape(*result_scalar.shape[:-2], -1)
        output_features.append(result_scalar)
        result_point_global = result_point_global.view(*result_point_global.shape[:-3], -1, 3)
        result_point_local = rigid[..., None].invert_apply(result_point_global)
        result_point_local_x, result_point_local_y, result_point_local_z = torch.split(result_point_local, 1, dim=-1)

        output_features.extend([torch.squeeze(result_point_local_x, -1), torch.squeeze(result_point_local_y, -1), 
                                torch.squeeze(result_point_local_z, -1)])
        point_norms = torch.linalg.norm(result_point_local, dim=(-1))
        output_features.append(point_norms)
        result_attention_over_2d = torch.einsum('...ijh, ...ijc->...ihc', attn, act_2d)
        result_attention_over_2d = result_attention_over_2d.reshape(*result_attention_over_2d.shape[:-2], -1)
        output_features.append(result_attention_over_2d)
        final_act = torch.cat(output_features, -1)

        out = self.final_r(final_act)

        return out

class PredictSidechains(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        num_in_c = config['num_channel']
        num_c = config['sidechain']['num_channel']
        self.num_torsions = 7

        self.s_cur = nn.Linear(num_in_c, num_c)
        self.s_ini = nn.Linear(num_in_c, num_c)
        self.relu = nn.ReLU()

        self.res1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_c, num_c),
            nn.ReLU(),
            nn.Linear(num_c, num_c)
        )

        self.res2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_c, num_c),
            nn.ReLU(),
            nn.Linear(num_c, num_c)
        )

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_c, self.num_torsions * 2)
        )

    def forward(self, s_cur, s_ini):
        a = self.s_cur(self.relu(s_cur.clone())) + self.s_ini(self.relu(s_ini))
        a += self.res1(a.clone())
        a += self.res2(a.clone())
        unnormalized_angles = self.final(a).reshape(*a.shape[:-1], self.num_torsions, 2)
        norm = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_angles ** 2, dim=-1, keepdim=True),
                min=1e-12,
            )
        )
        normalized_angles = unnormalized_angles / norm
        return normalized_angles, unnormalized_angles

class StructureModuleIteration(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.InvariantPointAttention = InvariantPointAttention(config, global_config)
        if np.isclose(config['dropout'], 0.0):
            self.drop = nn.Identity()
        else:
            self.drop = nn.Dropout(config['dropout'])
        self.rec_norm = nn.LayerNorm(config['num_channel'])
        self.rec_norm2 = nn.LayerNorm(config['num_channel'])
        self.position_scale = config['position_scale']

        num_1dc = config['num_channel']
        self.transition_r = nn.Sequential(
            nn.Linear(num_1dc, num_1dc),
            nn.ReLU(),
            nn.Linear(num_1dc, num_1dc),
            nn.ReLU(),
            nn.Linear(num_1dc, num_1dc)
        )

        self.backbone_update = nn.Linear(num_1dc, 6)
        self.PredictSidechains = PredictSidechains(config)

    def forward(self, act, rigid, initial_act, act_2d, aatype, sequence_mask):
        self.InvariantPointAttention.to(act.device)

        # IPA
        act = act.clone()
        rec_1d_update = self.InvariantPointAttention(act, act_2d, sequence_mask, rigid)
        act = act + rec_1d_update
        act = self.rec_norm(self.drop(act))
        input_act = act.clone()
        act = self.transition_r(act)
        act = act + input_act
        act = self.rec_norm2(self.drop(act))

        rigid_flat = self.backbone_update(act)
        rigids = rigid.compose_q_update_vec(rigid_flat)
        norm_sc, unnorm_sc = checkpoint(self.PredictSidechains, act, initial_act, use_reentrant=False)

        bb_to_global = Rigid(
                Rotation(
                    rot_mats=rigids.get_rots().get_rot_mats(),
                    quats=None
                ),
                rigids.get_trans(),
            )

        bb_to_global = bb_to_global.scale_translation(self.position_scale)
        all_frames_to_global = all_atom_multimer.torsion_angles_to_frames(bb_to_global, norm_sc, aatype)
        pred_positions = all_atom_multimer.frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global)
        scaled_rigids = rigids.scale_translation(self.position_scale)
        sc = {
            'angles_sin_cos': norm_sc, 
            'unnormalized_angles_sin_cos': unnorm_sc, 
            'atom_pos': pred_positions, 
            'sc_frames': all_frames_to_global.to_tensor_4x4(), 
            'frames': scaled_rigids.to_tensor_7()
        }

        rigids = rigids.stop_rot_gradient()

        return act, rigids, sc


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict

class StructureModule(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.num_iter = config['num_layer']
        num_1dc = config['num_channel']

        self.StructureModuleIteration = StructureModuleIteration(config, global_config)
        self.layers = [self.StructureModuleIteration for _ in range(self.num_iter)]
        self.single_layer_norm = nn.LayerNorm(num_1dc)
        self.pair_layer_norm = nn.LayerNorm(global_config['pair_channel'])
        self.initial_projection = nn.Linear(num_1dc, num_1dc)

        self.config = config
        self.global_config = global_config

    def forward(self, single_representation, pair_representation, batch):
        sequence_mask = batch['seq_mask'][..., None]
        act = self.single_layer_norm(single_representation)
        initial_act = act.clone()
        act = self.initial_projection(act)
        act_2d = self.pair_layer_norm(pair_representation)
        rigids = Rigid.identity(act.shape[:-1], act.dtype, act.device, False, fmt="quat")

        out = []
        for l in self.layers:
            l.to(act.device)
            act, rigids, sc = l(act, rigids, initial_act, act_2d, batch['aatype'], sequence_mask)
            out.append(sc)
        outputs = dict_multimap(torch.stack, out)
        outputs['act'] = act
        return outputs

