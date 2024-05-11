# Copyright 2021 AlQuraishi Laboratory
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
#
#
# Modifications Copyright 2022 Applied BioComputation Group, Stony Brook University

import numpy as np
import torch
from tuned_abs.common import residue_constants
from .rigid import Rigid, Rotation
from .tensor_utils import batched_gather

def _make_restype_atom14_mask():
  """Mask of which atoms are present for which residue type in atom14."""
  restype_atom14_mask = []

  for rt in residue_constants.restypes:
    atom_names = residue_constants.restype_name_to_atom14_names[
        residue_constants.restype_1to3[rt]]
    restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

  restype_atom14_mask.append([0.] * 14)
  restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)
  return restype_atom14_mask

def _make_restype_atom37_mask():
  """Mask of which atoms are present for which residue type in atom37."""
  # create the corresponding mask
  restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
  for restype, restype_letter in enumerate(residue_constants.restypes):
    restype_name = residue_constants.restype_1to3[restype_letter]
    atom_names = residue_constants.residue_atoms[restype_name]
    for atom_name in atom_names:
      atom_type = residue_constants.atom_order[atom_name]
      restype_atom37_mask[restype, atom_type] = 1
  return restype_atom37_mask

def _make_restype_atom37_to_atom14():
  """Map from atom37 to atom14 per residue type."""
  restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
  for rt in residue_constants.restypes:
    atom_names = residue_constants.restype_name_to_atom14_names[
        residue_constants.restype_1to3[rt]]
    atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
    restype_atom37_to_atom14.append([
        (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
        for name in residue_constants.atom_types
    ])

RESTYPE_ATOM14_MASK = _make_restype_atom14_mask()
RESTYPE_ATOM37_MASK = _make_restype_atom37_mask()
RESTYPE_ATOM37_TO_ATOM14 = _make_restype_atom37_to_atom14()

def get_atom14_mask(aatype):
  return batched_gather(torch.tensor(RESTYPE_ATOM14_MASK), aatype)

def get_atom37_mask(aatype):
  return batched_gather(torch.tensor(RESTYPE_ATOM37_MASK), aatype)

def get_atom37_to_atom14_map(aatype):
  return batched_gather(torch.tensor(RESTYPE_ATOM37_TO_ATOM14), aatype)

def atom14_to_atom37(atom14_data, aatype):
    assert atom14_data.shape[1] == 14
    assert aatype.ndim == 1
    assert aatype.shape[0] == atom14_data.shape[0]

    residx_atom37_to_atom14 = torch.tensor(residue_constants.restype_name_to_atom14_ids, 
                                           device=aatype.device, dtype=aatype.dtype)[aatype]
    atom14_data_flat = atom14_data.reshape(*atom14_data.shape[:2], -1)
    # add 15th field used as placeholder in restype_name_to_atom14_ids
    atom14_data_flat = torch.cat([atom14_data_flat, torch.zeros_like(atom14_data_flat[:, :1])], dim=1)
    out = torch.gather(atom14_data_flat, 1, residx_atom37_to_atom14[..., None].repeat(1, 1, atom14_data_flat.shape[-1]))
    return out.reshape(atom14_data.shape[0], 37, *atom14_data.shape[2:])

def atom37_to_atom14(
    atom37_data: torch.Tensor,  # (N, 37, ...)
    aatype: torch.Tensor,  # (N)
) -> torch.Tensor:  # (N, 14, ...)
    assert atom37_data.shape[1] == 37
    assert aatype.ndim == 1
    assert aatype.shape[0] == atom37_data.shape[0]

    residx_atom14_to_atom37 = torch.tensor(residue_constants.restype_name_to_atom37_ids, 
                                           device=aatype.device, dtype=aatype.dtype)[aatype]
    atom37_data_flat = atom37_data.reshape(*atom37_data.shape[:2], -1)
    atom37_data_flat = torch.cat([atom37_data_flat, torch.zeros_like(atom37_data_flat[:, :1])], dim=1)
    out = torch.gather(atom37_data_flat, 1, 
                       residx_atom14_to_atom37[..., None].repeat(1, 1, atom37_data_flat.shape[-1]))
    return out.reshape(atom37_data.shape[0], 14, *atom37_data.shape[2:])


def atom_37_mask(aatype):
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32, device=aatype.device
    )
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[aatype]
    return residx_atom37_mask

def atom_14_mask(aatype):
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[residue_constants.restype_1to3[rt]]
        restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    restype_atom14_mask.append([0.] * 14)
    restype_atom14_mask = torch.tensor(np.array(restype_atom14_mask, dtype=np.float32), device=aatype.device)[aatype]
    return restype_atom14_mask

def torsion_angles_to_frames(rigid, angle, aatype):
    m = torch.tensor(
                residue_constants.restype_rigid_group_default_frame,
                dtype=angle.dtype,
                device=angle.device,
                requires_grad=False,
            )
    default_frames = m[aatype.long(), ...]
    default_rot = rigid.from_tensor_4x4(default_frames)
    backbone_rot = angle.new_zeros((*((1,) * len(angle.shape[:-1])), 2))
    backbone_rot[..., 1] = 1
    angle = torch.cat([backbone_rot.expand(*angle.shape[:-2], -1, -1), angle], dim=-2)
    all_rots = angle.new_zeros(default_rot.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = angle[..., 1]
    all_rots[..., 1, 2] = -angle[..., 0]
    all_rots[..., 2, 1:] = angle
    all_rots = Rigid(Rotation(rot_mats=all_rots), None)
    all_frames = default_rot.compose(all_rots)
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = rigid[..., None].compose(all_frames_to_bb)
    return all_frames_to_global

def frames_and_literature_positions_to_atom14_pos(aatype,all_frames_to_global):
    residx_to_group_idx = torch.tensor(
                residue_constants.restype_atom14_to_rigid_group,
                device=all_frames_to_global.get_rots().device,
                requires_grad=False,
            )
    group_mask = residx_to_group_idx[aatype.long(), ...]
    group_mask = torch.nn.functional.one_hot(group_mask, num_classes=8)
    map_atoms_to_global = all_frames_to_global[..., None, :] * group_mask

    map_atoms_to_global = map_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    lit_positions = torch.tensor(
                residue_constants.restype_atom14_rigid_group_positions,
                dtype=all_frames_to_global.get_rots().dtype,
                device=all_frames_to_global.get_rots().device,
                requires_grad=False,
            )
    lit_positions = lit_positions[aatype.long(), ...]

    mask = torch.tensor(
                residue_constants.restype_atom14_mask,
                dtype=all_frames_to_global.get_rots().dtype,
                device=all_frames_to_global.get_rots().device,
                requires_grad=False,
            )
    mask = mask[aatype.long(), ...].unsqueeze(-1)
    pred_positions = map_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * mask
    return pred_positions

def get_chi_atom_indices():
  """Returns atom indices needed to compute chi angles for all residue types.

  Returns:
    A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
    in the order specified in residue_constants.restypes + unknown residue type
    at the end. For chi angles which are not defined on the residue, the
    positions indices are by default set to 0.
  """
  chi_atom_indices = []
  for residue_name in residue_constants.restypes:
    residue_name = residue_constants.restype_1to3[residue_name]
    residue_chi_angles = residue_constants.chi_angles_atoms[residue_name]
    atom_indices = []
    for chi_angle in residue_chi_angles:
      atom_indices.append(
          [residue_constants.atom_order[atom] for atom in chi_angle])
    for _ in range(4 - len(atom_indices)):
      atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
    chi_atom_indices.append(atom_indices)

  chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

  return np.asarray(chi_atom_indices)

def compute_chi_angles(positions, mask, aatype):
    assert positions.shape[-2] == residue_constants.atom_type_num
    assert mask.shape[-1] == residue_constants.atom_type_num

    chi_atom_indices = torch.cat((torch.tensor(get_chi_atom_indices(), device=positions.device), 
                                  torch.zeros((1,4,4), device=positions.device, dtype=torch.int)), dim=0)
    
    atom_indices = chi_atom_indices[aatype.long(),...]
    
    atom_indices_flattern = atom_indices.reshape(*atom_indices.shape[:-2], -1)
    positions_unbind = torch.unbind(positions, dim=-1)
    positions_gather = [torch.gather(x, -1, atom_indices_flattern) for x in positions_unbind]
    chi_angle_atoms = [x.reshape(-1, 4, 4, 1) for x in positions_gather]
    chi_angle_atoms = torch.cat(chi_angle_atoms, dim=-1)
    a, b, c, d = [chi_angle_atoms[...,i, :] for i in range(4)]
    v1 = a - b
    v2 = b - c
    v3 = d - c

    c1 = torch.cross(v1, v2, dim=-1)
    c2 = torch.cross(v3, v2, dim=-1)
    c3 = torch.cross(c2, c1, dim=-1)
    
    v2_mag = torch.sqrt(torch.sum(v2**2, dim=-1))
    c3_v2 = torch.sum(c3*v2, dim=-1)
    c1_c2 = torch.sum(c1*c2, dim=-1)
    chi_angles = torch.atan2(c3_v2, v2_mag*c1_c2)

    chi_angles_mask = list(residue_constants.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = torch.tensor(np.asarray(chi_angles_mask), device=positions.device)

    chi_mask = chi_angles_mask[aatype.long(),...]
    chi_angle_atoms_mask = torch.gather(mask, -1, atom_indices_flattern)
    chi_angle_atoms_mask = chi_angle_atoms_mask.reshape(-1, 4, 4)
    chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, -1)
    chi_mask = chi_mask * chi_angle_atoms_mask.type(positions.dtype)
    return chi_angles, chi_mask

def atom37_to_frames(aatype, all_atom_positions, all_atom_mask, eps=1e-8):
    batch_dims = len(aatype.shape[:-1])

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    for restype, restype_letter in enumerate(residue_constants.restypes):
        resname = residue_constants.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if residue_constants.chi_angles_mask[restype][chi_idx]:
                names = residue_constants.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[
                    restype, chi_idx + 4, :
                ] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*aatype.shape[:-1], 21, 8),
    )
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(
        residue_constants.chi_angles_mask
    )

    lookuptable = residue_constants.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom37_idx = aatype.new_tensor(
        restype_rigidgroup_base_atom37_idx,
    )
    restype_rigidgroup_base_atom37_idx = (
        restype_rigidgroup_base_atom37_idx.view(
            *((1,) * batch_dims), *restype_rigidgroup_base_atom37_idx.shape
        )
    )

    residx_rigidgroup_base_atom37_idx = batched_gather(
        restype_rigidgroup_base_atom37_idx,
        aatype,
        dim=-3,
        no_batch_dims=batch_dims,
    )

    base_atom_pos = batched_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom37_idx,
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )

    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
    )

    group_exists = batched_gather(
        restype_rigidgroup_mask,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom37_idx,
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)

    gt_frames = gt_frames.compose(Rigid(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(
        *((1,) * batch_dims), 21, 8
    )
    restype_rigidgroup_rots = torch.eye(
        3, dtype=all_atom_mask.dtype, device=aatype.device
    )
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1,) * batch_dims), 21, 8, 1, 1),
    )

    for resname, _ in residue_constants.residue_atom_renaming_swaps.items():
        restype = residue_constants.restype_order[residue_constants.restype_3to1[resname]]
        chi_idx = int(sum(residue_constants.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 2, 2] = -1

    residx_rigidgroup_is_ambiguous = batched_gather(
        restype_rigidgroup_is_ambiguous,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        aatype,
        dim=-4,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = Rotation(
        rot_mats=residx_rigidgroup_ambiguity_rot
    )
    alt_gt_frames = gt_frames.compose(
        Rigid(residx_rigidgroup_ambiguity_rot, None)
    )

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    return gt_frames_tensor, gt_exists, group_exists, residx_rigidgroup_is_ambiguous, alt_gt_frames_tensor

