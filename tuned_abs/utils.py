from typing import Optional

import numpy as np
import torch

from tuned_abs.common import residue_constants

PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)

def cvt_output_to_pdb(batch, output):
    residue_index = batch['residue_index'][0].cpu().numpy()
    
    plddt = _compute_plddt(output['predicted_lddt']['logits'])
    plddt = plddt.detach().cpu().numpy()
    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )
    pdb_out = _protein_to_pdb(
        batch['aatype'][0].cpu().numpy(),
        output['final_atom_positions'].detach().cpu().numpy(),
        residue_index + 1,
        batch['asym_id'][0].cpu().numpy(),
        output['final_atom_mask'].cpu().numpy(), plddt_b_factors[0]
    )
    return pdb_out


def compute_ranking_confidence(output: dict):
    ptm = _predicted_tm_score(
        logits=output['predicted_aligned_error']['logits'],
        breaks=output['predicted_aligned_error']['breaks'],
        asym_id=None)

    iptm = _predicted_tm_score(
        logits=output['predicted_aligned_error']['logits'],
        breaks=output['predicted_aligned_error']['breaks'],
        asym_id=output['predicted_aligned_error']['asym_id'],
        interface=True)

    ranking_confidence = (0.8 * iptm + 0.2 * ptm)

    return ranking_confidence

def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers

def _predicted_tm_score(
    logits: torch.Tensor,
    breaks: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])
    bin_centers = _calculate_bin_centers(breaks)

    num_res = logits.shape[-2]
    clipped_num_res = max(num_res, 19)
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    pair_mask = torch.ones((num_res, num_res), dtype=bool, device=logits.device)
    if interface:
        pair_mask *= asym_id[..., None] != asym_id[None, ...]

    predicted_tm_term *= pair_mask
    pair_residue_weights = pair_mask * (residue_weights[None, :] * residue_weights[:, None])
    normed_residue_mask = pair_residue_weights / (1e-8 + torch.sum(
      pair_residue_weights, dim=-1, keepdim=True))
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
    weighted = per_alignment * residue_weights
    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]

def _compute_plddt(logits):
  """Computes per-residue pLDDT from logits.

  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.

  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = torch.arange(0.5 * bin_width, 1.0, bin_width, device=logits.device)
  probs = torch.nn.functional.softmax(logits, dim=-1)
  predicted_lddt_ca = torch.sum(probs * bin_centers[None, ...], dim=-1)
  return predicted_lddt_ca * 100

def _protein_to_pdb(aatype, atom_positions, residue_index, chain_index, atom_mask, b_factors, out_mask=None):
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []
    residue_index = residue_index.astype(np.int32)
    chain_index = chain_index.astype(np.int32)
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
          f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]
    for i in range(aatype.shape[0]):
        if out_mask is not None and out_mask[i] == 0:
            continue
        if last_chain_index != chain_index[i]:
            pdb_lines.append(_chain_end(
            atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]],
            residue_index[i - 1]))
            last_chain_index = chain_index[i]
            atom_index += 1

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1
    pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                              chain_ids[chain_index[-1]], residue_index[-1]))
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'

def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')

