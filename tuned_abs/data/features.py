import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import requests
import numpy as np
from .msa import Msa
from .templates import TemplateHit
from . import mmcif_parsing
from tuned_abs.common import residue_constants

def create_example_features(chains: Dict[str, str], msas: Dict[str, Msa],
                            templates: Dict[str, Optional[TemplateHit]]):
    """Create features corresponding to single example"""
    per_chain_features = {}
    for chain, sequence in chains.items():
        msa = msas[chain]
        template = templates[chain]
        per_chain_features[chain] = create_chain_features(sequence, msa,
                                                          template)
    features = merge_chain_features(per_chain_features)
    return features

def cast_features_for_model_input(features: dict):
    # Copy to avoid modification of input dict
    features = features.copy()

    for feat_name in ['aatype', 'residue_index', 'msa', 'deletion_matrix',
                      'asym_id', 'sym_id', 'entity_id']:
        if feat_name in features:
            features[feat_name] = features[feat_name].astype(np.int32)
    for feat_name in ['seq_mask', 'msa_mask', 'template_all_atom_positions']:
        if feat_name in features:
            features[feat_name] = features[feat_name].astype(np.float32)
    for feat_name in ['template_aatype']:
        if feat_name in features:
            features[feat_name] = features[feat_name].astype(np.int64)
    for feat_name in ['template_all_atom_mask']:
        if feat_name in features:
            features[feat_name] = features[feat_name].astype(bool)
    return features

def create_chain_features(sequence: str, msa: Msa, 
                          templates: Optional[List[TemplateHit]] = None):
    """Create features corresponding to a single chain"""
    aatype = make_aatype_feature(sequence)
    residue_index = np.arange(len(sequence))
    msa_ids, deletion_matrix = make_msa_features(msa)

    if templates is None:
        template_features = {}
    else:
        template_aatype, template_all_atom_positions, template_all_atom_mask = \
            make_template_features(sequence, templates)
        template_features = {
            'template_aatype': template_aatype,
            'template_all_atom_positions': template_all_atom_positions,
            'template_all_atom_mask': template_all_atom_mask
        }

    return {'aatype': aatype, 'residue_index': residue_index,
            'msa': msa_ids, 'deletion_matrix': deletion_matrix,
            **template_features}

def merge_chain_features(per_chain_features: Dict[str, Dict[str, np.ndarray]]):
    """Merge the per-chain features into example-level features"""
    # Copy to avoid modification of the original dict
    per_chain_features = _deepcopy_dict_list_only(per_chain_features)

    # Add assembly features
    #TODO: account for repetitions
    for chain_idx, chain_features in enumerate(per_chain_features.values()):
        for feat_name, id_ in [('sym_id', 1), ('asym_id', chain_idx + 1),
                              ('entity_id', chain_idx + 1)]:
            chain_features[feat_name] = np.full_like(
                chain_features['residue_index'],
                fill_value=id_
            )

    # Convenience functions
    def left_to_right_stack(feats):
        # concatenate along 1st for 1D, along 2nd for others
        return np.hstack(feats)
    def block_diagonal_stack(feats, fill_value):
        total_height = sum(feat.shape[0] for feat in feats)
        total_width = sum(feat.shape[1] for feat in feats)
        bl_diag_stack = np.full_like(
            feats[0], 
            shape=((total_height, total_width) + feats[0].shape[2:]),
            fill_value=fill_value,
        )
        cur_h, cur_w = 0, 0
        for feat in feats:
            block_h, block_w = feat.shape[:2]
            bl_diag_stack[cur_h:cur_h+block_h, cur_w:cur_w+block_w] = feat
            cur_h += block_h
            cur_w += block_w
        return bl_diag_stack

    # Create a mapping from feature names to features
    all_chains_features = defaultdict(list)
    for chain_features in per_chain_features.values():
        for feat_name, feat_val in chain_features.items():
            all_chains_features[feat_name].append(feat_val)

    # Merge features
    for feat_name in ['aatype', 'residue_index', 
                      'sym_id', 'asym_id', 'entity_id']:
        all_chains_features[feat_name] = left_to_right_stack(
            all_chains_features[feat_name]
        )

    # Merge features
    #TODO: implement cross-chain genetics
    for feat_name in ['msa', 'deletion_matrix']:
        first_rows = [feat[0] for feat in all_chains_features[feat_name]]
        other_rows = [feat[1:] for feat in all_chains_features[feat_name]]
        first_rows = left_to_right_stack(first_rows)
        if feat_name == 'msa':
            hhblits_gap_id = residue_constants.HHBLITS_AA_TO_ID['-']
            fill_value = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[hhblits_gap_id]
        else:
            fill_value = 0
        other_rows = block_diagonal_stack(other_rows, fill_value)
        merged = np.vstack([first_rows[None], other_rows])
        all_chains_features[feat_name] = merged

    for feat_name in ['template_aatype', 'template_all_atom_positions',
                      'template_all_atom_mask']:
        all_chains_features[feat_name] = left_to_right_stack(
            all_chains_features[feat_name]
        )

    return dict(all_chains_features)

def collate(per_example_features: List[Dict[str, np.ndarray]]):
    """Collate the per-example features into a batch"""
    # Copy to avoid modification of the original dict
    per_example_features = _deepcopy_dict_list_only(per_example_features)

    # Add mask features to examples
    for example in per_example_features:
        example['seq_mask'] = np.ones_like(example['residue_index'], 
                                           dtype=bool)
        example['msa_mask'] = np.ones_like(example['msa'], dtype=bool)

    # Function for collating same feature of all examples
    def collate_with_resize(all_examples_feat: List[np.ndarray]):
        all_examples_shapes = [feat.shape for feat in all_examples_feat]
        max_shape = []
        for all_examples_dims in zip(*all_examples_shapes):
            max_dim = max(all_examples_dims)
            max_shape.append(max_dim)
        feats_to_collate = []
        for feat in all_examples_feat:
            feat = feat.copy()
            feat.resize(max_shape)
            feats_to_collate.append(feat[None])
        collated_feat = np.concatenate(feats_to_collate, axis=0)
        return collated_feat

    collated_feats = {}

    # Collate accross all examples for each feature separately
    for feat_name in per_example_features[0]:
        all_examples_feat = [example_features[feat_name] 
                             for example_features in per_example_features]
        collated_feats[feat_name] = collate_with_resize(all_examples_feat)

    return collated_feats

def make_aatype_feature(sequence):
    unknown_id = residue_constants.restype_order_with_x['X']
    def map_letter_to_id(letter):
        return residue_constants.restype_order_with_x.get(
            letter, 
            unknown_id
        )
    aatype = list(map(map_letter_to_id, sequence))
    aatype = np.array(aatype, dtype=np.int32)
    return aatype

def make_msa_features(msa: Msa):
    # Convert MSA to numpy array of ids
    msa_ascii = np.array(
        [list(seq.encode('ascii')) for seq in msa.sequences], 
        dtype=np.uint8
    )
    conversion_array = np.zeros(256, dtype=np.uint8)
    for aa, id in residue_constants.HHBLITS_AA_TO_ID.items():
        conversion_array[ord(aa)] = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[id]
    msa_ids = conversion_array[msa_ascii]
    # msa_ids = np.array([[residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[residue_constants.HHBLITS_AA_TO_ID[aa]] for aa in seq] for seq in msa.sequences], dtype=np.int32)
    
    # Convert deletion matrix to numpy array of integers
    deletion_matrix = np.array(msa.deletion_matrix, dtype=np.int32)

    # Deduplication
    _, idx_dedup = np.unique(msa_ids, return_index=True, axis=0)
    idx_dedup = np.sort(idx_dedup)
    msa_ids_dedup = msa_ids[idx_dedup]
    deletion_matrix_dedup = deletion_matrix[idx_dedup]

    return msa_ids_dedup, deletion_matrix_dedup

def make_template_features(query_seq_full: str, templates: List[TemplateHit]):
    templates = sorted(templates, key=lambda x: x.sum_probs, reverse=True)

    aatype = []
    atom_positions = []
    atom_mask = []

    for template_hit in templates:
        try:
            cur_aatype, cur_atom_positions, cur_atom_mask = make_template_features_single(
                query_seq_full, template_hit
            )
            aatype.append(cur_aatype)
            atom_positions.append(cur_atom_positions)
            atom_mask.append(cur_atom_mask)
        except Exception:
            continue
        if len(aatype) >= 4: #TODO: make number of templates configurable
            break
    return np.stack(aatype), np.stack(atom_positions), np.stack(atom_mask)

def make_template_features_single(query_seq_full, template_hit):
    # Parse PDB ID and chain ID from the template hit
    pattern = r'([a-zA-Z\d]{4})_(.)'
    pdb_id, chain_id = re.match(pattern, template_hit.name).groups()

    # Download the cif file
    cif_url = f'https://files.rcsb.org/download/{pdb_id}.cif'
    cif_string = requests.get(cif_url).text
    
    # Parse cif to get mmCIF object
    mmcif_object = mmcif_parsing.parse(
        file_id='', mmcif_string=cif_string
    ).mmcif_object
    if mmcif_object is None:
        raise RuntimeError("Could not parse mmcif file")

    # Get the full hit sequence (TemplateHit has only part of it)
    hit_seq_full = mmcif_object.chain_to_seqres[chain_id]

    # Build mapping from residues index of target to residues index of full 
    # hit sequence
    mapping = _build_target_to_template_mapping(
        template_hit.query, template_hit.hit_sequence,
        query_seq_full, hit_seq_full
    )

    # Get the atom positions and atom mask for the whole chain
    atom_positions_full, atom_mask_full = _get_atom_positions(
        mmcif_object, chain_id, max_ca_ca_distance=150.0)

    # Constructur positions and mask features
    aatype = ['-'] * len(query_seq_full)
    atom_positions = np.zeros(
        (len(query_seq_full), residue_constants.atom_type_num, 3),
        dtype=np.float32
    )
    atom_mask = np.zeros(
        (len(query_seq_full), residue_constants.atom_type_num),
        dtype=bool
    )
    
    mapping_keys = list(mapping.keys())
    mapping_values = list(mapping.values())
    
    for target_idx, full_hit_idx in mapping.items():
        aatype[target_idx] = hit_seq_full[full_hit_idx]
    atom_positions[mapping_keys] = atom_positions_full[mapping_values]
    atom_mask[mapping_keys] = atom_mask_full[mapping_values]

    def restype_to_id(restype):
        id = residue_constants.HHBLITS_AA_TO_ID[restype]
        id = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[id]
        return id

    aatype = np.array(list(map(restype_to_id, aatype)), dtype=np.int32)

    return aatype, atom_positions, atom_mask

#TODO: maybe better make use indices from .hhr file
def _build_target_to_template_mapping(query_seq: str, hit_seq: str, 
                                      query_seq_full: str, hit_seq_full: str):
    """Return a dictionary mapping residues indices within a target AA sequence
    to respective residues indices whithin full chain of template hit.
    Args:
        query_seq: str
            AA sequence with possibly gap characters
        hit_seq: str
            AA sequence with possibly gap characters, same length as query_seq
        query_seq_full: str
            AA sequence of the target
        hit_seq_full: str
            AA sequence from PDB or mmCIF file

    query_seq and hit_seq should be aligned
    """

    if len(query_seq) != len(hit_seq):
        raise ValueError("Expected query_seq nad hit_seq to be of the same "
                         "length.")

    mapping = {}

    i, j = 0, 0
    for q, h in zip(query_seq, hit_seq):
        if q != '-' and h != '-':
            mapping[i] = j

        if q != '-':
            i += 1

        if h != '-':
            j += 1

    def find_offset(seq, seq_full):
        seq_no_gaps = seq.replace('-', '')
        seq_offset = seq_full.find(seq_no_gaps)
        return seq_offset

    query_offset = find_offset(query_seq, query_seq_full)
    hit_offset = find_offset(hit_seq, hit_seq_full)

    if query_offset == -1 or hit_offset == -1:
        raise ValueError("query_seq (hit_seq) should be a contigues substring"
                         "of query_seq_full (hit_seq_full)")

    mapping = {k + query_offset : v + hit_offset for k, v in mapping.items()}

    return mapping

def _get_atom_positions(
    mmcif_object: mmcif_parsing.MmcifObject,
    auth_chain_id: str,
    max_ca_ca_distance: float) -> Tuple[np.ndarray, np.ndarray]:
  """Gets atom positions and mask from a list of Biopython Residues."""
  num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])

  relevant_chains = [c for c in mmcif_object.structure.get_chains()
                     if c.id == auth_chain_id]
  if len(relevant_chains) != 1:
    raise ValueError(
        f'Expected exactly one chain in structure with id {auth_chain_id}.')
  chain = relevant_chains[0]

  all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
  all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                dtype=np.int64)
  for res_index in range(num_res):
    pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
    mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
    res_at_position = mmcif_object.seqres_to_structure[auth_chain_id][res_index]
    if not res_at_position.is_missing:
      res = chain[(res_at_position.hetflag,
                   res_at_position.position.residue_number,
                   res_at_position.position.insertion_code)]
      for atom in res.get_atoms():
        atom_name = atom.get_name()
        x, y, z = atom.get_coord()
        if atom_name in residue_constants.atom_order.keys():
          pos[residue_constants.atom_order[atom_name]] = [x, y, z]
          mask[residue_constants.atom_order[atom_name]] = 1.0
        elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
          # Put the coordinates of the selenium atom in the sulphur column.
          pos[residue_constants.atom_order['SD']] = [x, y, z]
          mask[residue_constants.atom_order['SD']] = 1.0

      # Fix naming errors in arginine residues where NH2 is incorrectly
      # assigned to be closer to CD than NH1.
      cd = residue_constants.atom_order['CD']
      nh1 = residue_constants.atom_order['NH1']
      nh2 = residue_constants.atom_order['NH2']
      if (res.get_resname() == 'ARG' and
          all(mask[atom_index] for atom_index in (cd, nh1, nh2)) and
          (np.linalg.norm(pos[nh1] - pos[cd]) >
           np.linalg.norm(pos[nh2] - pos[cd]))):
        pos[nh1], pos[nh2] = pos[nh2].copy(), pos[nh1].copy()
        mask[nh1], mask[nh2] = mask[nh2].copy(), mask[nh1].copy()

    all_positions[res_index] = pos
    all_positions_mask[res_index] = mask
  _check_residue_distances(
      all_positions, all_positions_mask, max_ca_ca_distance)
  return all_positions, all_positions_mask

def _check_residue_distances(all_positions: np.ndarray,
                             all_positions_mask: np.ndarray,
                             max_ca_ca_distance: float):
  """Checks if the distance between unmasked neighbor residues is ok."""
  ca_position = residue_constants.atom_order['CA']
  prev_is_unmasked = False
  prev_calpha = None
  for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):
    this_is_unmasked = bool(mask[ca_position])
    if this_is_unmasked:
      this_calpha = coords[ca_position]
      if prev_is_unmasked:
        distance = np.linalg.norm(this_calpha - prev_calpha)
        if distance > max_ca_ca_distance:
          raise RuntimeError(
              'The distance between residues %d and %d is %f > limit %f.' % (
                  i, i + 1, distance, max_ca_ca_distance))
      prev_calpha = this_calpha
    prev_is_unmasked = this_is_unmasked

def generate_random_example(config: dict):
    num_aatypes = config['aatype']
    sequence_len = 150
    raw_msa_seqs = 4096
    num_templates = 4

    example = {
        'aatype': np.random.randint(0, num_aatypes, size=sequence_len, 
                                    dtype=np.int32),
        'seq_mask': np.random.randint(
            0, 1, size=sequence_len
        ).astype(np.float32),
        'residue_index': np.arange(sequence_len, dtype=np.int32),
        'msa': np.random.randint(
            0, num_aatypes + 1, size=(raw_msa_seqs, sequence_len),
            dtype=np.int32
        ),
        'msa_mask': np.random.randint(
            0, 2, size=(raw_msa_seqs, sequence_len)
        ).astype(np.float32),
        'deletion_matrix': np.zeros(shape=(raw_msa_seqs, sequence_len), 
                                    dtype=np.int32),
        'template_aatype': np.random.randint(
            0, num_aatypes, size=(num_templates, sequence_len),
            dtype=np.int64
        ),
        'template_all_atom_mask': np.random.randint(
            0, 2, size=(num_templates, sequence_len, 37)
        ).astype(bool),
        'template_all_atom_positions': np.random.randn(
            num_templates, sequence_len, 37, 3
        ).astype(np.float32),
        'asym_id': np.zeros(sequence_len, dtype=np.int32),
        'sym_id': np.zeros(sequence_len, dtype=np.int32),
        'entity_id': np.zeros(sequence_len, dtype=np.int32),
    }
    
    return example

def _deepcopy_dict_list_only(obj: Any):
    if isinstance(obj, dict):
        new_dict = {k: _deepcopy_dict_list_only(v) for k, v in obj.items()}

        return new_dict
    elif isinstance(obj, list):
        new_list = [_deepcopy_dict_list_only(el) for el in obj]
        
        return new_list

    return obj
