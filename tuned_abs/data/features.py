from typing import Dict, List, Any
from collections import defaultdict
import numpy as np
from .msa import Msa
# from .templates import Template
from tuned_abs.common import residue_constants

def create_example_features(chains: Dict[str, str], msas: Dict[str, Msa]):
    """Create features corresponding to single example"""
    per_chain_features = {}
    for chain, sequence in chains.items():
        msa = msas[chain]
        per_chain_features[chain] = create_chain_features(sequence, msa)
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

def create_chain_features(sequence: str, msa: Msa):
    """Create features corresponding to a single chain"""
    aatype = make_aatype_feature(sequence)
    residue_index = np.arange(len(sequence))
    msa_ids, deletion_matrix = make_msa_features(msa)
    #TODO: add template features
    return {'aatype': aatype, 'residue_index': residue_index,
            'msa': msa_ids, 'deletion_matrix': deletion_matrix}

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
        return np.concatenate(feats, axis=0)
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
    msa_ids = np.array([[residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[residue_constants.HHBLITS_AA_TO_ID[aa]] for aa in seq] for seq in msa.sequences], dtype=np.int32)
    
    # Convert deletion matrix to numpy array of integers
    deletion_matrix = np.array(msa.deletion_matrix, dtype=np.int32)

    # Deduplication
    _, idx_dedup = np.unique(msa_ids, return_index=True, axis=0)
    idx_dedup = np.sort(idx_dedup)
    msa_ids_dedup = msa_ids[idx_dedup]
    deletion_matrix_dedup = deletion_matrix[idx_dedup]

    return msa_ids_dedup, deletion_matrix_dedup

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
