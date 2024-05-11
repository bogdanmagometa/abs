from __future__ import annotations
from typing import Dict, NamedTuple, Optional
import math

import torch
import numpy as np

from .data.features import (
    collate,
    cast_features_for_model_input
)
from .data.data_pipeline import DataPipeline
from .model.modules_multimer import create_alphafold
from .model.tensor_utils import to_device
from .model.tensor_utils import tree_map
from .configs import get_config
from .weights.weights import get_weights_path
from .utils import compute_ranking_confidence, cvt_output_to_pdb

class PipelineSettings(NamedTuple):
    # Jointly specifies the network config and params
    # One of 'finetuned', 'finetunedvalid' or 'singlesequence'
    network: str = 'finetuned'

    # If False, pass single sequence as MSA
    msa: bool = True

    # If False, pass no templates
    templates: bool = True

    # Number of confomation space samples
    num_conf_samples: int = 1

    # Whether to refine the final structure with PyRosetta
    refine: bool = False 

class Pipeline:
    def __init__(self, data_pipeline, model, num_conf_samples):
        self.data_pipeline = data_pipeline
        self.model = model
        self.num_conf_samples = num_conf_samples

    def run(self, chains: Dict[str, str], device: str = 'cuda', 
            n_conf_space_samples: Optional[int] = None):
        input_features = self.data_pipeline(chains)
        if n_conf_space_samples is None:
            n_conf_space_samples = self.num_conf_samples

        batch, best_output = self._sample_conformational_space(
            input_features, n_conf_space_samples, device
        )

        pdb_string = cvt_output_to_pdb(batch, best_output)
        
        return pdb_string

    def _sample_conformational_space(self, features: dict, n: int, device: str):
        self.model.to(device)

        best_conf = -math.inf
        best_out = None

        batch = collate([features])
        batch = cast_features_for_model_input(batch)
        batch = tree_map(torch.tensor, batch, np.ndarray)
        batch = to_device(batch, device)

        for _ in range(n):
            with torch.no_grad():
                out = self.model(batch)

            rank_conf = compute_ranking_confidence(out)

            if rank_conf > best_conf:
                best_conf = rank_conf
                best_out = out

            del out

        return batch, best_out

    @classmethod
    def from_name(clas, name: str) -> Pipeline:
        """Create a pipeline object according to the provided name.
        
        Args:
            name: one of 'Finetuned', 'Finetuned 1x5', 'FinetunedValid',
        'FinetunedValid 1x5', 'FinetunedValidRefined', 'SingleSequence'.
        """
        
        pipeline_settings = clas._parse_name(name)
        
        if pipeline_settings is None:
            raise ValueError(f'Cannot recognize the name: {name}')

        pipeline = clas.from_settings(pipeline_settings)
        
        return pipeline

    @classmethod
    def from_settings(clas, settings: PipelineSettings):
        data_pipeline = DataPipeline(msa=settings.msa, 
                                     templates=settings.templates)

        config = get_config(settings.network)
        model = create_alphafold(config)
        
        weights_path = get_weights_path(settings.network)
        weights = torch.load(weights_path, map_location='cpu')
        model.net_iteration.load_state_dict(weights)

        pipeline = Pipeline(data_pipeline, model, settings.num_conf_samples)

        return pipeline

    @classmethod
    def _parse_name(clas, name: str):
        name = name.lower()
        name = name.replace(' ', '')
        
        settings = {
            'finetuned': PipelineSettings(),
            'finetuned1x5': PipelineSettings(num_conf_samples=5),
            'finetunedvalid': PipelineSettings(network='finetunedvalid'),
            'finetunedvalid1x5': PipelineSettings(
                network='finetunedvalid',
                num_conf_samples=5,
            ),
            'finetunedvalidrefined': PipelineSettings(
                network='finetunedvalid',
                refine=True
            ),
            'singlesequence': PipelineSettings(
                network='singlesequence',
                msa=False,
                templates=False,
            ),
        }.get(name, None)

        return settings
