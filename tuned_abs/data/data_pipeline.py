import os
import numpy as np
import gdown
from .jackhmmer import JackhmmerRunner
from .features import create_example_features
from .msa import Msa

_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

MSA_AB_URL = 'https://drive.google.com/uc?id=1xqmoQpRyU7uDx4CeD4cGpAQ-fro9Gir6'
MSA_AB_FILE = os.path.join(_ROOT_DIR, 'antibody_db.fasta')
# MSA_AB_FILE = '/titan/bohdan/antibody_db.fasta'

class DataPipeline:
    def __init__(self, msa=True, templates=True):
        if msa:
            if not os.path.exists(MSA_AB_FILE):
                gdown.download(MSA_AB_URL, MSA_AB_FILE, quiet=False)
            self.msa_runner = JackhmmerRunner(MSA_AB_FILE)
        else:
            self.msa_runner = None
    def __call__(self, chains):
        MAX_SEQ_PER_MSA = 2048
        if self.msa_runner is None:
            msas = {
                chain: Msa(
                    sequences=[chains[chain]], 
                    deletion_matrix=[[0] * len(chains[chain])],
                    descriptions=[''], #TODO: is empty description valid?
                )
                for chain in chains
            }
        else:
            msas = {
                chain: self.msa_runner(chains[chain]).truncate(MAX_SEQ_PER_MSA)
                for chain in chains
            }
        input_features = create_example_features(chains, msas)

        if self.msa_runner is None:
            for feat_name in ['msa', 'deletion_matrix']:
                input_features[feat_name] = np.vstack(
                    [input_features[feat_name]] * 2
                )

        return input_features
