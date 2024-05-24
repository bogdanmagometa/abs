import os
import subprocess
import numpy as np
import gdown
from .jackhmmer import JackhmmerRunner
from .hhsearch import HHSearchRunner
from .features import create_example_features
from .msa import parse_stockholm
from .templates import parse_hhr
from .esl_reformat import EslReformatRunner

_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

MSA_AB_URL = 'https://drive.google.com/uc?id=1xqmoQpRyU7uDx4CeD4cGpAQ-fro9Gir6'
MSA_AB_FILE = os.path.join(_ROOT_DIR, 'antibody_db.fasta')
# MSA_AB_FILE = '/titan/bohdan/antibody_db.fasta'

TEMPLATES_DB_URL = 'https://drive.google.com/drive/folders/1UrmoblMwnZFGrQNvh-zXIbsWM2HzvYfU'
TEMPLATES_DB = os.path.join(_ROOT_DIR, 'antibody_hhsearch_db')
# TEMPLATES_DB = '/titan/bohdan/abs/tuned_abs/antibody_hhsearch_db'
TEMPLATES_DB_PREFIX = 'rep_fas'

class DataPipeline:
    def __init__(self, msa=True, templates=True):
        if msa:
            if not os.path.exists(MSA_AB_FILE):
                gdown.download(MSA_AB_URL, MSA_AB_FILE, quiet=False)
            self.msa_runner = JackhmmerRunner(MSA_AB_FILE)
        else:
            self.msa_runner = _DummyMsaRunner()

        self.format_converter = EslReformatRunner()

        if templates:
            if not os.path.exists(TEMPLATES_DB):
                gdown.download_folder(TEMPLATES_DB_URL, output=TEMPLATES_DB, 
                                      quiet=False)
            template_db = os.path.join(TEMPLATES_DB, TEMPLATES_DB_PREFIX)
            self.templates_runner = HHSearchRunner(template_db)
        else:
            self.templates_runner = _DummyTemplateRunner()
    def __call__(self, chains):
        # Get MSAs
        MAX_SEQ_PER_MSA = 2048
        raw_msas = {
            chain: self.msa_runner(chains[chain])
            for chain in chains
        }
        msas = {
            chain: parse_stockholm(raw_msa).truncate(MAX_SEQ_PER_MSA)
            for chain, raw_msa in raw_msas.items()
        }

        # Convert .sto to .a3m
        for chain in chains:
            raw_msa = raw_msas[chain]
            sequence = chains[chain]
            raw_msa = self.format_converter(raw_msa)
            raw_msa = f">{chain}\n{sequence}\n" + raw_msa
            raw_msas[chain] = raw_msa

        # Get templates
        templates = {
            chain: parse_hhr(self.templates_runner(raw_msa)) 
            for chain, raw_msa in raw_msas.items()
        }

        # Build numpy features
        input_features = create_example_features(chains, msas, templates)

        # Duplicate the MSA to later have the same sequence for cluster and
        # extra MSA stacks
        if isinstance(self.msa_runner, _DummyMsaRunner): #TODO: refactor
            for feat_name in ['msa', 'deletion_matrix']:
                input_features[feat_name] = np.vstack(
                    [input_features[feat_name]] * 2
                )

        return input_features

class _DummyMsaRunner:
    def __call__(sequence: str):
        return ">chain\n" + sequence

class _DummyTemplateRunner:
    def __call__(sequence: str):
        return None
