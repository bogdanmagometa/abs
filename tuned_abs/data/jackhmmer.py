import shutil
import os
import subprocess
import tempfile
from .msa import Msa, parse_stockholm

AF_M_V2_JACKHMMER_OPTIONS = [
    '--noali',
    '--F1', '0.0005',
    '--F2', '5e-05',
    '--F3', '5e-07',
    '-E', '0.0001',
    '-N', '1',
]

class JackhmmerRunner:
    def __init__(self, msaAb_file):
        self.msaAb_file = msaAb_file
        self.jackhmmer_bin_path = os.path.abspath(
            shutil.which('jackhmmer')
        )
    def __call__(self, chain: str) -> Msa:
        fasta_input = f'>chain\n{chain}'

        # create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.close() # the file is closed, but not removed

            subprocess.run([self.jackhmmer_bin_path,
                            '--cpu', str(8),
                            '-o', os.devnull,
                            '-A', fp.name,
                            *AF_M_V2_JACKHMMER_OPTIONS,
                            '-', self.msaAb_file], 
                           text=True, input=fasta_input)
            with open(fp.name, mode='r') as f:
                msa = f.read()

        # the file is now removed

        msa = parse_stockholm(msa)

        return msa
