import shutil
import os
import subprocess

class HHSearchRunner:
    def __init__(self, pdb_database):
        self.pdb_database = pdb_database
        self.hhsearch_bin_path = os.path.abspath(
            shutil.which('hhsearch')
        )
    def __call__(self, msa: str) -> str:
        fasta_input = msa

        completed_process = subprocess.run([self.hhsearch_bin_path,
                        '-i', 'stdin',
                        '-d', self.pdb_database,
                        '-o', 'stdout',
                        '-cpu', str(4)],
                        text=True, input=fasta_input, capture_output=True)
        hhr_content = completed_process.stdout
        return hhr_content
