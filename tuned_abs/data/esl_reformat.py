import shutil
import os
import subprocess

class EslReformatRunner:
    def __init__(self, output_format: str = 'a2m'):
        self.output_format = output_format
        self.esl_reformat_bin_path = os.path.abspath(
            shutil.which('esl-reformat')
        )
    def __call__(self, raw_msa: str) -> str:
        completed_process = subprocess.run([
            self.esl_reformat_bin_path,
            self.output_format,
            '-',
        ], text=True, input=raw_msa, capture_output=True)
        output_msa = completed_process.stdout
        return output_msa
