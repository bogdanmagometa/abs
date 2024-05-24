# tuned-abs

You can try `tuned-abs` in [Google Colab](https://colab.research.google.com/drive/1-MzEszyV9bVcX3VKz1FG2ythNfA_XYk1?usp=sharing).

## Requirements

The following tools should be installed:
- HMMER (`jackhmmer` and `esl-reformat` binaries are used)
- HH-suite (`hhsearch` binary is used)

In conda environment, HMMER and HH-suite can be installed with the 
following commands:
```bash
$ conda install bioconda::hmmer
$ conda install -c conda-forge -c bioconda hhsuite
```

## Installation

```bash
pip3 install git+https://github.com/bogdanmagometa/abs
```

## Usage

**Note**: the first use of the pipeline will cause the download of weights 
unless other pipelines that require the same weights have been already used.

### From command line
```bash
$ python3 -m tuned_abs --help
usage: python3 -m tuned_abs <input fasta> <output pdb> [--pipeline PIPELINE]

positional arguments:
  input_fasta          Path to .fasta file with two sequences (one for each chain)
  output_pdb           Save the predicted structure to specified .pdb file

optional arguments:
  -h, --help           show this help message and exit
  --pipeline PIPELINE  Which pipeline to use. Can be one of 'Finetuned', 'Finetuned 1x5',
                       'FinetunedValid', 'FinetunedValid 1x5', 'FinetunedValidRefined',
                       'SingleSequence' (default: SingleSequence)
  --quiet              Inference in quiet mode (default: False)
$ python3 -m tuned_abs input.fasta output.pdb --pipeline 'Finetuned 1x5'
```

### From code
```python
from tuned_abs import Pipeline

pipeline = Pipeline.from_name('SingleSequence') # 'Finetuned', 'Finetuned 1x5', 
# 'FinetunedValid', 'FinetunedValid 1x5', 'FinetunedValidRefined', 'SingleSequence'

pdb_string = pipeline.run(
    {
        'H': 'EVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVIWYDGSNRYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCHRNYYDSSGPFDYWGQGTLVTVSS',
        'L': 'DIQMTQSPSTLSASVGDRVTITCRASQFISRWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSETHFTLTISSLQPDDVATYYCQEYTSYGRTFGQGTKVEIKRTV',
    }
)

with open('output.pdb') as f:
    f.write(pdb_string)
```

### Visualize

You can visualize the predicted structure using [Mol* website](https://molstar.org/), [PyMol program](https://pymol.org/), [ProDy package](https://pypi.org/project/ProDy/) or other tools:

![myfile](img/tuned_abs_vis.gif)

