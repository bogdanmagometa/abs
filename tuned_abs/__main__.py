from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tuned_abs import Pipeline

# Parse command line arguments
parser = ArgumentParser(
    'tuned_abs',
    usage='python3 -m tuned_abs <input fasta> <output pdb> [--pipeline PIPELINE]',
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument('input_fasta', help='Path to .fasta file with two'
                    ' sequences (one for each chain)')
parser.add_argument('output_pdb', help='Save the predicted structure to '
                    'specified .pdb file')
parser.add_argument('--pipeline', help='Which pipeline to use. Can be one of '
                    "'Finetuned', 'Finetuned 1x5', 'FinetunedValid', "
                    "'FinetunedValid 1x5', 'FinetunedValidRefined', "
                    "'SingleSequence'", default='SingleSequence')
args = parser.parse_args()


# Parse input .fasta file
input_sequences = []
with open(args.input_fasta) as input_fasta_f:
    for line in input_fasta_f.readlines():
        if line[0] != '>' and line.strip():
            input_sequences.append(line.strip())

if len(input_sequences) != 2:
    raise RuntimeError('Expected input fasta to contain two sequences. '
                       f'Found {len(input_sequences)}')

# Initialize pipeline
pipeline = Pipeline.from_name(args.pipeline) # 'SingleSequence', 'Finetuned', 'FinetunedValid'

# Run all steps of the pipeline
pdb_string = pipeline.run(
    {
        'chain1': input_sequences[0],
        'chain2': input_sequences[1],
    }
)

# Save predicted structure to output pdb
with open(args.output_pdb, 'w') as output_pdb_f:
    output_pdb_f.write(pdb_string)
