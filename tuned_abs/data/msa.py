# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for parsing files with MSA"""

import collections
import dataclasses
from typing import Sequence

DeletionMatrix = Sequence[Sequence[int]]

@dataclasses.dataclass(frozen=True)
class Msa:
    """Class representing a parsed MSA file."""

    sequences: Sequence[str]
    deletion_matrix: DeletionMatrix
    descriptions: Sequence[str]

    def __post_init__(self):
        if not (
            len(self.sequences) == len(self.deletion_matrix) == len(self.descriptions)
        ):
            raise ValueError(
                "All fields for an MSA must have the same length. "
                f"Got {len(self.sequences)} sequences, "
                f"{len(self.deletion_matrix)} rows in the deletion matrix and "
                f"{len(self.descriptions)} descriptions."
            )

    def __len__(self):
        return len(self.sequences)

    def truncate(self, max_seqs: int):
        return Msa(
            sequences=self.sequences[:max_seqs],
            deletion_matrix=self.deletion_matrix[:max_seqs],
            descriptions=self.descriptions[:max_seqs],
        )


def parse_stockholm(stockholm_string: str) -> Msa:
    """Parses sequences and deletion matrix from stockholm format alignment.

    Args:
      stockholm_string: The string contents of a stockholm file. The first
        sequence in the file should be the query sequence.

    Returns:
      A tuple of:
        * A list of sequences that have been aligned to the query. These
          might contain duplicates.
        * The deletion matrix for the alignment as a list of lists. The element
          at `deletion_matrix[i][j]` is the number of residues deleted from
          the aligned sequence i at residue position j.
        * The names of the targets matched, including the jackhmmer subsequence
          suffix.
    """
    name_to_sequence = collections.OrderedDict()
    for line in stockholm_string.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        name, sequence = line.split()
        if name not in name_to_sequence:
            name_to_sequence[name] = ""
        name_to_sequence[name] += sequence

    msa = []
    deletion_matrix = []

    query = ""
    keep_columns = []
    for seq_index, sequence in enumerate(name_to_sequence.values()):
        if seq_index == 0:
            # Gather the columns with gaps from the query
            query = sequence
            keep_columns = [i for i, res in enumerate(query) if res != "-"]

        # Remove the columns with gaps in the query from all sequences.
        aligned_sequence = "".join([sequence[c] for c in keep_columns])

        msa.append(aligned_sequence)

        # Count the number of deletions w.r.t. query.
        deletion_vec = []
        deletion_count = 0
        for seq_res, query_res in zip(sequence, query):
            if seq_res != "-" or query_res != "-":
                if query_res == "-":
                    deletion_count += 1
                else:
                    deletion_vec.append(deletion_count)
                    deletion_count = 0
        deletion_matrix.append(deletion_vec)

    return Msa(
        sequences=msa,
        deletion_matrix=deletion_matrix,
        descriptions=list(name_to_sequence.keys()),
    )
