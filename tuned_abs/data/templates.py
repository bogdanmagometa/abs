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
#
#
# Modifications Copyright 2022 Applied BioComputation Group, Stony Brook University

"""Functions for parsing files with templates"""

import dataclasses
import re
from typing import List, Optional, Sequence


@dataclasses.dataclass(frozen=True)
class TemplateHit:
  """Class representing a template hit."""
  index: int
  name: str
  aligned_cols: int
  sum_probs: Optional[float]
  query: str
  hit_sequence: str
  indices_query: List[int]
  indices_hit: List[int]


def _get_hhr_line_regex_groups(
    regex_pattern: str, line: str) -> Sequence[Optional[str]]:
  match = re.match(regex_pattern, line)
  if match is None:
    raise RuntimeError(f'Could not parse query line {line}')
  return match.groups()


def _update_hhr_residue_indices_list(
    sequence: str, start_index: int, indices_list: List[int]):
  """Computes the relative indices for each residue with respect to the original sequence."""
  counter = start_index
  for symbol in sequence:
    if symbol == '-':
      indices_list.append(-1)
    else:
      indices_list.append(counter)
      counter += 1


def _parse_hhr_hit(detailed_lines: Sequence[str]) -> TemplateHit:
  """Parses the detailed HMM HMM comparison section for a single Hit.

  This works on .hhr files generated from both HHBlits and HHSearch.

  Args:
    detailed_lines: A list of lines from a single comparison section between 2
      sequences (which each have their own HMM's)

  Returns:
    A dictionary with the information from that detailed comparison section

  Raises:
    RuntimeError: If a certain line cannot be processed
  """
  # Parse first 2 lines.
  number_of_hit = int(detailed_lines[0].split()[-1])
  name_hit = detailed_lines[1][1:]

  # Parse the summary line.
  pattern = (
      'Probab=(.*)[\t ]*E-value=(.*)[\t ]*Score=(.*)[\t ]*Aligned_cols=(.*)[\t'
      ' ]*Identities=(.*)%[\t ]*Similarity=(.*)[\t ]*Sum_probs=(.*)[\t '
      ']*Template_Neff=(.*)')
  match = re.match(pattern, detailed_lines[2])
  if match is None:
    raise RuntimeError(
        'Could not parse section: %s. Expected this: \n%s to contain summary.' %
        (detailed_lines, detailed_lines[2]))
  (_, _, _, aligned_cols, _, _, sum_probs, _) = [float(x)
                                                 for x in match.groups()]

  # The next section reads the detailed comparisons. These are in a 'human
  # readable' format which has a fixed length. The strategy employed is to
  # assume that each block starts with the query sequence line, and to parse
  # that with a regexp in order to deduce the fixed length used for that block.
  query = ''
  hit_sequence = ''
  indices_query = []
  indices_hit = []
  length_block = None

  for line in detailed_lines[3:]:
    # Parse the query sequence line
    if (line.startswith('Q ') and not line.startswith('Q ss_dssp') and
        not line.startswith('Q ss_pred') and
        not line.startswith('Q Consensus')):
      # Thus the first 17 characters must be 'Q <query_name> ', and we can parse
      # everything after that.
      #              start    sequence       end       total_sequence_length
      patt = r'[\t ]*([0-9]*) ([A-Z-]*)[\t ]*([0-9]*) \([0-9]*\)'
      groups = _get_hhr_line_regex_groups(patt, line[17:])

      # Get the length of the parsed block using the start and finish indices,
      # and ensure it is the same as the actual block length.
      start = int(groups[0]) - 1  # Make index zero based.
      delta_query = groups[1]
      end = int(groups[2])
      num_insertions = len([x for x in delta_query if x == '-'])
      length_block = end - start + num_insertions
      assert length_block == len(delta_query)

      # Update the query sequence and indices list.
      query += delta_query
      _update_hhr_residue_indices_list(delta_query, start, indices_query)

    elif line.startswith('T '):
      # Parse the hit sequence.
      if (not line.startswith('T ss_dssp') and
          not line.startswith('T ss_pred') and
          not line.startswith('T Consensus')):
        # Thus the first 17 characters must be 'T <hit_name> ', and we can
        # parse everything after that.
        #              start    sequence       end     total_sequence_length
        patt = r'[\t ]*([0-9]*) ([A-Z-]*)[\t ]*[0-9]* \([0-9]*\)'
        groups = _get_hhr_line_regex_groups(patt, line[17:])
        start = int(groups[0]) - 1  # Make index zero based.
        delta_hit_sequence = groups[1]
        assert length_block == len(delta_hit_sequence)

        # Update the hit sequence and indices list.
        hit_sequence += delta_hit_sequence
        _update_hhr_residue_indices_list(delta_hit_sequence, start, indices_hit)

  return TemplateHit(
      index=number_of_hit,
      name=name_hit,
      aligned_cols=int(aligned_cols),
      sum_probs=sum_probs,
      query=query,
      hit_sequence=hit_sequence,
      indices_query=indices_query,
      indices_hit=indices_hit,
  )


def parse_hhr(hhr_string: str) -> Sequence[TemplateHit]:
  """Parses the content of an entire HHR file."""
  lines = hhr_string.splitlines()

  # Each .hhr file starts with a results table, then has a sequence of hit
  # "paragraphs", each paragraph starting with a line 'No <hit number>'. We
  # iterate through each paragraph to parse each hit.

  block_starts = [i for i, line in enumerate(lines) if line.startswith('No ')]

  hits = []
  if block_starts:
    block_starts.append(len(lines))  # Add the end of the final block.
    for i in range(len(block_starts) - 1):
      hits.append(_parse_hhr_hit(lines[block_starts[i]:block_starts[i + 1]]))
  return hits

