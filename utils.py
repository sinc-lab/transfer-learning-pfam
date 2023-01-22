"""Adapted from https://github.com/google-research/google-research/blob/master/using_dl_to_annotate_protein_universe/neural_network/utils.py"""
import os 
import numpy as np
import pandas as pd

AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]

_PFAM_GAP_CHARACTER = '.'

# Other characters representing amino-acids not in AMINO_ACID_VOCABULARY.
_ADDITIONAL_AA_VOCABULARY = [
    # Substitutions
    'U',
    'O',
    # Ambiguous Characters
    'B',
    'Z',
    'X',
    # Gap Character
    _PFAM_GAP_CHARACTER
]

# Vocab of all possible tokens in a valid input sequence
FULL_RESIDUE_VOCAB = AMINO_ACID_VOCABULARY + _ADDITIONAL_AA_VOCABULARY

# Map AA characters to their index in FULL_RESIDUE_VOCAB.
_RESIDUE_TO_INT = {aa: idx for idx, aa in enumerate(FULL_RESIDUE_VOCAB)}

def residues_to_indices(amino_acid_residues):
  return [_RESIDUE_TO_INT[c] for c in amino_acid_residues]

def residues_to_one_hot(amino_acid_residues):
  """Given a sequence of amino acids, return one hot array.
  Supports ambiguous amino acid characters B, Z, and X by distributing evenly
  over possible values, e.g. an 'X' gets mapped to [.05, .05, ... , .05].
  Supports rare amino acids by appropriately substituting. See
  normalize_sequence_to_blosum_characters for more information.
  Supports gaps and pads with the '.' and '-' characters; which are mapped to
  the zero vector.
  Args:
    amino_acid_residues: string. consisting of characters from
      AMINO_ACID_VOCABULARY
  Returns:
    A numpy array of shape (len(amino_acid_residues),
     len(AMINO_ACID_VOCABULARY)).
  Raises:
    KeyError: if amino_acid_residues has a character not in FULL_RESIDUE_VOCAB.
  """
  residue_encodings = _build_one_hot_encodings()
  int_sequence = residues_to_indices(amino_acid_residues)
  return residue_encodings[int_sequence]

def _build_one_hot_encodings():
  """Create array of one-hot embeddings.
  Row `i` of the returned array corresponds to the one-hot embedding of amino
    acid FULL_RESIDUE_VOCAB[i].
  Returns:
    np.array of shape `[len(FULL_RESIDUE_VOCAB), 20]`.
  """
  base_encodings = np.eye(len(AMINO_ACID_VOCABULARY))
  to_aa_index = AMINO_ACID_VOCABULARY.index

  special_mappings = {
      'B':
          .5 *
          (base_encodings[to_aa_index('D')] + base_encodings[to_aa_index('N')]),
      'Z':
          .5 *
          (base_encodings[to_aa_index('E')] + base_encodings[to_aa_index('Q')]),
      'X':
          np.ones(len(AMINO_ACID_VOCABULARY)) / len(AMINO_ACID_VOCABULARY),
      _PFAM_GAP_CHARACTER:
          np.zeros(len(AMINO_ACID_VOCABULARY)),
  }
  special_mappings['U'] = base_encodings[to_aa_index('C')]
  special_mappings['O'] = special_mappings['X']
  special_encodings = np.array(
      [special_mappings[c] for c in _ADDITIONAL_AA_VOCABULARY])
  return np.concatenate((base_encodings, special_encodings), axis=0)


# 
def read_original_data(path):
    """adapted from https://www.kaggle.com/code/petersarvari/protcnn-fast"""
    shards = []
    for fn in os.listdir(path):
        with open(os.path.join(path, fn)) as f:
            shards.append(pd.read_csv(f, index_col=None))
    return pd.concat(shards)

