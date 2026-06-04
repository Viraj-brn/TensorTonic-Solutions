import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.empty((0, max_len or 0))

    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    N = len(seqs)
    padded_matrix = np.full((N, max_len), pad_value)
    # Your code here
    for i, seq in enumerate(seqs):
        truc_seq = seq[:max_len]

        padded_matrix[i, :len(truc_seq)] = truc_seq

    return padded_matrix
    pass