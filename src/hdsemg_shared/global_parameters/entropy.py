"""
Permutation Entropy of EMG signal.

This nonlinear measure quantifies signal complexity and is often used in motor control studies.
It is robust to noise and insensitive to amplitude scaling.

References:
- Bandt & Pompe (2002), Physical Review Letters
- CEDE Single Motor Unit Matrix (Martinez-Valdes et al., 2023)

Usage:
>>> entropy_value = compute_entropy(signal)
"""

from antropy import perm_entropy
import numpy as np

def compute_entropy(signal: np.ndarray) -> float:
    return perm_entropy(signal, normalize=True)
