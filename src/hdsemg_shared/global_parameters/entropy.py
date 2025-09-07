"""
Compute the permutation entropy of an EMG signal.

This nonlinear complexity metric quantifies the irregularity in a time series.
It is often used in motor control, fatigue, or pathological muscle activation studies.

References:
- Bandt & Pompe (2002), "Permutation Entropy: A Natural Complexity Measure for Time Series", Physical Review Letters.
- Martinez-Valdes et al. (2023), CEDE Project – Single Motor Unit Matrix.

Usage:
>>> entropy_value = compute_entropy(emg_signal)

Returns a scalar value indicating the normalized permutation entropy.
"""

import numpy as np
from antropy import perm_entropy

def compute_entropy(signal: np.ndarray) -> float:
    """
    Compute the permutation entropy of a 1D EMG signal.

    Parameters:
    ----------
    signal : np.ndarray
        One-dimensional EMG signal (e.g., single channel time series).

    Returns:
    -------
    float
        Normalized permutation entropy (range: 0 to 1).
        Higher values indicate higher complexity.

    Example:
    -------
    >>> val = compute_entropy(signal)
    >>> print(f"Entropy: {val:.3f}")
    """
    return perm_entropy(signal, normalize=True)
