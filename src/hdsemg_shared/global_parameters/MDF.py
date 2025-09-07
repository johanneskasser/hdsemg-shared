"""
Compute the Median Frequency (MDF) of an EMG signal.

MDF is the frequency that divides the power spectral density (PSD)
into two regions with equal total power. It is commonly used in 
neuromuscular studies to assess fatigue or recruitment strategies.

References:
- Farina & Merletti (2003). Methods for estimating muscle fibre conduction velocity from surface electromyographic signals. *Medical & Biological Engineering & Computing*.
- CEDE Force Estimation Matrix, Martinez-Valdes et al. (2023).

Usage:
>>> mdf_value = compute_mdf(signal, fs)
"""

import numpy as np
from scipy.signal import welch

def compute_mdf(signal: np.ndarray, fs: float) -> float:
    """
    Compute the Median Frequency (MDF) from the power spectral density of an EMG signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional EMG signal (e.g., a single channel).
    fs : float
        Sampling frequency of the signal in Hz.

    Returns
    -------
    float
        Median Frequency (MDF) in Hz.

    Example
    -------
    >>> emg = np.random.randn(1024)
    >>> fs = 2000
    >>> mdf = compute_mdf(emg, fs)
    >>> print(f"MDF: {mdf:.2f} Hz")
    """
    f, pxx = welch(signal, fs=fs)
    cumsum = np.cumsum(pxx)
    mdf_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    return f[mdf_idx]
