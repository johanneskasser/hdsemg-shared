"""
Mean Frequency (MNF) computation for EMG signal.

MNF is computed from the power spectral density using Welch’s method.
It is sensitive to muscle fatigue and conduction velocity changes.

References:
- Phinyomark et al. (2012), Muscle fatigue and EMG features
- CEDE: Estimating Muscle Force with EMG (Farina et al., 2023)

Usage:
>>> mnf_value = compute_mnf(signal, fs)
"""

import numpy as np
from scipy.signal import welch

def compute_mnf(signal: np.ndarray, fs: float) -> float:
    f, pxx = welch(signal, fs=fs)
    return np.sum(f * pxx) / np.sum(pxx)
