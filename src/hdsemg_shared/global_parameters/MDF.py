"""
Median Frequency (MDF) computation.

MDF is the frequency at which the EMG power spectrum is divided into two regions with equal power.
Useful for detecting muscle fatigue and neuromuscular efficiency.

References:
- Farina & Merletti (2003), Biomedical Engineering
- CEDE Force Estimation Matrix

Usage:
>>> mdf_value = compute_mdf(signal, fs)
"""

import numpy as np
from scipy.signal import welch

def compute_mdf(signal: np.ndarray, fs: float) -> float:
    f, pxx = welch(signal, fs=fs)
    cumsum = np.cumsum(pxx)
    mdf_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    return f[mdf_idx]
