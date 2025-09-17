"""
Compute the Mean Frequency (MNF) of an EMG signal.

The Mean Frequency is the average frequency of the power spectral density (PSD)
and reflects the spectral centroid of an EMG signal. It is often used to monitor
muscle fatigue, changes in motor unit recruitment, and conduction velocity.

References:
- Phinyomark, A., et al. (2012). Muscle fatigue and feature extraction. *Expert Systems with Applications*, 39(3), 3003–3025.
- Farina, D., et al. (2023). Consensus for Experimental Design in Electromyography (CEDE): Estimating Muscle Force with EMG. *Journal of Electromyography and Kinesiology*.

Usage:
>>> mnf_value = compute_mnf(signal, fs)
"""

import numpy as np
from scipy.signal import welch

def compute_mnf(signal: np.ndarray, fs: float) -> float:
    """
    Compute the Mean Frequency (MNF) from the power spectral density of an EMG signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional EMG signal (e.g., a single channel).
    fs : float
        Sampling frequency of the signal in Hz.

    Returns
    -------
    float
        Mean Frequency (MNF) in Hz.

    Example
    -------
    >>> emg = np.random.randn(1024)
    >>> fs = 1000
    >>> mnf = compute_mnf(emg, fs)
    >>> print(f"MNF: {mnf:.2f} Hz")
    """
    f, pxx = welch(signal, fs=fs)
    return np.sum(f * pxx) / np.sum(pxx)
