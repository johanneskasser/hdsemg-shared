"""
Compute Integrated EMG (IEMG) from a rectified EMG signal.

The Integrated EMG (IEMG) is a time-domain feature that quantifies the total
muscle activation over a given period. It is computed as the sum of the
absolute (rectified) EMG signal. IEMG is commonly used to evaluate overall
muscle effort, compare activation levels, and assess fatigue or recovery.

References:
- Merletti, R., & Farina, D. (2016). Physiology of Muscle Activation and Force Generation.
- CEDE Project – Amplitude Normalization Matrix (2023)

Usage:
>>> iemg_value = integrated_emg(signal_segment)
"""

import numpy as np

def integrated_emg(signal: np.ndarray) -> float:
    """
    Compute the Integrated EMG (IEMG) of a 1D EMG signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional EMG signal (e.g., a windowed segment).

    Returns
    -------
    float
        Integrated EMG value (sum of rectified signal).

    Raises
    ------
    ValueError
        If input signal is not 1-dimensional.

    Example
    -------
    >>> emg = np.array([0.1, -0.2, 0.3, -0.4])
    >>> iemg = integrated_emg(emg)
    >>> print(f"IEMG: {iemg:.4f}")
    """
    if signal.ndim != 1:
        raise ValueError("Signal must be a 1D array.")
    return np.sum(np.abs(signal))
