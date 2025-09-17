"""
Compute Root Mean Square (RMS) value of a 1D EMG signal.

The Root Mean Square (RMS) quantifies the energy content of an EMG signal
by calculating the square root of the mean squared values. RMS is commonly
used to assess muscle activation level and fatigue, being particularly
sensitive to amplitude changes and signal energy.

References:
- Merletti & Farina, *Surface EMG: Physiology, Engineering and Applications*
- Clancy et al. (2023), CEDE Amplitude Best Practice Recommendations

Usage:
>>> rms = root_mean_square(signal_segment)
"""

import numpy as np

def root_mean_square(signal: np.ndarray) -> float:
    """
    Compute the Root Mean Square (RMS) of a 1D EMG signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional EMG signal segment.

    Returns
    -------
    float
        RMS value (energy-related amplitude measure).

    Raises
    ------
    ValueError
        If the input signal is not one-dimensional.

    Example
    -------
    >>> emg = np.array([0.1, -0.2, 0.3, -0.4])
    >>> rms = root_mean_square(emg)
    >>> print(f"RMS: {rms:.4f}")
    """
    if signal.ndim != 1:
        raise ValueError("Signal must be a 1D array.")
    return np.sqrt(np.mean(np.square(signal)))
