"""
Compute Average Rectified Value (ARV) of a 1D EMG signal.

The Average Rectified Value (ARV) is a time-domain feature representing the
mean of the absolute values of the EMG signal. It provides a linear estimate 
of the EMG amplitude and is commonly used for assessing muscle activation levels.

ARV is often used in biomechanical modeling and fatigue analysis due to its
robustness against noise and its suitability in amplitude-based evaluations.

References:
- Dideriksen, J. L. et al. (2023). CEDE Amplitude Normalization Matrix.
- Clancy, E. A. et al. (2023). CEDE Amplitude Best Practice Recommendations.

Usage:
>>> arv = average_rectified_value(signal_segment)
"""

import numpy as np

def average_rectified_value(signal: np.ndarray) -> float:
    """
    Compute the Average Rectified Value (ARV) of a 1D EMG signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional EMG signal segment.

    Returns
    -------
    float
        ARV value (mean of rectified signal).

    Raises
    ------
    ValueError
        If the input signal is not one-dimensional.

    Example
    -------
    >>> emg = np.array([0.1, -0.2, 0.3, -0.4])
    >>> arv = average_rectified_value(emg)
    >>> print(f"ARV: {arv:.4f}")
    """
    if signal.ndim != 1:
        raise ValueError("Signal must be a 1D array.")
    return np.mean(np.abs(signal))
