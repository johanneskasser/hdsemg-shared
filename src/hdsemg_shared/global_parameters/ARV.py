"""
Average Rectified Value (ARV) computation for EMG signal.

ARV is the mean of the absolute values of the signal. It is a linear envelope estimator,
robust for amplitude-based assessments of muscle activation.

References:
- CEDE Amplitude Normalization Matrix (Dideriksen et al., 2023)
- Clancy et al. (2023), CEDE Amplitude Best Practice

Usage:
>>> arv_value = average_rectified_value(signal_segment)
"""

import numpy as np

def average_rectified_value(signal: np.ndarray) -> float:
    if signal.ndim > 1:
        raise ValueError("Signal must be a 1D array.")
    return np.mean(np.abs(signal))
