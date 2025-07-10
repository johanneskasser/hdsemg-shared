"""
Root Mean Square (RMS) computation for EMG signal.

This function estimates the square root of the average squared signal amplitude over a segment.
It is used to quantify the energy content of the muscle activity and is sensitive to amplitude fluctuations.

Reference:
- Merletti & Farina, *Surface EMG: Physiology, Engineering and Applications*
- Clancy et al. (2023), CEDE recommendations

Usage:
>>> rms_value = root_mean_square(signal_segment)
"""

import numpy as np

def root_mean_square(signal: np.ndarray) -> float:
    if signal.ndim > 1:
        raise ValueError("Signal must be a 1D array.")
    return np.sqrt(np.mean(signal ** 2))
