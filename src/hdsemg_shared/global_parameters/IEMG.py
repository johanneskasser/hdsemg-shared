"""
Integrated EMG (IEMG) computation.

IEMG reflects the total electrical activity in a muscle during a period.
It is calculated as the sum of the rectified signal.

References:
- Merletti & Farina (2016), Physiology of Muscle Activation and Force Generation
- CEDE Amplitude Normalization Matrix

Usage:
>>> iemg_value = integrated_emg(signal_segment)
"""

import numpy as np

def integrated_emg(signal: np.ndarray) -> float:
    if signal.ndim > 1:
        raise ValueError("Signal must be a 1D array.")
    return np.sum(np.abs(signal))
