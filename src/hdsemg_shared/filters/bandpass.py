import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(data: np.ndarray, order: int, lowcut: float, highcut: float, fs: float) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to 1D data.

    Parameters
    ----------
    data : np.ndarray
        1D array of signal samples.
    order : int
        Filter order.
    lowcut : float
        Lower cutoff frequency (Hz).
    highcut : float
        Upper cutoff frequency (Hz).
    fs : float
        Sampling rate (Hz).

    Returns
    -------
    np.ndarray
        The filtered signal, same shape as `data`.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output="sos", fs=fs)
    return sosfiltfilt(sos, data, axis=-1)