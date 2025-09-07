"""
Envelope computation for EMG signals using rectification and smoothing.

This module provides the function `compute_emg_envelopes()` to compute:
- ARV envelope (Average Rectified Value)
- RMS envelope (Root Mean Square)
- MDN envelope (Median absolute value)

The EMG signal is first bandpass filtered, then rectified, and finally smoothed via lowpass filtering.

References:
- Merletti & Farina (2016), Surface EMG in Movement Analysis
- CEDE Project: Amplitude Normalization Matrix, CEDE-Check

Usage example:
--------------
>>> from envelope import compute_emg_envelopes
>>> arv_env, rms_env, mdn_env = compute_emg_envelopes(emg_data, fs, bpf, lpf)

Where:
- emg_data: ndarray of shape (n_channels, n_samples)
- fs: sampling frequency in Hz
- bpf: dict with keys {'fcl', 'fch', 'N'} for bandpass filter
- lpf: dict with keys {'fc', 'N'} for lowpass filter
"""

import numpy as np
from scipy.signal import butter, filtfilt

def _bandpass_filter(signal: np.ndarray, fs: float, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    """
    Apply a bandpass Butterworth filter to the input signal.

    Parameters:
        signal (np.ndarray): 1D EMG signal.
        fs (float): Sampling frequency in Hz.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        order (int): Filter order. Default is 4.

    Returns:
        np.ndarray: Bandpass filtered signal.
    """
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return filtfilt(b, a, signal)

def _lowpass_filter(signal: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """
    Apply a lowpass Butterworth filter to the input signal.

    Parameters:
        signal (np.ndarray): 1D signal (already rectified).
        fs (float): Sampling frequency in Hz.
        cutoff (float): Cutoff frequency in Hz.
        order (int): Filter order. Default is 4.

    Returns:
        np.ndarray: Lowpass filtered (smoothed) signal.
    """
    b, a = butter(order, cutoff, btype='low', fs=fs)
    return filtfilt(b, a, signal)

def compute_emg_envelopes(emg: np.ndarray, fs: float, bpf: dict, lpf: dict):
    """
    Compute ARV, RMS, and MDN envelopes from multi-channel EMG data.

    The envelopes are calculated per channel after bandpass filtering and rectification:
    - ARV: Mean absolute value of the EMG signal
    - RMS: Root mean square of the EMG signal
    - MDN: Median absolute value of the EMG signal

    Each metric is smoothed using a lowpass filter to extract the envelope.

    Parameters:
        emg (np.ndarray): 2D EMG array with shape (n_channels, n_samples).
        fs (float): Sampling frequency in Hz.
        bpf (dict): Bandpass filter configuration with keys:
                    - 'fcl' (low cutoff frequency)
                    - 'fch' (high cutoff frequency)
                    - 'N' (filter order)
        lpf (dict): Lowpass filter configuration with keys:
                    - 'fc' (cutoff frequency)
                    - 'N' (filter order)

    Returns:
        tuple: Three 1D np.ndarrays (ARV envelope, RMS envelope, MDN envelope).
               Each has length equal to number of samples.
    """
    # Apply bandpass filter to each channel
    emg_filtered = np.array([
        _bandpass_filter(ch, fs, bpf['fcl'], bpf['fch'], bpf['N']) for ch in emg
    ])

    # Rectify and compute metrics across channels
    arv = np.nanmean(np.abs(emg_filtered), axis=0)
    rms = np.sqrt(np.nanmean(emg_filtered ** 2, axis=0))
    mdn = np.nanmedian(np.abs(emg_filtered), axis=0)

    # Smooth using lowpass filter
    arv_env = _lowpass_filter(arv, fs, lpf['fc'], lpf['N'])
    rms_env = _lowpass_filter(rms, fs, lpf['fc'], lpf['N'])
    mdn_env = _lowpass_filter(mdn, fs, lpf['fc'], lpf['N'])

    return arv_env, rms_env, mdn_env
