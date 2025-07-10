"""
Compute EMG signal envelope using rectification + lowpass filtering.

The function returns ARV, RMS and Median envelopes for multi-channel EMG.

References:
- Merletti & Farina (2016), Surface EMG in Movement Analysis
- CEDE-Check & Amplitude Normalization Matrix

Usage:
>>> arv_env, rms_env, mdn_env = compute_emg_envelopes(emg, fs, bpf, lpf)
"""

import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return filtfilt(b, a, signal)

def lowpass_filter(signal, fs, cutoff, order=4):
    b, a = butter(order, cutoff, btype='low', fs=fs)
    return filtfilt(b, a, signal)

def compute_emg_envelopes(emg: np.ndarray, fs: float, bpf: dict, lpf: dict):
    emg_filtered = np.array([
        bandpass_filter(ch, fs, bpf['fcl'], bpf['fch'], bpf['N']) for ch in emg
    ])
    
    arv = np.nanmean(np.abs(emg_filtered), axis=0)
    rms = np.sqrt(np.nanmean(emg_filtered ** 2, axis=0))
    mdn = np.nanmedian(np.abs(emg_filtered), axis=0)

    arv_env = lowpass_filter(arv, fs, lpf['fc'], lpf['N'])
    rms_env = lowpass_filter(rms, fs, lpf['fc'], lpf['N'])
    mdn_env = lowpass_filter(mdn, fs, lpf['fc'], lpf['N'])

    return arv_env, rms_env, mdn_env
