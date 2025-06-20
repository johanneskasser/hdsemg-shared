import numpy as np
import pytest
from hdsemg_shared.filters.bandpass import bandpass_filter
from scipy.fft import rfft, rfftfreq

def test_bandpass_filter_basic():
    fs = 1000  # Hz
    t = np.linspace(0, 1.0, fs, endpoint=False)

    # Create a signal with 3 components: 10 Hz, 50 Hz, 200 Hz
    signal = (np.sin(2*np.pi*10*t) +
              np.sin(2*np.pi*50*t) +
              np.sin(2*np.pi*200*t))

    # Bandpass filter: only keep 40–100 Hz
    filtered = bandpass_filter(signal, N=4, fcl=40, fch=100, fs=fs)

    # FFT of original and filtered signals
    f = rfftfreq(len(t), 1/fs)
    fft_orig = np.abs(rfft(signal))
    fft_filt = np.abs(rfft(filtered))

    # Frequencies to check
    assert fft_filt[np.abs(f - 10).argmin()] < 0.1 * fft_orig[np.abs(f - 10).argmin()]
    assert fft_filt[np.abs(f - 200).argmin()] < 0.1 * fft_orig[np.abs(f - 200).argmin()]
    assert fft_filt[np.abs(f - 50).argmin()] > 0.5 * fft_orig[np.abs(f - 50).argmin()]

    # Output length unchanged
    assert filtered.shape == signal.shape