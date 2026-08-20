import numpy as np
import pytest
from hdsemg_shared.filters.bandpass import bandpass_filter, bandpass_filter_exact_corners
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


def _min_fs(fc, N):
    """Smallest fs for which the normalised cut-off of `fc` stays below 1."""
    beta = (np.sqrt(2) - 1) ** (1 / N)
    return 2.0 * fc / beta


def test_bandpass_raises_when_fs_too_small_for_upper_cutoff():
    N, fcl, fch = 2, 3.0, 6.0
    fs_min = _min_fs(fch, N)  # 18.645... Hz
    data = np.random.default_rng(0).standard_normal(256)

    with pytest.raises(ValueError, match="higher cut-off frequency"):
        bandpass_filter(data, N=N, fcl=fcl, fch=fch, fs=fs_min * 0.99)


def test_bandpass_raises_in_the_gap_matlabs_own_guard_misses():
    # MATLAB's guard uses the UN-halved N and passes from 2*(-1+sqrt(2))**
    # (-(1/2)/N)*fc on (14.958 Hz here), but butter() only works from
    # 18.645 Hz on. Inside that gap MATLAB errors inside butter(); we must
    # error too rather than clip.
    N, fch = 2, 6.0
    matlab_guard = 2 * (-1 + np.sqrt(2)) ** (-(1 / 2) / N) * fch
    true_min = _min_fs(fch, N)
    assert matlab_guard < true_min  # the gap exists

    fs_in_gap = (matlab_guard + true_min) / 2
    data = np.random.default_rng(0).standard_normal(256)

    with pytest.raises(ValueError, match="sampling frequency is too small"):
        bandpass_filter(data, N=N, fcl=3.0, fch=fch, fs=fs_in_gap)


def test_bandpass_raises_when_fs_too_small_for_lower_cutoff():
    # fcl alone over the limit: only reachable when fch is over it as well,
    # so assert the LOWER cut-off is the one reported (MATLAB checks it first).
    N, fcl, fch = 2, 30.0, 450.0
    data = np.random.default_rng(0).standard_normal(256)

    with pytest.raises(ValueError, match="lower cut-off frequency"):
        bandpass_filter(data, N=N, fcl=fcl, fch=fch, fs=_min_fs(fcl, N) * 0.99)


def test_bandpass_accepts_fs_just_above_the_limit():
    N, fcl, fch = 2, 3.0, 6.0
    data = np.random.default_rng(0).standard_normal(256)

    out = bandpass_filter(data, N=N, fcl=fcl, fch=fch, fs=_min_fs(fch, N) * 1.01)

    assert out.shape == data.shape
    assert np.isfinite(out).all()


def test_bandpass_realistic_emg_settings_unaffected():
    # The guard must not fire for the project's normal settings.
    data = np.random.default_rng(0).standard_normal(4096)
    out = bandpass_filter(data, N=2, fcl=30, fch=450, fs=2048)
    assert np.isfinite(out).all()


def _realised_corner(fn, bracket, N, fcl, fch, fs):
    """The frequency inside `bracket` where fn's steady-state gain is -3 dB."""
    from scipy.optimize import brentq

    t = np.arange(int(8 * fs)) / fs
    middle = slice(len(t) // 4, 3 * len(t) // 4)

    def gain(f):
        x = np.sin(2 * np.pi * f * t)
        y = fn(x, N=N, fcl=fcl, fch=fch, fs=fs)
        return np.sqrt(np.mean(y[middle] ** 2)) / np.sqrt(np.mean(x[middle] ** 2))

    return brentq(lambda f: gain(f) - 1 / np.sqrt(2), *bracket)


@pytest.mark.parametrize("fcl,fch", [(15.0, 450.0), (30.0, 450.0)])
def test_exact_corners_land_on_the_requested_frequencies(fcl, fch):
    fs = 2048.0

    lower = _realised_corner(bandpass_filter_exact_corners, (5.0, 200.0), 2, fcl, fch, fs)
    upper = _realised_corner(bandpass_filter_exact_corners, (200.0, 900.0), 2, fcl, fch, fs)

    assert lower == pytest.approx(fcl, rel=0.01)
    assert upper == pytest.approx(fch, rel=0.01)


@pytest.mark.parametrize(
    "fcl,fch,realised_low,realised_high",
    [(15.0, 450.0, 35.3, 574.9), (30.0, 450.0, 68.8, 582.3)],
)
def test_prewarp_corners_still_land_where_matlab_puts_them(fcl, fch, realised_low, realised_high):
    # Regression lock on the untouched MATLAB replica. Ton van den Bogert's
    # pre-warp is derived for a SINGLE corner, so dividing BOTH band-pass
    # corners by beta < 1 pushes both up by 1/beta = 1.554. This is the
    # reason bandpass_filter_exact_corners exists; bandpass_filter must NOT
    # be "fixed", every current caller depends on these numbers.
    fs = 2048.0

    lower = _realised_corner(bandpass_filter, (5.0, 200.0), 2, fcl, fch, fs)
    upper = _realised_corner(bandpass_filter, (200.0, 900.0), 2, fcl, fch, fs)

    assert lower == pytest.approx(realised_low, rel=0.01)
    assert upper == pytest.approx(realised_high, rel=0.01)


def test_exact_corners_filters_each_row_of_a_matrix_on_its_own():
    fs = 2048.0
    t = np.arange(int(2 * fs)) / fs
    emg = np.vstack([np.sin(2 * np.pi * 5.0 * t), np.sin(2 * np.pi * 100.0 * t)])

    out = bandpass_filter_exact_corners(emg, 2, 15.0, 450.0, fs)

    assert out.shape == emg.shape
    assert np.sqrt(np.mean(out[0] ** 2)) < 0.2   # 5 Hz is rejected
    assert np.sqrt(np.mean(out[1] ** 2)) > 0.6   # 100 Hz passes


def test_exact_corners_rejects_bad_arguments():
    data = np.random.default_rng(0).standard_normal(4096)

    with pytest.raises(ValueError, match="even integer"):
        bandpass_filter_exact_corners(data, N=3, fcl=15.0, fch=450.0, fs=2048.0)
    with pytest.raises(ValueError, match="0 < fcl < fch < fs/2"):
        bandpass_filter_exact_corners(data, N=2, fcl=450.0, fch=15.0, fs=2048.0)
    with pytest.raises(ValueError, match="0 < fcl < fch < fs/2"):
        bandpass_filter_exact_corners(data, N=2, fcl=15.0, fch=1100.0, fs=2048.0)
