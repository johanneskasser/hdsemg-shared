import numpy as np
import pytest
from hdsemg_shared.filters.lowpass import lowpass_filter

FS = 2048.0


def _gain_at(fn, f, **kwargs):
    """Steady-state amplitude ratio of `fn` at frequency f, ends excluded."""
    t = np.arange(int(8 * FS)) / FS
    x = np.sin(2 * np.pi * f * t)
    y = fn(x, **kwargs)
    middle = slice(len(t) // 4, 3 * len(t) // 4)
    return np.sqrt(np.mean(y[middle] ** 2)) / np.sqrt(np.mean(x[middle] ** 2))


@pytest.mark.parametrize("N", [2, 4])
@pytest.mark.parametrize("fc", [6.0, 15.0])
def test_lowpass_realises_minus_3db_at_fc(N, fc):
    # A single corner is the case van den Bogert's pre-warp is exact for,
    # so no exact-corner variant is needed here (unlike the band-pass).
    gain = _gain_at(lowpass_filter, fc, N=N, fc=fc, fs=FS)

    assert gain == pytest.approx(1 / np.sqrt(2), rel=1e-3)


def test_lowpass_passes_below_and_rejects_above():
    assert _gain_at(lowpass_filter, 1.0, N=2, fc=15.0, fs=FS) > 0.99
    assert _gain_at(lowpass_filter, 200.0, N=2, fc=15.0, fs=FS) < 0.02  # 2-pole rolloff


def test_lowpass_filters_each_row_of_a_matrix_on_its_own():
    t = np.arange(int(2 * FS)) / FS
    emg = np.vstack([np.sin(2 * np.pi * 2.0 * t), np.sin(2 * np.pi * 300.0 * t)])

    out = lowpass_filter(emg, 2, 15.0, FS)

    assert out.shape == emg.shape
    assert np.sqrt(np.mean(out[0] ** 2)) > 0.6   # 2 Hz passes
    assert np.sqrt(np.mean(out[1] ** 2)) < 0.03  # 300 Hz does not


def test_lowpass_rejects_an_odd_order():
    with pytest.raises(ValueError, match="even integer"):
        lowpass_filter(np.zeros(256), N=3, fc=15.0, fs=FS)


def test_lowpass_rejects_non_positive_arguments():
    with pytest.raises(ValueError, match="must be positive"):
        lowpass_filter(np.zeros(256), N=2, fc=-1.0, fs=FS)


def test_lowpass_raises_when_fs_too_small_for_fc():
    # MATLAB's own guard uses the un-halved order and is weaker than the
    # filter needs; testing the normalised cutoff rejects the same inputs.
    N, fc = 2, 6.0
    fs_min = 2.0 * fc / (np.sqrt(2) - 1) ** (1 / N)

    with pytest.raises(ValueError, match="sampling frequency is too small"):
        lowpass_filter(np.zeros(256), N=N, fc=fc, fs=fs_min * 0.99)

    out = lowpass_filter(np.zeros(256), N=N, fc=fc, fs=fs_min * 1.01)
    assert np.isfinite(out).all()
