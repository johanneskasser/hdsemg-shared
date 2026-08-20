import numpy as np
import pytest
from hdsemg_shared.filters.lowpass import lowpass_filter
from hdsemg_shared.filters.smoothing import moving_average, window_length_for_cutoff

FS = 2048.0


def _gain_at(f, **kwargs):
    """Steady-state amplitude ratio of moving_average at frequency f."""
    t = np.arange(int(8 * FS)) / FS
    x = np.sin(2 * np.pi * f * t)
    y = moving_average(x, FS, **kwargs)
    middle = slice(len(t) // 4, 3 * len(t) // 4)
    return np.sqrt(np.mean(y[middle] ** 2)) / np.sqrt(np.mean(x[middle] ** 2))


@pytest.mark.parametrize("kernel", ["bidirectional", "rectangular"])
@pytest.mark.parametrize("fc", [6.0, 15.0])
def test_moving_average_and_lowpass_share_the_bandwidth(fc, kernel):
    # Both are asked for the same fc, so both must attenuate a sine at fc
    # equally -- the kernel choice changes shape, never bandwidth. The 2 %
    # tolerance is the rounding of the window to a whole number of samples.
    t = np.arange(int(8 * FS)) / FS
    x = np.sin(2 * np.pi * fc * t)
    middle = slice(len(t) // 4, 3 * len(t) // 4)
    reference = lowpass_filter(x, 2, fc, FS)
    lowpass_gain = np.sqrt(np.mean(reference[middle] ** 2)) / np.sqrt(np.mean(x[middle] ** 2))

    assert _gain_at(fc, fc=fc, kernel=kernel) == pytest.approx(lowpass_gain, rel=0.02)
    assert lowpass_gain == pytest.approx(1 / np.sqrt(2), rel=1e-3)


def test_window_length_for_cutoff_matches_the_rule_of_thumb():
    # w ~ 0.319 * fs / fc for the default two-pass kernel
    assert window_length_for_cutoff(15.0, FS) == 44
    assert window_length_for_cutoff(6.0, FS) == 109


def test_rectangular_window_as_long_as_the_signal_is_the_plain_mean():
    # This is what makes the epoch amplitude and the amplitude time series
    # the same quantity; the triangular kernel weighs the middle more.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 500))

    out = moving_average(x, 100.0, window_s=5.0, kernel="rectangular")

    np.testing.assert_allclose(out[:, 250], x.mean(axis=1), atol=1e-12)


def test_moving_average_is_exactly_zero_phase():
    t = np.arange(1024) / FS
    x = np.exp(-((t - t.mean()) ** 2) / 1e-4)  # symmetric about its centre

    out = moving_average(x, FS, fc=50.0)

    np.testing.assert_allclose(out, out[::-1], atol=1e-12)


def test_moving_average_does_not_taper_the_ends_to_zero():
    # The ends are divided by the kernel weight that actually overlapped them.
    x = np.ones((2, 400))

    out = moving_average(x, FS, fc=100.0)

    np.testing.assert_allclose(out, 1.0, atol=1e-12)


def test_moving_average_smooths_each_row_on_its_own():
    x = np.vstack([np.ones(400), np.zeros(400)])

    out = moving_average(x, FS, fc=100.0)

    np.testing.assert_allclose(out[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(out[1], 0.0, atol=1e-12)


def test_moving_average_needs_exactly_one_of_window_s_and_fc():
    x = np.zeros(256)

    with pytest.raises(ValueError, match="exactly one"):
        moving_average(x, FS)
    with pytest.raises(ValueError, match="exactly one"):
        moving_average(x, FS, window_s=1.0, fc=15.0)


def test_smoothing_rejects_an_unknown_kernel():
    with pytest.raises(ValueError, match="kernel must be one of"):
        moving_average(np.zeros(256), FS, fc=15.0, kernel="triangular")


def test_window_length_rejects_a_cutoff_at_or_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        window_length_for_cutoff(FS / 2, FS)
