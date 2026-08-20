import numpy as np
import pytest
from hdsemg_shared.filters.bandpass import bandpass_filter_exact_corners
from hdsemg_shared.filters.padding import pad_samples, reflect_pad, trim_pad


def test_reflect_pad_is_an_even_mirror():
    x = np.arange(10.0).reshape(1, 10)

    padded = reflect_pad(x, 3)

    assert padded.shape == (1, 16)
    np.testing.assert_allclose(padded[0, :3], [3, 2, 1])
    np.testing.assert_allclose(padded[0, -3:], [8, 7, 6])


def test_trim_pad_undoes_reflect_pad():
    x = np.arange(20.0).reshape(2, 10)

    np.testing.assert_allclose(trim_pad(reflect_pad(x, 4), 4), x)


def test_pad_samples_clips_to_what_the_signal_can_supply():
    # 0.75 s at 4 Hz would be 3 samples, and a 10-sample signal can supply them
    assert pad_samples(10, 4, 0.75) == 3
    # 10 s at 4 Hz cannot be reflected out of a 10-sample signal
    assert pad_samples(10, 4, 10.0) == 9
    assert pad_samples(10, 4, 0.0) == 0


def test_zero_or_negative_pad_is_a_no_op():
    x = np.arange(10.0)

    np.testing.assert_allclose(reflect_pad(x, 0), x)
    np.testing.assert_allclose(reflect_pad(x, -5), x)
    np.testing.assert_allclose(trim_pad(x, 0), x)


def test_reflect_pad_removes_the_bandpass_leading_edge_bias():
    # filtfilt pads by 3 samples only, far less than a 15-450 Hz band-pass
    # settles in, so the first samples come out inflated. Measured over 200
    # noise realisations the unpadded bias is ~8 %, the padded one well under 1 %.
    fs, n, pad = 2048.0, 4096, 512
    rng = np.random.default_rng(0)

    unpadded, padded = [], []
    for _ in range(200):
        x = rng.standard_normal(n)
        raw = bandpass_filter_exact_corners(x, 2, 15.0, 450.0, fs)
        guarded = trim_pad(
            bandpass_filter_exact_corners(reflect_pad(x, pad), 2, 15.0, 450.0, fs), pad
        )
        middle = np.mean(raw[n // 4:3 * n // 4] ** 2)
        unpadded.append(np.mean(raw[:64] ** 2) / middle)
        padded.append(np.mean(guarded[:64] ** 2) / middle)

    assert abs(np.mean(unpadded) - 1.0) > 0.04
    assert abs(np.mean(padded) - 1.0) < 0.02


def test_pad_survives_a_signal_shorter_than_the_requested_pad():
    x = np.arange(5.0)
    pad = pad_samples(x.size, 100.0, 1.0)

    np.testing.assert_allclose(trim_pad(reflect_pad(x, pad), pad), x)
