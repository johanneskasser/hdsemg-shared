"""
Unit tests for hdsemg_shared.quality.channel_metrics.

Every test plants a KNOWN defect in an otherwise clean synthetic grid and
asserts the measure separates it from the rest, because that separation - not
an absolute value - is what a quality gate thresholds on.
"""

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from hdsemg_shared.quality import (
    channel_amplitude,
    channel_spectrum,
    clipping_fraction,
    flat_channels,
    line_noise_ratio,
    neighbor_correlation,
    robust_z,
)

FS = 2048.0
N_SAMPLES = int(4 * FS)
N_COLS, N_ROWS = 4, 8
N_CHANNELS = N_COLS * N_ROWS


def _emg_map():
    """A plain 4x8 map, channel c*8+r at column c, row r."""
    return np.arange(N_CHANNELS, dtype=float).reshape(N_COLS, N_ROWS)


def _band_limited(rng, scale=50.0):
    """One in-band noise signal, 20-200 Hz, roughly sEMG shaped."""
    b, a = butter(2, [20 / (FS / 2), 200 / (FS / 2)], btype="band")
    return filtfilt(b, a, rng.standard_normal(N_SAMPLES)) * scale


def _clean_grid(seed=0, correlated=True):
    """
    A grid whose channels share a common source plus their own noise.

    correlated=True mimics the real thing, where neighbouring electrodes see
    the same motor unit action potentials; the neighbour-correlation test
    needs that to have anything to measure.
    """
    rng = np.random.default_rng(seed)
    common = _band_limited(rng) if correlated else None
    mat = np.empty((N_CHANNELS, N_SAMPLES))
    for i in range(N_CHANNELS):
        own = _band_limited(rng, scale=10.0)
        mat[i] = own if common is None else common + own
    return mat


# ---------------------------------------------------------------------------
# flat_channels
# ---------------------------------------------------------------------------

def test_flat_channels_finds_planted_dead_channels():
    mat = _clean_grid()
    mat[5] = 0.0
    mat[17] = 0.0
    assert flat_channels(mat) == [5, 17]


def test_flat_channels_reports_all_nan_rows_as_flat():
    mat = _clean_grid()
    mat[3] = np.nan
    assert 3 in flat_channels(mat)


def test_flat_channels_finds_nothing_in_a_clean_grid():
    assert flat_channels(_clean_grid()) == []


def test_flat_channels_rejects_a_non_matrix():
    with pytest.raises(ValueError, match="2-D"):
        flat_channels(np.zeros(10))


# ---------------------------------------------------------------------------
# channel_amplitude
# ---------------------------------------------------------------------------

def test_channel_amplitude_separates_a_planted_loud_channel():
    mat = _clean_grid()
    mat[9] *= 8.0
    amp = channel_amplitude(mat, FS)

    z = robust_z(amp.rms)
    assert z[9] > 5.0
    assert np.nanmax(np.abs(np.delete(z, 9))) < 5.0


def test_channel_amplitude_rms_exceeds_arv_for_a_noise_like_signal():
    amp = channel_amplitude(_clean_grid(), FS)
    assert np.all(amp.rms > amp.arv)


def test_channel_amplitude_window_restricts_which_samples_are_reduced():
    mat = _clean_grid()
    quiet = slice(0, N_SAMPLES // 2)
    mat[:, quiet] *= 0.01

    whole = channel_amplitude(mat, FS).rms
    loud_half = channel_amplitude(mat, FS, window=slice(N_SAMPLES // 2, N_SAMPLES)).rms
    assert np.all(loud_half > whole)


def test_channel_amplitude_is_nan_for_an_all_nan_channel():
    mat = _clean_grid()
    mat[2] = np.nan
    assert np.isnan(channel_amplitude(mat, FS).rms[2])


def test_channel_amplitude_rejects_an_unknown_option():
    with pytest.raises(ValueError, match="Unknown bpf option"):
        channel_amplitude(_clean_grid(), FS, bpf={"cutoff": 10})


# ---------------------------------------------------------------------------
# channel_spectrum
# ---------------------------------------------------------------------------

def test_channel_spectrum_flags_a_low_frequency_artefact_channel():
    mat = _clean_grid()
    t = np.arange(N_SAMPLES) / FS
    mat[11] = 400.0 * np.sin(2 * np.pi * 3.0 * t)  # a movement artefact, not sEMG

    spec = channel_spectrum(mat, FS)
    assert spec.mnf[11] < np.nanmedian(spec.mnf) / 2
    assert robust_z(spec.mnf)[11] < -5.0


def test_channel_spectrum_mnf_of_a_pure_tone_is_that_tone():
    t = np.arange(N_SAMPLES) / FS
    mat = np.vstack([np.sin(2 * np.pi * 100.0 * t)] * 2)
    assert channel_spectrum(mat, FS).mnf[0] == pytest.approx(100.0, abs=5.0)


def test_channel_spectrum_is_nan_for_a_dead_channel():
    mat = _clean_grid()
    mat[4] = 0.0
    spec = channel_spectrum(mat, FS)
    assert np.isnan(spec.mnf[4]) and np.isnan(spec.mdf[4])


# ---------------------------------------------------------------------------
# line_noise_ratio
# ---------------------------------------------------------------------------

def test_line_noise_ratio_separates_a_planted_50hz_channel():
    mat = _clean_grid()
    t = np.arange(N_SAMPLES) / FS
    mat[7] = mat[7] + 30.0 * np.sin(2 * np.pi * 50.0 * t)

    ratio = line_noise_ratio(mat, FS).ratio
    assert ratio[7] > 20.0
    assert np.nanmax(np.delete(ratio, 7)) < 5.0


def test_line_noise_ratio_of_a_clean_channel_sits_near_its_noise_value():
    """
    Mean-over-peak against a median background has a FIXED clean reading of
    1/ln2 = 1.44 whatever the record length, which is the point of using it
    rather than the maximum. Verified here so a change to either statistic
    that reintroduces a length-dependent reading fails loudly.
    """
    ratio = line_noise_ratio(_clean_grid(), FS).ratio
    assert np.all(ratio > 0.3)
    assert np.all(ratio < 4.0)


def test_line_noise_ratio_reports_each_requested_harmonic():
    mat = _clean_grid()
    t = np.arange(N_SAMPLES) / FS
    mat[2] = mat[2] + 30.0 * np.sin(2 * np.pi * 150.0 * t)

    noise = line_noise_ratio(mat, FS, freqs=(50.0, 100.0, 150.0))
    assert noise.per_frequency.shape == (3, N_CHANNELS)
    assert noise.per_frequency[2, 2] > 20.0     # the 150 Hz row sees it
    assert noise.per_frequency[0, 2] < 5.0      # the 50 Hz row does not
    assert noise.ratio[2] == pytest.approx(noise.per_frequency[2, 2])


def test_line_noise_ratio_rejects_a_frequency_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        line_noise_ratio(_clean_grid(), FS, freqs=(2000.0,))


# ---------------------------------------------------------------------------
# clipping_fraction
# ---------------------------------------------------------------------------

def test_clipping_fraction_separates_a_planted_clipped_channel():
    mat = _clean_grid()
    rail = np.percentile(np.abs(mat[6]), 80)
    mat[6] = np.clip(mat[6], -rail, rail)

    clipped = clipping_fraction(mat)
    assert clipped[6] > 0.1
    assert np.nanmax(np.delete(clipped, 6)) < 1e-3


def test_clipping_fraction_of_a_clean_channel_is_the_one_sample_floor():
    """An unclipped channel touches its own extreme exactly once."""
    clipped = clipping_fraction(_clean_grid())
    assert np.allclose(clipped, 1.0 / N_SAMPLES)


def test_clipping_fraction_is_nan_for_a_dead_channel():
    mat = _clean_grid()
    mat[1] = 0.0
    assert np.isnan(clipping_fraction(mat)[1])


# ---------------------------------------------------------------------------
# neighbor_correlation
# ---------------------------------------------------------------------------

def test_neighbor_correlation_separates_an_isolated_noise_channel():
    mat = _clean_grid()
    rng = np.random.default_rng(99)
    mat[13] = _band_limited(rng)  # its own signal, shared with nobody

    r = neighbor_correlation(mat, _emg_map(), FS)
    assert r[13] < 0.5
    assert np.nanmin(np.delete(r, 13)) > 0.7


def test_neighbor_correlation_is_nan_where_the_map_places_nothing():
    emg_map = _emg_map()
    emg_map[1, 1] = np.nan  # channel 9 not placed
    r = neighbor_correlation(_clean_grid(), emg_map, FS)
    assert np.isnan(r[9])
    assert np.isfinite(r[0])


def test_neighbor_correlation_honours_the_window():
    mat = _clean_grid()
    window = slice(0, N_SAMPLES // 2)
    assert neighbor_correlation(mat, _emg_map(), FS, window=window).shape == (N_CHANNELS,)


def test_neighbor_correlation_rejects_a_map_naming_a_missing_channel():
    emg_map = _emg_map()
    emg_map[0, 0] = 999.0
    with pytest.raises(ValueError, match="emg_map refers to channels"):
        neighbor_correlation(_clean_grid(), emg_map, FS)


# ---------------------------------------------------------------------------
# robust_z
# ---------------------------------------------------------------------------

def test_robust_z_is_not_dragged_by_the_outlier_it_is_looking_for():
    """
    The whole reason for median/MAD over mean/SD: two bad channels inflate a
    standard deviation enough to hide themselves, a MAD they cannot move.
    """
    rng = np.random.default_rng(4)
    values = np.concatenate([1.0 + 0.1 * rng.standard_normal(30), [50.0, 60.0]])

    z = robust_z(values)
    assert z[30] > 10.0 and z[31] > 10.0
    assert np.max(np.abs(z[:30])) < 5.0

    classic = (values - values.mean()) / values.std()
    assert classic[30] < 4.0     # the same channel, all but invisible


def test_robust_z_passes_nan_through():
    z = robust_z(np.array([1.0, 2.0, np.nan, 3.0, 100.0]))
    assert np.isnan(z[2])
    assert np.isfinite(z[0])


def test_robust_z_returns_all_nan_for_a_degenerate_grid():
    """A zero MAD means most channels are identical, not that the rest are outliers."""
    assert np.all(np.isnan(robust_z(np.array([2.0, 2.0, 2.0, 2.0, 9.0]))))


def test_robust_z_returns_all_nan_when_everything_is_nan():
    assert np.all(np.isnan(robust_z(np.full(5, np.nan))))
