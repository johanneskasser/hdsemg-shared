"""
Unit tests for hdsemg_shared.quality.propagation.

The grids here carry a PLANTED travelling wave: one band-limited source
repeated at every electrode, delayed by the distance the wave has covered.
The conduction velocity, the fibre direction and the innervation zone
position are therefore all known in advance, and each test asserts the
measurement recovers the one it was built for.

The delay resolution is one sample, so at 2048 Hz and 10 mm spacing a
velocity is only resolved to a few per cent; the tolerances say so.
"""

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from hdsemg_shared.quality import MAX_CV_MS, MIN_CV_MS, propagation

FS = 2048.0
N_SAMPLES = int(4 * FS)
IED_MM = 10.0


def _source(rng, scale=50.0):
    b, a = butter(2, [20 / (FS / 2), 200 / (FS / 2)], btype="band")
    return filtfilt(b, a, rng.standard_normal(N_SAMPLES)) * scale


def effective_cv(cv_ms):
    """
    The velocity a grid can actually carry at this sampling rate.

    A shift is a whole number of samples, so a nominal 6 m/s over 10 mm at
    2048 Hz (3.41 samples) is planted as 3 samples and IS 6.83 m/s. Tests
    compare against this, not against the nominal figure, otherwise they
    assert the measurement is wrong by exactly the quantisation the
    generator introduced.
    """
    shift = int(round((IED_MM * 1e-3) / cv_ms * FS))
    return (IED_MM * 1e-3) / (shift / FS)


def travelling_grid(n_cols=4, n_rows=12, cv_ms=4.0, iz_row=None, along="rows",
                    noise=2.0, seed=1):
    """
    A grid with a known travelling wave.

      along  ... "rows" makes it travel DOWN a grid column (0 deg), "cols"
                  ACROSS the columns (90 deg)
      iz_row ... plant an innervation zone at this row: the wave then leaves
                  it in BOTH directions, as it does under a real end-plate
                  region. None means a single direction throughout
    """
    rng = np.random.default_rng(seed)
    source = _source(rng)
    emg_map = np.arange(n_cols * n_rows, dtype=float).reshape(n_cols, n_rows)
    step_samples = int(round((IED_MM * 1e-3) / cv_ms * FS))
    mat = np.empty((n_cols * n_rows, N_SAMPLES))
    for col in range(n_cols):
        for row in range(n_rows):
            if along == "cols":
                steps = col
            elif iz_row is None:
                steps = row
            else:
                steps = abs(row - iz_row)
            mat[int(emg_map[col, row])] = (
                np.roll(source, steps * step_samples)
                + rng.standard_normal(N_SAMPLES) * noise
            )
    return mat, emg_map


# ---------------------------------------------------------------------------
# the clean case
# ---------------------------------------------------------------------------

def test_recovers_the_planted_direction_and_velocity():
    mat, emg_map = travelling_grid(cv_ms=4.0)
    result = propagation(mat, emg_map, IED_MM, FS)

    assert result.fiber_angle_deg == pytest.approx(0.0, abs=5.0)
    assert result.conduction_velocity_ms == pytest.approx(effective_cv(4.0), rel=0.05)
    assert result.cv_status == "ok"
    assert not result.iz_detected
    assert result.propagation_score > 0.9
    assert result.n_electrodes == 48


@pytest.mark.parametrize("cv_ms", [3.0, 4.0, 5.0, 6.0])
def test_recovers_a_range_of_velocities(cv_ms):
    mat, emg_map = travelling_grid(cv_ms=cv_ms)
    result = propagation(mat, emg_map, IED_MM, FS)
    assert result.conduction_velocity_ms == pytest.approx(effective_cv(cv_ms), rel=0.05)
    assert result.cv_status == "ok"


def test_recovers_propagation_across_the_columns_as_90_degrees():
    mat, emg_map = travelling_grid(n_cols=12, n_rows=4, along="cols")
    result = propagation(mat, emg_map, IED_MM, FS)
    assert abs(result.fiber_angle_deg) == pytest.approx(90.0, abs=5.0)
    assert result.conduction_velocity_ms == pytest.approx(effective_cv(4.0), rel=0.05)


# ---------------------------------------------------------------------------
# the innervation zone, the case this module exists for
# ---------------------------------------------------------------------------

def test_innervation_zone_is_found_at_the_row_it_was_planted_at():
    mat, emg_map = travelling_grid(n_rows=12, iz_row=5, cv_ms=4.0)
    result = propagation(mat, emg_map, IED_MM, FS)

    assert result.iz_detected
    assert result.cv_status == "iz_split"
    assert result.iz_position_m == pytest.approx(5 * IED_MM * 1e-3, abs=IED_MM * 1e-3)


def test_innervation_zone_does_not_destroy_the_direction_or_the_velocity():
    """
    The regression a straight-line criterion would select on collapses under
    an innervation zone; the delay-consistency criterion does not. Both are
    asserted here, because it is their DISAGREEMENT that identifies the case.
    """
    mat, emg_map = travelling_grid(n_rows=12, iz_row=5, cv_ms=4.0)
    result = propagation(mat, emg_map, IED_MM, FS)

    assert result.fiber_angle_deg == pytest.approx(0.0, abs=5.0)
    assert result.conduction_velocity_ms == pytest.approx(effective_cv(4.0), rel=0.05)
    assert result.propagation_score > 0.9      # the criterion that is used
    assert result.r_squared < 0.7              # the criterion that would fail


def test_innervation_zone_reports_a_matching_one_sided_velocity():
    mat, emg_map = travelling_grid(n_rows=14, iz_row=6, cv_ms=4.0)
    result = propagation(mat, emg_map, IED_MM, FS)

    assert result.cv_side_ms is not None
    assert result.cv_side_ms == pytest.approx(result.conduction_velocity_ms, rel=0.20)


def test_a_grid_without_an_innervation_zone_reports_no_side_velocity():
    mat, emg_map = travelling_grid(cv_ms=4.0)
    result = propagation(mat, emg_map, IED_MM, FS)
    assert result.cv_side_ms is None


# ---------------------------------------------------------------------------
# the degenerate cases, which must be named rather than guessed at
# ---------------------------------------------------------------------------

def test_pure_noise_scores_no_propagation():
    rng = np.random.default_rng(7)
    mat = np.vstack([_source(rng) for _ in range(48)])
    emg_map = np.arange(48, dtype=float).reshape(4, 12)

    result = propagation(mat, emg_map, IED_MM, FS)
    assert result.propagation_score < 0.5
    assert result.cv_status != "ok"


def test_identical_channels_are_not_reported_as_infinitely_fast():
    """
    Zero delay implies infinite velocity. It must be rejected, not clipped
    into the physiological range and reported as a measurement.
    """
    rng = np.random.default_rng(8)
    one = _source(rng)
    mat = np.vstack([one] * 48)
    emg_map = np.arange(48, dtype=float).reshape(4, 12)

    result = propagation(mat, emg_map, IED_MM, FS)
    assert result.cv_status in ("out_of_range", "too_few_pairs")
    assert not (MIN_CV_MS <= result.conduction_velocity_ms <= MAX_CV_MS)


def test_a_grid_with_too_few_live_electrodes_is_named_not_guessed():
    mat, emg_map = travelling_grid(n_cols=1, n_rows=2)
    result = propagation(mat, emg_map, IED_MM, FS)
    assert result.cv_status == "too_few_pairs"
    assert np.isnan(result.conduction_velocity_ms)


def test_dead_channels_are_dropped_without_being_masked_first():
    mat, emg_map = travelling_grid(cv_ms=4.0)
    mat[3] = 0.0
    mat[20] = np.nan

    result = propagation(mat, emg_map, IED_MM, FS)
    assert result.n_electrodes == 46
    assert result.conduction_velocity_ms == pytest.approx(effective_cv(4.0), rel=0.05)


def test_unwired_map_positions_are_simply_not_placed():
    mat, emg_map = travelling_grid(cv_ms=4.0)
    emg_map[0, 0] = np.nan
    emg_map[2, 5] = np.nan

    result = propagation(mat, emg_map, IED_MM, FS)
    assert result.n_electrodes == 46
    assert result.cv_status == "ok"


# ---------------------------------------------------------------------------
# argument handling
# ---------------------------------------------------------------------------

def test_the_window_is_honoured():
    mat, emg_map = travelling_grid(cv_ms=4.0)
    result = propagation(mat, emg_map, IED_MM, FS, window=slice(0, N_SAMPLES // 2))
    assert result.conduction_velocity_ms == pytest.approx(effective_cv(4.0), rel=0.05)


def test_unfiltered_mode_still_measures_the_planted_velocity():
    mat, emg_map = travelling_grid(cv_ms=4.0)
    result = propagation(mat, emg_map, IED_MM, FS, bpf=False)
    assert result.conduction_velocity_ms == pytest.approx(effective_cv(4.0), rel=0.05)


def test_a_restricted_angle_search_is_honoured():
    mat, emg_map = travelling_grid(cv_ms=4.0)
    result = propagation(mat, emg_map, IED_MM, FS, angles=np.arange(-10.0, 11.0))
    assert result.search_angles.size == 21
    assert result.search_score.size == 21
    assert result.fiber_angle_deg == pytest.approx(0.0, abs=5.0)


def test_a_negative_inter_electrode_distance_is_rejected():
    mat, emg_map = travelling_grid()
    with pytest.raises(ValueError, match="ied_mm must be positive"):
        propagation(mat, emg_map, -1.0, FS)


def test_a_map_naming_a_missing_channel_is_rejected():
    mat, emg_map = travelling_grid()
    emg_map[0, 0] = 9999.0
    with pytest.raises(ValueError, match="emg_map refers to channels"):
        propagation(mat, emg_map, IED_MM, FS)
