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

    # The criterion that would fail. The margin is narrower than it looks
    # like it should be, and for a reason worth knowing: the anchor
    # regression drops pairs whose implied velocity is unphysiological, and
    # the pairs that STRADDLE an innervation zone are exactly those - their
    # delays partly cancel, so they imply a very high velocity. Tightening
    # MAX_CV_MS from 10 to 7 m/s therefore filters out much of what used to
    # collapse this R^2. It still separates - a clean grid returns 1.000
    # against 0.69-0.89 here - but it is no longer the dramatic collapse the
    # module docstring describes from the 10 m/s era.
    assert result.r_squared < 0.95
    clean = propagation(*travelling_grid(n_rows=12, cv_ms=4.0), IED_MM, FS)
    assert clean.r_squared > result.r_squared + 0.05


def test_innervation_zone_reports_a_matching_one_sided_velocity():
    mat, emg_map = travelling_grid(n_rows=14, iz_row=6, cv_ms=4.0)
    result = propagation(mat, emg_map, IED_MM, FS)

    assert result.cv_side_ms is not None
    assert result.cv_side_ms == pytest.approx(result.conduction_velocity_ms, rel=0.20)


def test_a_grid_without_an_innervation_zone_reports_no_side_velocity():
    mat, emg_map = travelling_grid(cv_ms=4.0)
    result = propagation(mat, emg_map, IED_MM, FS)
    assert result.cv_side_ms is None


def test_an_unusable_innervation_zone_split_reports_no_velocity_at_all():
    """
    A short grid split by an innervation zone leaves too few pairs on either
    side, and then the whole-grid median must NOT be handed over as the
    result: it averages pairs on both sides of the end plate, where the
    potentials travel in opposite directions.

    The danger is that the whole-grid figure looks entirely respectable -
    here it is ~4 m/s, mid-physiological - so nothing about the number
    itself would tell a reader it is meaningless.
    """
    mat, emg_map = travelling_grid(n_rows=8, iz_row=3, cv_ms=4.0)
    result = propagation(mat, emg_map, IED_MM, FS)

    assert result.cv_status == "iz_split"
    assert result.cv_side_ms is None
    assert np.isnan(result.cv_reported_ms)
    # the whole-grid value is still exposed, and still plausible-looking
    assert 3.0 <= result.conduction_velocity_ms <= 5.5


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


# ---------------------------------------------------------------------------
# the double differential
# ---------------------------------------------------------------------------

def test_the_double_differential_recovers_the_planted_velocity():
    """DD differences once more than SD, so it loses a bin, but the velocity
    it measures is the same one."""
    mat, emg_map = travelling_grid(n_rows=14, cv_ms=4.0)
    sd = propagation(mat, emg_map, IED_MM, FS, derivation="SD")
    dd = propagation(mat, emg_map, IED_MM, FS, derivation="DD")

    assert dd.cv_reported_ms == pytest.approx(effective_cv(4.0), rel=0.05)
    assert dd.fiber_angle_deg == pytest.approx(0.0, abs=5.0)
    assert dd.n_valid_pairs == sd.n_valid_pairs - 1


def test_the_double_differential_suppresses_a_common_component():
    """
    The reason it is worth having. A component every electrode sees at once
    does not travel, and it is what inflates a single-differential velocity.
    Adding one must disturb DD LESS than it disturbs SD.
    """
    rng = np.random.default_rng(7)
    mat, emg_map = travelling_grid(n_rows=14, cv_ms=4.0)
    common = _source(rng, scale=400.0)              # far larger than the signal
    contaminated = mat + common

    truth = effective_cv(4.0)
    sd_error = abs(propagation(contaminated, emg_map, IED_MM, FS,
                               derivation="SD").cv_reported_ms - truth)
    dd_error = abs(propagation(contaminated, emg_map, IED_MM, FS,
                               derivation="DD").cv_reported_ms - truth)

    assert dd_error <= sd_error


def test_an_unknown_derivation_is_named_not_guessed():
    mat, emg_map = travelling_grid()
    with pytest.raises(ValueError, match="'MP', 'SD' or 'DD'"):
        propagation(mat, emg_map, IED_MM, FS, derivation="XX")


# ---------------------------------------------------------------------------
# the physiological window
# ---------------------------------------------------------------------------

def test_the_physiological_window_is_two_to_seven():
    """
    The range H Penasso gives for an AVERAGE fibre conduction velocity, and
    the one the reference MATLAB implementation bounds its own search at
    (ML_CV_MulitCol's localac: CVmin = 2, CVmax = 7).
    """
    assert (MIN_CV_MS, MAX_CV_MS) == (2.0, 7.0)


def test_a_velocity_above_the_ceiling_is_dropped_not_reported():
    """
    A wave travelling far too fast for a muscle fibre must leave NO velocity
    behind, rather than one clipped to the ceiling. At 9 m/s every pair along
    the true direction is outside the window, so nothing survives.

    Asked ALONG THE PLANTED AXIS, deliberately. Turned loose, the free search
    answers -75 deg with a plausible 2.5 m/s: an oblique direction projects
    the electrodes onto a shorter spacing, which turns an impossible velocity
    into a possible one. That is the lattice artefact the module docstring
    warns about, and it is worth knowing that the physiological window does
    NOT protect against it - only fixing the axis does.
    """
    mat, emg_map = travelling_grid(n_rows=12, cv_ms=9.0, noise=0.5)
    result = propagation(mat, emg_map, IED_MM, FS, angles=[0.0])

    assert result.cv_status in ("out_of_range", "too_few_pairs")
    assert np.isnan(result.cv_reported_ms)


# ---------------------------------------------------------------------------
# bad channels, and the refusal to measure timing on an interpolated one
# ---------------------------------------------------------------------------

def _laplace_fill(mat, emg_map, channels):
    """
    Fill `channels` with the mean of their live 4-connected neighbours.

    One Jacobi sweep is enough for the tests here because the planted bad
    channels are never adjacent to each other; the real solver does the whole
    system at once, but the fixed point it converges to is this same relation.
    """
    out = np.array(mat, dtype=np.float64, copy=True)
    at = {(c, r): int(emg_map[c, r])
          for c in range(emg_map.shape[0]) for r in range(emg_map.shape[1])}
    for ch in channels:
        col, row = next(p for p, v in at.items() if v == ch)
        near = [at[(col + dc, row + dr)]
                for dc, dr in ((0, 1), (0, -1), (1, 0), (-1, 0))
                if (col + dc, row + dr) in at]
        out[ch] = np.mean(mat[near], axis=0)
    return out


def test_bad_channels_takes_them_out_of_the_map():
    """
    Naming a channel bad must do exactly what NaN-ing the map position does.
    Anything else and the argument would be a second, subtly different way of
    excluding a channel.
    """
    mat, emg_map = travelling_grid(cv_ms=4.0)
    bad = [emg_map[1, 3], emg_map[2, 5]]

    by_argument = propagation(mat, emg_map, IED_MM, FS, bad_channels=bad)
    hand_masked = np.array(emg_map, dtype=float, copy=True)
    hand_masked[1, 3] = np.nan
    hand_masked[2, 5] = np.nan
    by_map = propagation(mat, hand_masked, IED_MM, FS)

    assert by_argument.n_electrodes == by_map.n_electrodes
    assert by_argument.cv_reported_ms == pytest.approx(by_map.cv_reported_ms,
                                                       nan_ok=True)
    assert by_argument.fiber_angle_deg == by_map.fiber_angle_deg


def test_bad_channels_accepts_a_boolean_mask():
    mat, emg_map = travelling_grid(cv_ms=4.0)
    mask = np.zeros(mat.shape[0], dtype=bool)
    mask[int(emg_map[1, 3])] = True

    by_mask = propagation(mat, emg_map, IED_MM, FS, bad_channels=mask)
    by_number = propagation(mat, emg_map, IED_MM, FS,
                            bad_channels=[emg_map[1, 3]])

    assert by_mask.n_electrodes == by_number.n_electrodes
    assert by_mask.cv_reported_ms == pytest.approx(by_number.cv_reported_ms,
                                                   nan_ok=True)


def test_bad_channels_outside_the_data_is_named_not_ignored():
    mat, emg_map = travelling_grid(cv_ms=4.0)

    with pytest.raises(ValueError, match="outside the data"):
        propagation(mat, emg_map, IED_MM, FS, bad_channels=[mat.shape[0] + 4])


def test_an_interpolated_channel_is_refused():
    """
    THE TRAP THIS EXISTS FOR. An amplitude pipeline has good reason to
    Laplace-fill its bad channels, and handing the same filled matrix to
    propagation is the natural next line. A filled channel is roughly the
    mean of its neighbours, so its second spatial difference collapses and
    the bin carries noise where a delay should be.
    """
    mat, emg_map = travelling_grid(cv_ms=4.0)
    bad = [int(emg_map[1, 3]), int(emg_map[2, 5])]
    filled = _laplace_fill(mat, emg_map, bad)

    with pytest.raises(ValueError, match="mean of their live neighbours") as e:
        propagation(filled, emg_map, IED_MM, FS)

    assert all(str(ch) in str(e.value) for ch in bad), "must name which ones"


def test_a_filled_matrix_measures_the_same_as_a_raw_one_once_declared():
    """
    Declaring the bad channels neutralises the fill COMPLETELY: those
    positions leave the map, so the filled values are never read and the
    answer is the one the raw matrix gives.
    """
    mat, emg_map = travelling_grid(cv_ms=4.0)
    bad = [int(emg_map[1, 3]), int(emg_map[2, 5])]
    filled = _laplace_fill(mat, emg_map, bad)

    from_filled = propagation(filled, emg_map, IED_MM, FS, bad_channels=bad)
    from_raw = propagation(mat, emg_map, IED_MM, FS, bad_channels=bad)

    assert from_filled.cv_reported_ms == pytest.approx(from_raw.cv_reported_ms,
                                                       nan_ok=True)
    assert from_filled.propagation_score == pytest.approx(
        from_raw.propagation_score)


def test_the_guard_can_be_overridden_deliberately():
    """The escape hatch exists, but it has to be asked for by name."""
    mat, emg_map = travelling_grid(cv_ms=4.0)
    filled = _laplace_fill(mat, emg_map, [int(emg_map[1, 3])])

    result = propagation(filled, emg_map, IED_MM, FS, allow_interpolated=True)

    assert result.n_electrodes == mat.shape[0]


def test_the_guard_does_not_fire_on_ordinary_data():
    """
    A false positive here would break every honest caller. Real signals miss
    their neighbour mean by orders of magnitude more than the tolerance, and
    a planted travelling wave is the hardest case for that - every channel IS
    a delayed copy of one source.
    """
    for cv in (2.5, 4.0, 6.0):
        for noise in (0.0, 2.0):
            mat, emg_map = travelling_grid(cv_ms=cv, noise=noise)
            propagation(mat, emg_map, IED_MM, FS)      # must not raise


def test_a_channel_with_one_live_neighbour_is_not_called_interpolated():
    """
    With a single neighbour a fill is a copy, which cannot be told apart from
    a genuine duplicate-wiring fault. Refusing there would be guessing, so
    the guard needs two neighbours before it accuses anything.
    """
    mat, emg_map = travelling_grid(n_cols=1, n_rows=12, cv_ms=4.0)
    mat[3] = mat[2]                       # a copy of its only live neighbour

    propagation(mat, emg_map, IED_MM, FS)                # must not raise


def test_duplicate_wiring_is_diagnosed_as_such_not_as_interpolation():
    """
    A channel identical to its identical neighbours trivially equals their
    mean, without anything having been interpolated. That is duplicate
    wiring, which propagation already reports as an unusable velocity, and
    the guard must not replace that correct diagnosis with a wrong one.
    """
    rng = np.random.default_rng(8)
    mat = np.vstack([_source(rng)] * 48)
    emg_map = np.arange(48, dtype=float).reshape(4, 12)

    result = propagation(mat, emg_map, IED_MM, FS)      # must not raise

    assert result.cv_status in ("out_of_range", "too_few_pairs")
