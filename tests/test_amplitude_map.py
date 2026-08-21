"""
Unit tests for hdsemg_shared.quality.amplitude_map.

The grids here carry a PLANTED amplitude pattern rather than a planted
travelling wave: one band-limited source scaled per electrode, so the map
the measurement should return is known in advance, and where the innervation
zone dip sits is known too.

The direction tests matter most. A single differential taken across the
columns instead of along them is a silent fault: it still produces a map, an
innervation zone and a barycentre, all of them wrong, and nothing about the
numbers themselves says so. So the axis is asserted explicitly.
"""

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from hdsemg_shared.quality import (
    amplitude_map,
    barycenter,
    innervation_zone_line,
    upsample_map,
)

FS = 2048.0
N_SAMPLES = int(2 * FS)
IED_MM = 10.0
N_COLS, N_ROWS = 4, 8


def _source(rng, scale=50.0):
    b, a = butter(2, [20 / (FS / 2), 200 / (FS / 2)], btype="band")
    return filtfilt(b, a, rng.standard_normal(N_SAMPLES)) * scale


def scaled_grid(row_gain=None, col_gain=None, n_cols=N_COLS, n_rows=N_ROWS,
                seed=0):
    """
    A grid whose MONOPOLAR channels are one shared source scaled per position.

    Neighbouring channels are then identical up to a factor, so their single
    differential is that same source scaled by the DIFFERENCE of the two
    gains: the map is predictable in closed form.
    """
    rng = np.random.default_rng(seed)
    source = _source(rng)
    row_gain = np.ones(n_rows) if row_gain is None else np.asarray(row_gain, float)
    col_gain = np.ones(n_cols) if col_gain is None else np.asarray(col_gain, float)
    emg_map = np.arange(n_cols * n_rows, dtype=float).reshape(n_cols, n_rows)
    mat = np.empty((n_cols * n_rows, N_SAMPLES))
    for c in range(n_cols):
        for r in range(n_rows):
            mat[int(emg_map[c, r])] = source * row_gain[r] * col_gain[c]
    return mat, emg_map


# ---------------------------------------------------------------------------
# the axis the differential is taken along
# ---------------------------------------------------------------------------

def test_the_differential_is_taken_along_a_grid_column_not_across_them():
    """
    Channels identical DOWN each column but different ACROSS columns must
    give a single differential of nothing at all. If the difference were
    taken across the columns instead, this map would be strongly non-zero -
    which is exactly the fault the test exists to catch.
    """
    mat, emg_map = scaled_grid(col_gain=[1.0, 4.0, 9.0, 16.0])
    amap = amplitude_map(mat, emg_map, IED_MM, FS)

    reference = amplitude_map(mat, emg_map, IED_MM, FS, derivation="MP")
    assert amap.values.max() < reference.values.max() * 1e-6


def test_a_gradient_down_the_column_survives_the_differential():
    """The mirror of the previous test: variation along the column is what a
    single differential is supposed to keep."""
    mat, emg_map = scaled_grid(row_gain=[1, 2, 3, 4, 5, 6, 7, 8])
    amap = amplitude_map(mat, emg_map, IED_MM, FS)

    # every SD is the source times a gain step of exactly 1, so the map is flat
    assert np.ptp(amap.values) < amap.values.mean() * 1e-6
    assert amap.values.mean() > 0


# ---------------------------------------------------------------------------
# the geometry the map is laid out on
# ---------------------------------------------------------------------------

def test_single_differential_positions_sit_between_the_electrodes():
    mat, emg_map = scaled_grid()
    amap = amplitude_map(mat, emg_map, IED_MM, FS)

    assert amap.values.shape == (N_ROWS - 1, N_COLS)
    assert amap.y_mm == pytest.approx(np.arange(N_ROWS - 1) * IED_MM + IED_MM / 2)
    assert amap.x_mm == pytest.approx(np.arange(N_COLS) * IED_MM)


def test_double_differential_positions_sit_on_the_electrodes_one_in():
    mat, emg_map = scaled_grid()
    amap = amplitude_map(mat, emg_map, IED_MM, FS, derivation="DD")

    assert amap.values.shape == (N_ROWS - 2, N_COLS)
    assert amap.y_mm == pytest.approx(np.arange(N_ROWS - 2) * IED_MM + IED_MM)


def test_monopolar_keeps_every_electrode_position():
    mat, emg_map = scaled_grid()
    amap = amplitude_map(mat, emg_map, IED_MM, FS, derivation="MP")

    assert amap.values.shape == (N_ROWS, N_COLS)
    assert amap.y_mm == pytest.approx(np.arange(N_ROWS) * IED_MM)


def test_a_grid_too_short_for_the_derivation_is_named_not_guessed():
    mat, emg_map = scaled_grid(n_rows=2)
    with pytest.raises(ValueError, match="cannot carry"):
        amplitude_map(mat, emg_map, IED_MM, FS, derivation="DD")


# ---------------------------------------------------------------------------
# the innervation zone
# ---------------------------------------------------------------------------

def _dip_grid(dip_row, n_rows=N_ROWS):
    """
    Gains that rise, pause, then rise again, so consecutive differences are
    1 everywhere except across dip_row where they are 0.1: an amplitude
    minimum at a known position, as an end plate makes.
    """
    steps = np.ones(n_rows - 1)
    steps[dip_row] = 0.1
    return np.concatenate([[1.0], 1.0 + np.cumsum(steps)])


def test_the_innervation_zone_is_found_at_the_planted_dip():
    mat, emg_map = scaled_grid(row_gain=_dip_grid(3))
    amap = amplitude_map(mat, emg_map, IED_MM, FS)
    iz = innervation_zone_line(amap)

    assert iz.full_width
    assert iz.center_xy_mm[1] == pytest.approx(amap.y_mm[3], abs=1e-6)


def test_the_innervation_zone_survives_upsampling_at_the_same_place():
    mat, emg_map = scaled_grid(row_gain=_dip_grid(4))
    amap = amplitude_map(mat, emg_map, IED_MM, FS)
    coarse = innervation_zone_line(amap)
    fine = innervation_zone_line(upsample_map(amap))

    assert fine.center_xy_mm[1] == pytest.approx(coarse.center_xy_mm[1], abs=1.0)


def test_the_electrode_spacing_survives_upsampling():
    """
    The continuity tolerance is half an INTER-ELECTRODE distance. After
    upsampling, the spacing of x_mm is 0.1 mm, and reading the tolerance off
    the axis would make it 200 times too strict, so the map carries the real
    spacing with it.
    """
    mat, emg_map = scaled_grid(row_gain=_dip_grid(3))
    fine = upsample_map(amplitude_map(mat, emg_map, IED_MM, FS))

    assert fine.ied_mm == IED_MM
    assert fine.x_mm[1] - fine.x_mm[0] == pytest.approx(0.1)
    assert innervation_zone_line(fine).full_width


def test_a_column_falling_monotonically_offers_no_innervation_zone():
    """
    The lowest value in a monotonic column is its last electrode, which is
    the edge of the grid and not an end plate. A local-minimum rule returns
    nothing here; a plain min() would return the edge.
    """
    mat, emg_map = scaled_grid(row_gain=np.arange(N_ROWS, 0, -1) * 1.0)
    iz = innervation_zone_line(amplitude_map(mat, emg_map, IED_MM, FS))

    assert iz.columns_covered == 0
    assert np.isnan(iz.center_xy_mm[1])


def test_dips_scattered_between_columns_are_not_called_a_line():
    """An end plate is a band across the muscle. Unrelated dips one full
    inter-electrode distance apart from column to column are not."""
    mat, emg_map = scaled_grid()
    amap = amplitude_map(mat, emg_map, IED_MM, FS)
    values = np.tile(np.arange(1.0, amap.values.shape[0] + 1)[:, None],
                     (1, amap.values.shape[1]))
    for c in range(values.shape[1]):
        values[(c * 2) % values.shape[0], c] = 0.0     # a dip that jumps about
    iz = innervation_zone_line(amap._replace(values=values))

    assert not iz.full_width


def test_a_monopolar_map_is_refused_rather_than_answered():
    mat, emg_map = scaled_grid(row_gain=_dip_grid(3))
    amap = amplitude_map(mat, emg_map, IED_MM, FS, derivation="MP")
    with pytest.raises(ValueError, match="DIFFERENTIAL"):
        innervation_zone_line(amap)


# ---------------------------------------------------------------------------
# missing electrodes, measures, barycentre
# ---------------------------------------------------------------------------

def test_an_unwired_position_is_left_missing_rather_than_filled_with_zero():
    mat, emg_map = scaled_grid()
    emg_map[1, 3] = np.nan
    amap = amplitude_map(mat, emg_map, IED_MM, FS)

    # the two single differentials that would have used that electrode
    assert np.isnan(amap.values[2, 1])
    assert np.isnan(amap.values[3, 1])
    assert np.isfinite(amap.values[0, 1])


def test_upsampling_fills_a_missing_position_from_its_neighbours():
    mat, emg_map = scaled_grid()
    emg_map[1, 3] = np.nan
    fine = upsample_map(amplitude_map(mat, emg_map, IED_MM, FS))

    assert np.all(np.isfinite(fine.values))


def test_arv_is_smaller_than_rms_on_the_same_signal():
    mat, emg_map = scaled_grid(row_gain=np.arange(1.0, N_ROWS + 1))
    rms = amplitude_map(mat, emg_map, IED_MM, FS, measure="RMS")
    arv = amplitude_map(mat, emg_map, IED_MM, FS, measure="ARV")

    assert np.all(arv.values < rms.values)


def test_the_barycentre_follows_the_strong_side_of_the_grid():
    # a gradient down the column, identical in every column: the single
    # differential map is flat, so the barycentre must sit dead centre
    mat, emg_map = scaled_grid(row_gain=np.arange(1.0, N_ROWS + 1))
    amap = amplitude_map(mat, emg_map, IED_MM, FS)
    centre = (N_COLS - 1) * IED_MM / 2
    assert barycenter(amap)[0] == pytest.approx(centre, abs=1e-6)

    values = amap.values.copy()
    values[:, 0] *= 50.0                       # make the first column dominate
    assert barycenter(amap._replace(values=values))[0] < centre


def test_the_window_is_honoured():
    mat, emg_map = scaled_grid(row_gain=np.arange(1.0, N_ROWS + 1))
    mat[:, : int(FS)] = 0.0
    quiet = amplitude_map(mat, emg_map, IED_MM, FS, window=slice(0, int(FS) // 2))
    loud = amplitude_map(mat, emg_map, IED_MM, FS,
                         window=slice(int(1.5 * FS), N_SAMPLES))

    assert np.nanmax(quiet.values) < np.nanmax(loud.values) * 1e-3


@pytest.mark.parametrize("kwargs, match", [
    (dict(ied_mm=-1.0), "ied_mm"),
    (dict(derivation="XX"), "derivation"),
    (dict(measure="XX"), "measure"),
])
def test_bad_arguments_are_named(kwargs, match):
    mat, emg_map = scaled_grid()
    call = dict(ied_mm=IED_MM)
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        amplitude_map(mat, emg_map, call.pop("ied_mm"), FS, **call)
