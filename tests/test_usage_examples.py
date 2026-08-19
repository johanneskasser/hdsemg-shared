"""
The integration examples from docs/usage/global_parameters.md, executed.

Each helper below is copied verbatim from that page. If one has to change
here, change it there too -- these tests exist so the documented recipes
cannot rot silently.
"""

import numpy as np
import pytest

from hdsemg_shared.global_parameters import global_amplitude
from hdsemg_shared.preprocessing.grid_map import emg_map_from_indices

FS = 2048.0
N_SAMPLES = int(2 * FS)


def _signal(n_channels):
    """One in-band 100 uV RMS sine per channel, each at its own frequency."""
    t = np.arange(N_SAMPLES) / FS
    return np.vstack(
        [100.0 * np.sqrt(2) * np.sin(2 * np.pi * (80 + i) * t) for i in range(n_channels)]
    )


def _steady(signal):
    return signal[len(signal) // 4:3 * len(signal) // 4]


# --------------------------------------------------------------------------
# From openhdemg
# --------------------------------------------------------------------------

def emg_map_from_code(n_channels, code, orientation):
    """The channel numbers of an OTB matrix, as an emg_map, at one orientation."""
    import pandas as pd
    import openhdemg.library as emg

    probe = {"RAW_SIGNAL": pd.DataFrame(np.arange(n_channels, dtype=float).reshape(1, -1))}
    sorted_probe = emg.sort_rawemg(probe, code=code, orientation=orientation,
                                   dividebycolumn=True)
    if isinstance(sorted_probe, Exception):
        raise ValueError(sorted_probe)   # sort_rawemg RETURNS its errors
    return np.array([sorted_probe["col{}".format(j)].iloc[0, :].to_numpy()
                     for j in range(len(sorted_probe))])


GR08MM1305_SORTING_ORDER_0 = [
    [63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51],
    [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    [37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25],
    [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, np.nan],
]
"""openhdemg's own base0_sorting_order for GR08MM1305 at orientation 0."""


def test_a_custom_sorting_order_is_an_emg_map_unchanged():
    # The FORMAT contract, pinned without needing openhdemg installed:
    # outer index is the column, inner index the row.
    out = global_amplitude(_signal(64), GR08MM1305_SORTING_ORDER_0, FS, derivation="SD")

    assert out.grid_shape == (5, 12)                # 13 rows -> 12 after one difference
    assert out.n_channels == 59                     # 4 full columns x 12, plus 11
    assert np.isfinite(out.amplitude).all()


def test_openhdemg_holds_the_rotation_as_a_reordered_grid():
    # electrodes.py stores one base0_sorting_order literal PER ORIENTATION,
    # alongside a base0_nanpos for the empty channel, so the rotation is data
    # openhdemg already owns. Ask sort_rawemg for the orientation you want;
    # never rotate a map yourself.
    pytest.importorskip("openhdemg.library", reason="openhdemg is an optional integration")

    at_0 = emg_map_from_code(64, "GR08MM1305", orientation=0)
    at_180 = emg_map_from_code(64, "GR08MM1305", orientation=180)

    # the sorting order quoted in the docs IS openhdemg's orientation-0 table
    np.testing.assert_array_equal(at_0, np.array(GR08MM1305_SORTING_ORDER_0, dtype=float))

    # the two orientations are the same physical grid, turned 180 degrees
    np.testing.assert_array_equal(np.rot90(at_180, 2), at_0)

    # ... so the ONE physical corner electrode changes index but stays a corner
    assert np.argwhere(np.isnan(at_0)).tolist() == [[4, 12]]
    assert np.argwhere(np.isnan(at_180)).tolist() == [[0, 0]]


@pytest.mark.parametrize("orientation", [0, 180])
@pytest.mark.parametrize(
    "derivation,direction,expected",
    [("SD", "cols", 59),   # 4 columns x 12, plus the gapped column's 11
     ("DD", "cols", 54),   # 4 columns x 11, plus 10
     ("SD", "rows", 51),   # 12 rows x 4, plus the gapped row's 3
     ("DD", "rows", 38)],  # 12 rows x 3, plus 2
)
def test_channel_counts_do_not_depend_on_the_orientation(orientation, derivation,
                                                         direction, expected):
    # A 180 degree turn maps a column onto a column, so it can move which INDEX
    # the missing electrode carries but not which PHYSICAL axis a derivation
    # runs along -- the counts must come out identical at 0 and at 180.
    pytest.importorskip("openhdemg.library", reason="openhdemg is an optional integration")

    emg_map = emg_map_from_code(64, "GR08MM1305", orientation=orientation)
    out = global_amplitude(_signal(64), emg_map, FS,
                           derivation=derivation, diff_direction=direction)

    assert out.n_channels == expected


@pytest.mark.parametrize(
    "code,n_channels,grid_shape", [("GR10MM0804", 32, (4, 8)), ("GR08MM1305", 64, (5, 13))]
)
def test_emg_map_from_code_recipe(code, n_channels, grid_shape):
    pytest.importorskip("openhdemg.library", reason="openhdemg is an optional integration")

    emg_map = emg_map_from_code(n_channels, code, orientation=0)

    assert emg_map.shape == grid_shape
    out = global_amplitude(_signal(n_channels), emg_map, FS)
    assert out.grid_shape == grid_shape
    assert _steady(out.amplitude).mean() == pytest.approx(100.0, rel=1e-2)


# --------------------------------------------------------------------------
# From hdsemg-pipe
# --------------------------------------------------------------------------

def test_the_pipe_recipe_over_every_grid_of_a_file():
    # pipe holds EMGFile.data (samples x channels) plus Grid objects carrying
    # a flat emg_indices and rows/cols, and nothing else about the geometry.
    class _Grid:
        def __init__(self, emg_indices, rows, cols, grid_key, ied_mm, muscle):
            self.emg_indices, self.rows, self.cols = emg_indices, rows, cols
            self.grid_key, self.ied_mm, self.muscle = grid_key, ied_mm, muscle

    data = np.vstack([_signal(24), np.full((2, N_SAMPLES), 1e6)]).T   # 2 reference channels
    grids = [_Grid(list(range(0, 12)), 4, 3, "10mm_4x3", 10, "VL"),
             _Grid(list(range(12, 24)), 4, 3, "10mm_4x3_2", 10, "VM")]

    emg_channels = data.T
    results = {}
    for grid in grids:
        emg_map = emg_map_from_indices(grid.emg_indices, grid.rows, grid.cols)
        out = global_amplitude(emg_channels, emg_map, FS, method="RMS", derivation="MP")
        results[grid.grid_key] = {
            "muscle": grid.muscle,
            "ied_mm": grid.ied_mm,
            "n_channels": out.n_channels,
            "mean_uv": float(out.amplitude.mean()),
            "per_channel_uv": {
                "{}_{}".format(col, row): float(envelope.mean())
                for (col, row), envelope in zip(out.positions, out.per_channel)
            },
        }

    assert set(results) == {"10mm_4x3", "10mm_4x3_2"}
    for entry in results.values():
        assert entry["n_channels"] == 12
        assert len(entry["per_channel_uv"]) == 12
        assert entry["mean_uv"] == pytest.approx(100.0, rel=0.05)

    import json
    json.dumps({"grids": results})   # the sidecar must be JSON-serialisable


# --------------------------------------------------------------------------
# From hdsemg-select
# --------------------------------------------------------------------------

def emg_map_from_selection(display_grid, emg_indices, channel_status):
    """select's display grid + selection mask, as an emg_map."""
    rows, cols = display_grid.shape
    emg_map = np.full((cols, rows), np.nan)         # emg_map is (nCols, nRows)
    for row in range(rows):
        for col in range(cols):
            local = display_grid[row, col]
            if np.isnan(local):
                continue                            # no electrode at this position
            channel = emg_indices[int(local)]
            if channel_status[channel]:             # keep only selected channels
                emg_map[col, row] = channel
    return emg_map


def test_the_select_recipe_resolves_local_indices_and_honours_the_mask():
    # display_grid entries are LOCAL indices into emg_indices, not global data
    # columns -- grid_setup_handler resolves them as indices[int(...)].
    emg = np.vstack([np.full((8, N_SAMPLES), 1e6), _signal(12)])
    emg_indices = list(range(8, 20))
    display_grid = np.array([[0.0, 4.0, 8.0],
                             [1.0, 5.0, 9.0],
                             [2.0, 6.0, 10.0],
                             [3.0, 7.0, np.nan]])
    channel_status = [True] * 20                    # references default to True
    channel_status[13] = False                      # one deselected EMG channel

    emg_map = emg_map_from_selection(display_grid, emg_indices, channel_status)

    assert emg_map.shape == (3, 4)
    assert np.isnan(emg_map[2, 3])                  # the absent corner
    assert np.isnan(emg_map[1, 1])                  # local 5 -> channel 13, deselected

    out = global_amplitude(emg, emg_map, FS, method="ARV")

    assert out.n_channels == 10
    assert _steady(out.amplitude).mean() == pytest.approx(
        2 * 100.0 * np.sqrt(2) / np.pi, rel=1e-2
    )


def test_the_select_density_frame_recipe():
    emg = _signal(12)
    emg_map = emg_map_from_indices(range(12), rows=4, cols=3)
    out = global_amplitude(emg, emg_map, FS, derivation="SD")
    sample_index = N_SAMPLES // 2

    n_cols, n_rows = out.grid_shape
    frame = np.full((n_rows, n_cols), np.nan)
    for (col, row), envelope in zip(out.positions, out.per_channel):
        frame[row, col] = envelope[sample_index]

    assert frame.shape == (3, 3)                     # SD along 'cols': 4 rows -> 3
    assert np.isfinite(frame).all()


def test_the_select_density_frame_recipe_tolerates_gaps():
    # With a gap the frame keeps a NaN hole rather than mis-placing a channel.
    emg = _signal(12)
    emg_map = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, np.nan]]
    out = global_amplitude(emg, emg_map, FS)

    n_cols, n_rows = out.grid_shape
    frame = np.full((n_rows, n_cols), np.nan)
    for (col, row), envelope in zip(out.positions, out.per_channel):
        frame[row, col] = envelope[N_SAMPLES // 2]

    assert frame.shape == (4, 3)
    assert np.isnan(frame[3, 2])                     # the gap
    assert np.isfinite(frame[:3, :]).all()


@pytest.mark.parametrize(
    "gap,expected_sd",
    [((4, 12), 59),   # gap at a column END   -> that column loses 1 difference
     ((4, 6), 58)],   # gap MID-column        -> it loses 2
)
def test_where_a_gap_sits_decides_how_many_differences_it_costs(gap, expected_sd):
    # 5 x 13 with one unwired position. SD along 'cols' gives 5 x 12 = 60
    # derived channels before the gap is accounted for. A gap at the first or
    # last row of its column breaks ONE adjacent pair; a gap in the middle
    # breaks TWO. GR08MM1305's missing electrode is a physical CORNER, so it
    # lands on an end at every orientation openhdemg offers and the grid
    # always reports 59 -- see
    # test_channel_counts_do_not_depend_on_the_orientation.
    emg_map = np.arange(65, dtype=float).reshape(5, 13)
    emg_map[gap] = np.nan

    out = global_amplitude(_signal(65), emg_map, FS, derivation="SD")

    assert out.grid_shape == (5, 12)
    assert out.n_channels == expected_sd


@pytest.mark.parametrize("orientation", [0, 180])
def test_the_gr08mm1305_gap_is_a_physical_corner_at_every_orientation(orientation):
    # 64 electrodes in 5 x 13 = 65 slots. The empty slot is a CORNER of the
    # physical array, so it stays a corner however the grid is turned -- first
    # or last row of a first or last column. That is what keeps the derived
    # channel counts orientation-independent.
    pytest.importorskip("openhdemg.library", reason="openhdemg is an optional integration")

    emg_map = emg_map_from_code(64, "GR08MM1305", orientation=orientation)
    gaps = np.argwhere(np.isnan(emg_map))

    assert len(gaps) == 1
    col, row = gaps[0]
    assert col in (0, emg_map.shape[0] - 1)          # first or last column
    assert row in (0, emg_map.shape[1] - 1)          # first or last row
