import numpy as np
import pytest
from hdsemg_shared.preprocessing.grid_map import emg_map_from_indices, map_to_columns


def test_map_to_columns_splits_in_map_order():
    emg = np.arange(60.0).reshape(12, 5)
    emg_map = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    columns = map_to_columns(emg, emg_map)

    assert len(columns) == 3
    assert all(c.shape == (4, 5) for c in columns)
    np.testing.assert_allclose(columns[1], emg[4:8])


def test_map_to_columns_turns_a_nan_entry_into_an_all_nan_row():
    emg = np.arange(60.0).reshape(12, 5)
    emg_map = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, np.nan]]

    columns = map_to_columns(emg, emg_map)

    assert np.all(np.isnan(columns[2][3]))
    np.testing.assert_allclose(columns[2][:3], emg[8:11])


def test_map_to_columns_ignores_channels_the_map_does_not_refer_to():
    # A recording carries force and path references next to the grid.
    emg = np.arange(100.0).reshape(20, 5)
    emg_map = [[10, 11], [12, 13]]

    columns = map_to_columns(emg, emg_map)

    np.testing.assert_allclose(np.vstack(columns), emg[10:14])


def test_map_to_columns_accepts_an_array_as_well_as_a_list():
    emg = np.arange(60.0).reshape(12, 5)

    from_list = map_to_columns(emg, [[0, 1], [2, 3]])
    from_array = map_to_columns(emg, np.array([[0.0, 1.0], [2.0, 3.0]]))

    np.testing.assert_allclose(np.vstack(from_list), np.vstack(from_array))


def test_emg_map_from_indices_column_order_matches_matlab_mat2grid():
    # MATLAB's mat2grid reshapes to [nRows, nCols] column-major, so the
    # channels run DOWN the first column before moving to the next.
    emg_map = emg_map_from_indices(range(12), rows=4, cols=3)

    assert emg_map.shape == (3, 4)
    np.testing.assert_allclose(emg_map[0], [0, 1, 2, 3])
    np.testing.assert_allclose(emg_map[2], [8, 9, 10, 11])


def test_emg_map_from_indices_row_order_transposes():
    emg_map = emg_map_from_indices(range(12), rows=4, cols=3, order="row")

    assert emg_map.shape == (3, 4)
    np.testing.assert_allclose(emg_map[0], [0, 3, 6, 9])


def test_emg_map_from_indices_round_trips_through_map_to_columns():
    # The pipe path: a Grid's flat emg_indices straight into the chain.
    emg = np.arange(120.0).reshape(24, 5)
    emg_indices = list(range(8, 20))

    columns = map_to_columns(emg, emg_map_from_indices(emg_indices, rows=4, cols=3))

    np.testing.assert_allclose(np.vstack(columns), emg[8:20])


def test_emg_map_from_indices_rejects_a_grid_with_an_unwired_position():
    # HD08MM1305 has 64 electrodes in 5x13 = 65 positions and needs an
    # explicit map with NaN at the empty one.
    with pytest.raises(ValueError, match="explicit map with NaN"):
        emg_map_from_indices(range(64), rows=13, cols=5)


def test_emg_map_from_indices_rejects_bad_geometry():
    with pytest.raises(ValueError, match="must be positive"):
        emg_map_from_indices(range(12), rows=0, cols=3)
    with pytest.raises(ValueError, match="order must be"):
        emg_map_from_indices(range(12), rows=4, cols=3, order="snake")


def test_map_to_columns_rejects_a_ragged_map():
    with pytest.raises(ValueError, match="ragged"):
        map_to_columns(np.zeros((12, 5)), [[0, 1], [2]])


def test_map_to_columns_rejects_a_repeated_channel():
    with pytest.raises(ValueError, match="more than once"):
        map_to_columns(np.zeros((12, 5)), [[0, 1], [1, 2]])


def test_map_to_columns_rejects_an_out_of_range_channel():
    with pytest.raises(ValueError, match="emg_channels holds only"):
        map_to_columns(np.zeros((12, 5)), [[0, 1], [2, 99]])


def test_map_to_columns_rejects_an_all_nan_map():
    with pytest.raises(ValueError, match="no channel at all"):
        map_to_columns(np.zeros((12, 5)), [[np.nan, np.nan]])


def test_map_to_columns_rejects_a_non_2d_channel_matrix():
    with pytest.raises(ValueError, match="must be 2-D"):
        map_to_columns(np.zeros(12), [[0, 1]])
