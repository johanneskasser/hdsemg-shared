"""
GRID_MAP: turning a channel matrix into HDsEMG grid columns

  An emg_map says which channel of a recording sits at which electrode
  position. The format follows openhdemg's custom_sorting_order: a list of
  lists (or a 2-D array) where the OUTER index is the grid COLUMN and the
  INNER index is the ROW, holding base-0 channel numbers, with NaN at
  positions that carry no electrode or were excluded.

    emg_map = [[0, 1,  2,  3     ],   # col0
               [4, 5,  6,  7     ],   # col1
               [8, 9, 10, np.nan]]    # col2, one gap

  Nothing here filters or differentiates - it only selects and orders.

  ADDITION of the Python implementation, no MATLAB equivalent (MATLAB's
  mat2grid takes gridSize from the caller's input structure).

  (c) H Penasso. Written for hdsemg-shared by Claude Opus 5, 2026-08-19,
  from the MATLAB conventions of Global-HDsEMG-Analysis.
"""

import numpy as np


def map_to_columns(emg_channels, emg_map):
    """
    MAP_TO_COLUMNS splits a channel matrix into one matrix per grid column

      columns = map_to_columns(emg.T, emg_map)

      INPUT
        emg_channels ... matrix, each row contains one channel's signal over
                          time (rows = channels, columns = samples). May hold
                          more channels than the map refers to, e.g. force or
                          path references [xV]
        emg_map      ... list of lists or 2-D array, outer index = grid
                          column, inner index = row, entries are base-0
                          channel numbers into emg_channels, NaN where no
                          electrode sits []

      OUTPUT
        columns      ... list of nCols matrices, each nRows-by-nSamples, in
                          map order. A NaN map entry becomes an all-NaN row
                          [xV]

      RAISES
        ValueError ... if the map is ragged or not 2-D, if an entry is not a
                        valid row index of emg_channels, or if a channel is
                        used more than once

      *INFO* ... EMGFile.data is samples-by-channels, so pass its transpose
      *INFO* ... excluding a channel means putting NaN in its map position;
                  global_amplitude then drops it, matching the MATLAB
                  '*INFO* NaN channels are ignored'
    """
    emg = np.asarray(emg_channels, dtype=np.float64)
    if emg.ndim != 2:
        raise ValueError(f"emg_channels must be 2-D (channels x samples), got {emg.ndim}-D.")

    grid = _as_map(emg_map)
    _check_indices(grid, emg.shape[0])

    # Build one matrix per column, NaN rows where the map has a gap
    columns = []
    for col in grid:
        block = np.full((col.size, emg.shape[1]), np.nan, dtype=np.float64)
        wired = ~np.isnan(col)
        block[wired] = emg[col[wired].astype(int)]
        columns.append(block)
    return columns


def emg_map_from_indices(emg_indices, rows, cols, order = 'column'):
    """
    EMG_MAP_FROM_INDICES builds an emg_map from a flat list of channel numbers

      emg_map = emg_map_from_indices(grid.emg_indices, grid.rows, grid.cols)

      INPUT
        emg_indices ... sequence of base-0 channel numbers, in acquisition
                         order, exactly rows*cols of them []
        rows        ... number of electrode rows of the grid []
        cols        ... number of electrode columns of the grid []
        order       ... char, 'column' (default) if the channels run down the
                         first column before moving to the next, 'row' if
                         they run along the first row []

      OUTPUT
        emg_map     ... 2-D array, cols-by-rows, base-0 channel numbers, ready
                         for map_to_columns []

      RAISES
        ValueError ... if rows or cols are not positive, if order is unknown,
                        or if len(emg_indices) != rows*cols

      *INFO* ... grids with an unwired position, e.g. HD08MM1305 with 64
                  electrodes in 5x13 = 65 positions, cannot be described this
                  way. Pass an explicit map with NaN at the empty position
                  instead - openhdemg's sort_rawemg and hdsemg-select's
                  electrode display grid both produce one
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be positive, got {rows} and {cols}.")
    if order not in ('column', 'row'):
        raise ValueError(f"order must be 'column' or 'row', got {order!r}.")

    idx = np.asarray(emg_indices, dtype=np.float64).ravel()
    if idx.size != rows * cols:
        raise ValueError(
            f"emg_indices holds {idx.size} channels but rows*cols is {rows * cols}. "
            "A grid with an unwired position needs an explicit map with NaN there."
        )

    if order == 'column':
        return idx.reshape(cols, rows)

    # Acquisition ran along the rows, so reshape row-wise and transpose
    return idx.reshape(rows, cols).T


def _as_map(emg_map):
    """The map as a float array, cols-by-rows, rejecting ragged input."""
    grid = np.asarray(emg_map, dtype=object) if isinstance(emg_map, list) else np.asarray(emg_map)
    try:
        grid = np.asarray([np.asarray(col, dtype=np.float64).ravel() for col in grid])
    except ValueError as err:
        raise ValueError("emg_map is ragged; every column must hold the same number of rows.") from err

    if grid.ndim != 2 or grid.dtype == object:
        raise ValueError("emg_map must be 2-D: outer index = column, inner index = row.")
    if grid.size == 0:
        raise ValueError("emg_map is empty.")
    return grid.astype(np.float64)


def _check_indices(grid, n_channels):
    """Every finite entry a valid, unique row index of the channel matrix."""
    wired = grid[~np.isnan(grid)]
    if wired.size == 0:
        raise ValueError("emg_map holds no channel at all, every position is NaN.")
    if not np.all(wired == np.round(wired)):
        raise ValueError("emg_map entries must be whole channel numbers or NaN.")
    if wired.min() < 0 or wired.max() >= n_channels:
        raise ValueError(
            f"emg_map refers to channels {int(wired.min())}..{int(wired.max())} but "
            f"emg_channels holds only {n_channels}."
        )
    if np.unique(wired).size != wired.size:
        raise ValueError("emg_map uses at least one channel more than once.")
