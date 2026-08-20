# Preprocessing

The `preprocessing` package provides utilities for preparing and transforming HD-sEMG signals before analysis. Currently, it focuses on differential signal computation, which is a common preprocessing step in EMG analysis.

---

## Differential Analysis

The package provides functionality to compute differential signals from HD-sEMG channels, with optional filtering. This is particularly useful for reducing common-mode noise and enhancing local muscle activity detection.

### API Reference

```python
def to_differential(
    mats: list[np.ndarray],
    sr: float,
    f: dict[str, float]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute differential channels with optional filtering."""
```

#### Parameters

- **mats** (`list[np.ndarray]`): 
    - List of 2D arrays containing EMG signals
    - Each array shape: (j, T), where:
        - j: number of channels
        - T: number of time samples
    - Type: float

- **sr** (`float`): 
    - Sampling rate in Hz
    - Used for the bandpass filter if applied

- **f** (`dict[str, float]`): 
    - Filter parameters dictionary with keys:
        - 'n': filter order (int)
        - 'low': lower cutoff frequency in Hz
        - 'up': upper cutoff frequency in Hz

#### Returns

- **dmat** (`list[np.ndarray]`):
    - List of filtered differential signals
    - Each array shape: (j-1, T)
    - Bandpass filtered using specified parameters

- **dmat_no_filter** (`list[np.ndarray]`):
    - List of unfiltered differential signals
    - Each array shape: (j-1, T)
    - Raw differences between adjacent channels

### Example Usage

```python
import numpy as np
from hdsemg_shared.preprocessing.differential import to_differential

# Create sample EMG data (2 matrices, 8 channels each, 1000 samples)
emg1 = np.random.randn(8, 1000)
emg2 = np.random.randn(8, 1000)
mats = [emg1, emg2]

# Define filter parameters
filter_params = {
    'n': 4,      # 4th order Butterworth
    'low': 20,   # 20 Hz highpass
    'up': 450    # 450 Hz lowpass
}

# Compute differential signals
diff_filtered, diff_raw = to_differential(
    mats=mats,
    sr=2000,     # 2kHz sampling rate
    f=filter_params
)

# Results:
# - diff_filtered[0].shape == (7, 1000)  # Filtered differentials from emg1
# - diff_filtered[1].shape == (7, 1000)  # Filtered differentials from emg2
# - diff_raw[0].shape == (7, 1000)       # Raw differentials from emg1
# - diff_raw[1].shape == (7, 1000)       # Raw differentials from emg2
```

### Implementation Details

The differential computation:
1. Takes adjacent pairs of channels (j and j+1)
2. Computes their difference (channel_j+1 - channel_j)
3. Optionally applies a bandpass filter to the differential signals
4. Returns both filtered and unfiltered versions

This results in j-1 differential channels for j input channels.

### Best Practices

1. **Signal Preparation**:
   - Ensure input signals are properly scaled
   - Remove DC offset if present
   - Check for artifacts or bad channels before computing differentials

2. **Filter Parameter Selection**:
   - Use standard EMG frequency bands (20-450 Hz typical)
   - Match filter order to your signal quality needs
   - Consider edge effects in filtered signals

3. **Channel Ordering**:
   - Ensure channels are ordered correctly spatially
   - Verify channel alignment matches your electrode layout
   - Document channel arrangement for reproducibility

### Common Use Cases

- Reducing common-mode noise
- Enhancing local muscle activity detection
- Preprocessing for motor unit decomposition
- Improving spatial resolution of EMG recordings

## Grid Mapping

`hdsemg_shared.preprocessing.grid_map` turns a flat channel matrix into HDsEMG
grid columns. It only selects and orders — no filtering, no differentiation.

### The `emg_map` format

Follows openhdemg's `custom_sorting_order`: a list of lists (or a 2-D array)
where the **outer** index is the grid **column** and the **inner** index is the
**row**, holding base-0 channel numbers, with `np.nan` at positions that carry
no electrode or were excluded.

```python
emg_map = [[0, 1,  2,  3     ],   # col0
           [4, 5,  6,  7     ],   # col1
           [8, 9, 10, np.nan]]    # col2, one gap
```

### API Reference

```python
map_to_columns(emg_channels, emg_map)
```

- `emg_channels` — rows are channels, columns are samples. May hold more
  channels than the map refers to (force and path references, for example).
  `EMGFile.data` is samples-by-channels, so pass its transpose.
- returns a list of `nCols` matrices, each `nRows x nSamples`, in map order.
  A `NaN` map entry becomes an all-`NaN` row.

Raises `ValueError` on a ragged map, an entry that is not a valid channel
number, or a channel used more than once.

```python
emg_map_from_indices(emg_indices, rows, cols, order='column')
```

Builds a map from a `Grid`'s flat `emg_indices` plus `rows`/`cols`.
`order='column'` (default) means the channels run **down** the first column
before moving to the next — this matches MATLAB's `mat2grid`. `order='row'`
transposes that.

Grids with an unwired position — HD08MM1305 has 64 electrodes in 5x13 = 65
positions — cannot be described this way and raise. Pass an explicit map with
`NaN` at the empty position instead; openhdemg's `sort_rawemg` and
hdsemg-select's electrode display grid both produce one.

### Example Usage

```python
import numpy as np
from hdsemg_shared.fileio.file_io import EMGFile
from hdsemg_shared.preprocessing.grid_map import emg_map_from_indices, map_to_columns
from hdsemg_shared.preprocessing.differential import to_differential

emg_file = EMGFile.load("recording.otb+")
grid = emg_file.grids[0]

emg_map = emg_map_from_indices(grid.emg_indices, grid.rows, grid.cols)
columns = map_to_columns(emg_file.data.T, emg_map)

dmat, dmat_no_filter = to_differential(columns, emg_file.sampling_frequency,
                                       {"n": 2, "low": 20.0, "up": 450.0})
```

`hdsemg_shared.global_parameters.global_amplitude` takes the `emg_map` directly
and does the mapping and the MP/SD/DD differentiation itself.

## Source Code

> The implementations can be found in `src/hdsemg_shared/preprocessing/` — `differential.py` and `grid_map.py`. The code includes type hints and detailed docstrings for easy integration with development tools.

---

### API Documentation

::: hdsemg_shared.preprocessing.differential
    handler: python
    options:
      heading_level: 3
      show_root_heading: false

::: hdsemg_shared.preprocessing.grid_map
    handler: python
    options:
      heading_level: 3
