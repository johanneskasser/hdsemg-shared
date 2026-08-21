# Global EMG Parameter Documentation

This section documents the computation of global HD-sEMG parameters, their theoretical background, scientific relevance, and Python implementation.

---

## Amplitude-Based Features

### 1. Root Mean Square (RMS)

**Description:**

* Quantifies the signal’s energy content.
* Sensitive to muscle activation level.

**Formula:**
$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$

**Python Implementation:**

```python
import numpy as np

def root_mean_square(signal: np.ndarray) -> float:
    if signal.ndim > 1:
        raise ValueError("Signal must be a 1D array.")
    return np.sqrt(np.mean(signal ** 2))
```

**References:**

* Merletti & Farina (2016), *Surface EMG: Physiology, Engineering and Applications*
* Clancy et al. (2023), CEDE recommendations

---

### 2. Average Rectified Value (ARV)

**Description:**

* Linear estimator of the envelope.
* Mean absolute value of the signal.

**Formula:**
$\text{ARV} = \frac{1}{N} \sum_{i=1}^{N} |x_i|$

**Python Implementation:**

```python
import numpy as np

def average_rectified_value(signal: np.ndarray) -> float:
    if signal.ndim > 1:
        raise ValueError("Signal must be a 1D array.")
    return np.mean(np.abs(signal))
```

**References:**

* CEDE Amplitude Normalization Matrix (Dideriksen et al., 2023)
* Clancy et al. (2023)

---

### 3. Integrated EMG (IEMG)

**Description:**

* Total electrical activity over a time window.

**Formula:**
$\text{IEMG} = \sum_{i=1}^{N} |x_i|$

**Python Implementation:**

```python
import numpy as np

def integrated_emg(signal: np.ndarray) -> float:
    if signal.ndim > 1:
        raise ValueError("Signal must be a 1D array.")
    return np.sum(np.abs(signal))
```

**References:**

* Merletti & Farina (2016)
* CEDE Amplitude Matrix

---

## Frequency-Based Features

### 4. Median Frequency (MDF)

**Description:**

* Frequency dividing the power spectrum into equal halves.
* Related to muscle fatigue.

**Python Implementation:**

```python
import numpy as np
from scipy.signal import welch

def compute_mdf(signal: np.ndarray, fs: float) -> float:
    f, pxx = welch(signal, fs=fs)
    cumsum = np.cumsum(pxx)
    mdf_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    return f[mdf_idx]
```

**References:**

* Farina & Merletti (2003), Biomedical Engineering
* CEDE Force Estimation Matrix

---

### 5. Mean Frequency (MNF)

**Description:**

* Center of gravity of the EMG power spectrum.
* Sensitive to fatigue and conduction velocity.

**Python Implementation:**

```python
import numpy as np
from scipy.signal import welch

def compute_mnf(signal: np.ndarray, fs: float) -> float:
    f, pxx = welch(signal, fs=fs)
    return np.sum(f * pxx) / np.sum(pxx)
```

**References:**

* Phinyomark et al. (2012)
* CEDE (Farina et al., 2023)

---

## Complexity-Based Feature

### 6. Permutation Entropy

**Description:**

* Nonlinear metric for signal complexity.
* Used in motor control and fatigue studies.

**Python Implementation:**

```python
from antropy import perm_entropy
import numpy as np

def compute_entropy(signal: np.ndarray) -> float:
    return perm_entropy(signal, normalize=True)
```

**References:**

* Bandt & Pompe (2002)
* CEDE Motor Unit Matrix (Martinez-Valdes et al., 2023)

---

## Grid-Wide Amplitude

### 7. Global Amplitude

**Description:**

* Reduces a whole HDsEMG grid to one amplitude over time, `[xV]` — the same
  volt prefix as the input. `EMGFile.unit` names it; convert with
  `emg.to_unit("uV")` before reporting an absolute number.
* Takes the raw channel matrix plus an `emg_map`; band-pass filtering,
  MP/SD/DD differentiation and the reduction all happen inside.
* `method='RMS'` (default) or `'ARV'`.

**Definition, in five steps:**

1. band-pass each channel — 15–450 Hz, exact corners, zero-lag
2. square each channel — `x_i(t)^2` (ARV: `|x_i(t)|`)
3. smooth each channel in time — 15 Hz equivalent, zero-lag
4. mean across the channels — the mean square over the grid area
5. square root, **last** — the global amplitude `A(t)`

The root comes last because Merletti & Cerone (eq. 5.1/5.2) define a region's
RMS as the root of the mean of `f^2` over **both** space and time, taken at
the very end, which makes `mean_space(RMS^2) == mean_time(RMS^2)`. Rooting per
channel first breaks that identity and makes the result depend on how many
channels survived, by `E[chi_n]/sqrt(n)` — 0.940 at 4 channels, 0.979 at 12,
0.995 at 48. That matters directly: an MVC trial and a tracking trial rarely
keep the same channels, and their ratio is the reported %MVC.

This is the one difference from the MATLAB original `globEMGAmpEnv`, which
reduces across the channels at every sample and smooths afterwards. Every
other MATLAB setting is reachable through the arguments.

**The `emg_map`:**

Follows openhdemg's `custom_sorting_order` — outer index is the grid
**column**, inner index the **row**, entries are base-0 channel numbers into
`emg_channels`, `np.nan` where no electrode sits or a channel is excluded.

```python
emg_map = [[0, 1,  2,  3     ],   # col0
           [4, 5,  6,  7     ],   # col1
           [8, 9, 10, np.nan]]    # col2, one gap
```

`hdsemg_shared.preprocessing.grid_map.emg_map_from_indices` builds one from a
`Grid`'s flat `emg_indices` plus `rows`/`cols`.

**Example Usage:**

```python
import numpy as np
from hdsemg_shared.fileio.file_io import EMGFile
from hdsemg_shared.global_parameters import global_amplitude
from hdsemg_shared.preprocessing.grid_map import emg_map_from_indices

emg_file = EMGFile.load("recording.otb+")
emg_file = emg_file.to_unit("uV")      # emg_file.unit is "mV" for OTB+/OTB4
grid = emg_file.grids[0]
emg_map = emg_map_from_indices(grid.emg_indices, grid.rows, grid.cols)

# EMGFile.data is (n_samples, n_channels), the chain wants channels first
out = global_amplitude(emg_file.data.T, emg_map, emg_file.sampling_frequency,
                       method='RMS', derivation='SD')

out.amplitude      # global amplitude over time      [uV, matching the input]
out.per_channel    # each channel's own envelope     [uV, matching the input]
out.positions      # (column, row) of each channel
out.grid_shape     # (nCols, nRows) after derivation
out.n_channels     # how many contributed
```

Excluding channels is the caller's job: intersect the selection with the
grid's own channels, then put `np.nan` in the map at every excluded position.
Channels whose signal carries `NaN` are dropped too, matching the MATLAB
`*INFO* NaN channels are ignored`.

**Options:**

| argument | default | meaning |
|---|---|---|
| `method` | `'RMS'` | `'RMS'` or `'ARV'` |
| `derivation` | `'MP'` | `'MP'`, `'SD'` or `'DD'` |
| `diff_direction` | `'cols'` | which grid axis SD/DD difference along — `'cols'` walks **down** a map column, `'rows'` walks **across** a row. Ignored for `'MP'` |
| `bpf` | `{'N': 2, 'fcl': 15.0, 'fch': 450.0, 'corners': 'exact'}` | band-pass; `corners='prewarp'` for MATLAB-identical filtering |
| `smooth` | `{'mode': 'moving', 'fc': 15.0, 'kernel': 'bidirectional'}` | `'moving'` boxcar or `'lowpass'` butterworth; same `fc` gives the same bandwidth |
| `pad_s` | `0.25` | reflection pad at each end, guards the filter transient |

The MATLAB settings are `bpf={'N': 2, 'fcl': 30.0, 'fch': 450.0,
'corners': 'prewarp'}`, `smooth={'mode': 'lowpass', 'fc': 6.0, 'N': 2}` and
`pad_s=0`.

### Choosing `diff_direction`

Point the differencing **along the muscle fibres** — that is what makes an SD
or DD signal a propagating-potential measure rather than an arbitrary spatial
filter.

`'cols'` (the default) differences down a map column, i.e. between vertically
adjacent electrodes. It is right when the map's columns run along the fibres,
which is how MATLAB's `mat2grid` and openhdemg's `sort_rawemg` lay a grid out.
Use `'rows'` for a grid rotated 90° against that.

On a 3-column × 4-row map built column-first, neighbours within a column are
one channel apart and neighbours across a row are four apart:

```python
emg_map = emg_map_from_indices(range(12), rows=4, cols=3)
# [[ 0,  1,  2,  3],      diff_direction='cols'  ->  (3, 3) grid, 9 channels
#  [ 4,  5,  6,  7],      diff_direction='rows'  ->  (2, 4) grid, 8 channels
#  [ 8,  9, 10, 11]]
```

Transposing the map turns one direction into the other exactly, so the two
spellings are equivalent in arithmetic — but prefer `diff_direction`. It
leaves the map in the layout openhdemg produced, so `positions` and
`grid_shape` keep referring to the physical grid; a transposed map quietly
swaps rows for columns in everything downstream.

Orientation is a separate matter from `diff_direction`: openhdemg's built-in
codes only ever flip 0° / 180°, which maps a column onto a column and so never
changes which physical axis `'cols'` runs along. See *Orientation, the corner,
and `diff_direction`* below.

**References:**

* Merletti R, Cerone GL. *Techniques for information extraction from the surface EMG signal*, eq. 5.1/5.2. In: Merletti & Farina (2016/2018).
* Del Vecchio A et al. (2025). *J Appl Physiol*, doi:10.1152/japplphysiol.00810.2024
* CEDE Best Practice: Amplitude Matrix

---

## Using it from other tools

### From openhdemg

**Let openhdemg place the electrodes.** `electrodes.py` stores one
`base0_sorting_order` grid literal *per orientation*, together with a
`base0_nanpos` marking the empty channel — the rotation is data openhdemg
already owns. Ask `sort_rawemg` for the orientation you want and use what it
returns; never rotate or transpose a map yourself, because that moves the
physical corner and silently re-labels the electrodes.

Get the map for a built-in matrix code by sorting the channel *numbers*
instead of the signal:

```python
import numpy as np
import pandas as pd
import openhdemg.library as emg
from hdsemg_shared.global_parameters import global_amplitude

def emg_map_from_code(n_channels, code, orientation):
    """The channel numbers of an OTB matrix, as an emg_map, at one orientation."""
    probe = {"RAW_SIGNAL": pd.DataFrame(np.arange(n_channels, dtype=float).reshape(1, -1))}
    sorted_probe = emg.sort_rawemg(probe, code=code, orientation=orientation,
                                   dividebycolumn=True)
    if isinstance(sorted_probe, Exception):
        raise ValueError(sorted_probe)   # sort_rawemg RETURNS its errors
    return np.array([sorted_probe["col{}".format(j)].iloc[0, :].to_numpy()
                     for j in range(len(sorted_probe))])

emgfile = emg.askopenfile(filesource="OTB", otb_ext_factor=8)
emg_map = emg_map_from_code(64, "GR08MM1305", orientation=0)

# RAW_SIGNAL is samples-by-channels; the chain wants channels first
out = global_amplitude(emgfile["RAW_SIGNAL"].to_numpy().T, emg_map,
                       emgfile["FSAMP"], method='RMS', derivation='SD')
```

Pass `orientation` explicitly. There is no safe default: it decides which
index the physical corner electrode carries.

`sort_rawemg` **returns** a `ValueError` rather than raising it for an
unsupported code/orientation, hence the explicit check.

If you have a grid openhdemg does not know, its `custom_sorting_order` is an
`emg_map` unchanged — same format, outer index = column, inner = row:

```python
custom_sorting_order = [
    [63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52,     51,],
    [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,     50,],
    [37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26,     25,],
    [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,     24,],
    [11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0, np.nan,],
]  # this particular one IS openhdemg's GR08MM1305 table at orientation 0

out = global_amplitude(emg_channels, custom_sorting_order, fs, derivation='SD')
```

Note that `hdsemg_shared` does **not** import openhdemg — it only follows the
same map format, so nothing here adds a dependency.

#### Orientation, the corner, and `diff_direction`

GR08MM1305 has 64 electrodes in 5 × 13 = 65 slots, so one slot is empty, and
that slot is a **corner** of the physical array. openhdemg's `orientation=0`
and `orientation=180` are the same physical grid turned 180°
(`np.rot90(map_180, 2) == map_0`), so the corner changes *index* — `(4, 12)`
at 0, `(0, 0)` at 180 — while remaining the same physical electrode.

A 180° turn maps a column onto a column, so it cannot change which physical
axis a derivation runs along. The derived channel counts are therefore
identical at both orientations:

| derivation | `diff_direction` | derived grid | channels | why |
|---|---|---|---|---|
| SD | `'cols'` | (5, 12) | **59** | 4 columns × 12, plus the gapped column's 11 |
| DD | `'cols'` | (5, 11) | **54** | 4 columns × 11, plus 10 |
| SD | `'rows'` | (4, 13) | **51** | 12 rows × 4, plus the gapped row's 3 |
| DD | `'rows'` | (3, 13) | **38** | 12 rows × 3, plus 2 |

Because the empty slot is a corner it always sits at the *end* of both its
column and its row, so it breaks exactly one adjacent pair in each direction.
A gap in the *middle* of a column would break two — that is the general rule,
not the 59.

openhdemg only ever offers 0° and 180° for its built-in codes, never 90°, so
for those `diff_direction='cols'` always runs along the same physical axis.
Use `'rows'` when the grid was physically mounted turned 90° against the
fibres, which no orientation flag can express.

### From hdsemg-pipe

pipe holds an `EMGFile` and its `Grid` objects, and nothing else about the
geometry, so build the map from the flat `emg_indices`:

```python
import json
import numpy as np
from hdsemg_shared.fileio.file_io import EMGFile
from hdsemg_shared.global_parameters import global_amplitude
from hdsemg_shared.preprocessing.grid_map import emg_map_from_indices

emg_file = EMGFile.load(cropped_file_path)

# The *_uv keys below claim microvolts, so earn the claim rather than assuming
# it: OTB+/OTB4 hand back millivolts, and a plain .mat declares nothing at all.
if emg_file.unit is None:
    raise ValueError(
        f"{cropped_file_path} declares no signal unit; refusing to write "
        f"microvolt fields from data of unknown scale."
    )
emg_file = emg_file.to_unit("uV")

emg_channels = emg_file.data.T          # (n_samples, n_channels) -> channels first

results = {}
for grid in emg_file.grids:
    emg_map = emg_map_from_indices(grid.emg_indices, grid.rows, grid.cols)

    out = global_amplitude(emg_channels, emg_map, emg_file.sampling_frequency,
                           method='RMS', derivation='MP')

    results[grid.grid_key] = {
        "muscle": grid.muscle,
        "ied_mm": grid.ied_mm,
        "n_channels": out.n_channels,
        "mean_uv": float(out.amplitude.mean()),
        "max_uv": float(out.amplitude.max()),
        "per_channel_uv": {
            "{}_{}".format(col, row): float(envelope.mean())
            for (col, row), envelope in zip(out.positions, out.per_channel)
        },
    }

with open(base_name + "_globalamp.json", "w") as fh:
    json.dump({"grids": results}, fh, indent=2)
```

This mirrors the existing `{base_name}_rms.json` sidecar contract, so a
consumer can pick it up the same way `RMSLoader` picks up the RMS one.

A grid with an unwired position — HD08MM1305 has 64 electrodes in 5×13 = 65
slots — cannot be described by `emg_map_from_indices` and raises. Pass an
explicit map with `np.nan` at the empty slot for those.

### From hdsemg-select

select holds three things the map has to be built from, and two of them are
easy to get wrong:

- `_electrode_display_grid` — a `(rows, cols)` float array whose entries are
  **local, 0-based indices into that grid's `emg_indices`**, not global data
  columns, with `np.nan` at absent positions.
- `channel_status` — a flat `list[bool]` indexed by **global** data column,
  whose `True` entries include the reference channels.
- the crop, via `get_effective_emg_data()`.

So a caller has to resolve local → global through `emg_indices`, and intersect
the mask with the grid's own channels. Excluded positions become `np.nan`:

```python
import numpy as np
from hdsemg_shared.global_parameters import global_amplitude

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

handler = ...                                       # GridSetupHandler for the active grid
emg_map = emg_map_from_selection(handler.get_electrode_display_grid(),
                                 grid.emg_indices,
                                 global_state.get_channel_status())

out = global_amplitude(global_state.get_effective_emg_data().T, emg_map,
                       emg_file.sampling_frequency, method='ARV')
```

For a density map, scatter `per_channel` back onto the grid with `positions` —
this replaces the ad-hoc `channels_to_grid` step, and it stays correct for
`'SD'`/`'DD'`, where a row of `per_channel` is a difference of two electrodes
and has no single channel number:

```python
n_cols, n_rows = out.grid_shape
frame = np.full((n_rows, n_cols), np.nan)
for (col, row), envelope in zip(out.positions, out.per_channel):
    frame[row, col] = envelope[sample_index]        # (rows, cols) for plotting
```

Excluding a channel changes how many contribute, and `amplitude` is
independent of that count by construction — so toggling a channel on or off in
the review tool does not shift the level, it only changes what is being
averaged over.

---

## API Documentation

::: hdsemg_shared.global_parameters.global_amplitude
    handler: python
    options:
      heading_level: 3
