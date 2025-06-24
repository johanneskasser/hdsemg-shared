# File I/O & Grid Extraction

The **`hdsemg_shared.fileio`** module provides a interface to:

* Load HD-sEMG data from MATLAB (`.mat`), OTB+ (`.otb+`, `.otb`) or OTB4 (`.otb4`) files
* Automatically sanitize and reshape the data/time arrays
* Extract electrode‐grid metadata (rows, columns, IED, reference channels, etc.)
* Cache remote grid‐configuration JSON for one week
* Save back to `.mat` if needed

---

## Core Types

### `EMGFile`

```python
from hdsemg_shared.fileio.file_io import EMGFile
```

A single class that bundles:

* Raw data & time vectors
* Channel descriptions
* Sampling frequency, file name, file size, file type
* Electrode‐grid metadata via the `.grids` property

#### Loading

```python
emg = EMGFile.load("session1.mat")
```

* **`load(filepath: str) -> EMGFile`**
  Detects the extension and dispatches to the appropriate loader
  (`.mat` → `MatFileIO.load`, `.otb+`/`.otb` → `otb_plus_file_io`,
  `.otb4` → `otb_4_file_io`), then sanitizes and returns an `EMGFile`.

#### Attributes

```python
emg.data               # np.ndarray, shape (nSamples × nChannels), float32
emg.time               # np.ndarray, shape (nSamples,)
emg.description        # list or array of channel‐description strings
emg.sampling_frequency # float
emg.file_name          # str
emg.file_size          # int (bytes)
emg.file_type          # "mat" | "otb" | "otb4"
emg.channel_count      # int, number of channels (= data.shape[1])
```

#### Grid Metadata

```python
from hdsemg_shared.fileio.file_io import Grid

grids: list[Grid] = emg.grids
```

* **`.grids`** (lazy‐loaded): a list of `Grid` objects (one per detected grid in the file).
* **`.get_grid(grid_key=…)`** or **`.get_grid(grid_uid=…)`**: retrieve a single `Grid` by its key (e.g. `"8x4"`) or UUID.

##### `Grid` dataclass

```python
@dataclass
class Grid:
    emg_indices: list[int]         # indices of EMG channels in data/time
    ref_indices: list[int]         # indices of reference channels
    rows: int                      # number of rows on the grid
    cols: int                      # number of columns on the grid
    ied_mm: int                    # inter‐electrode distance in millimeters
    electrodes: int                # total electrodes (rows × cols or remote lookup)
    grid_key: str                  # e.g. "8x4"
    grid_uid: str                  # unique UUID string
    requested_path_idx: int | None # index of “requested path” entry in description
    performed_path_idx: int | None # index of “performed path” entry in description
```

#### Saving

```python
emg.save("subset.mat")
```

* **`.save(save_path: str) -> None`**
  Currently only supports saving to `.mat` via `MatFileIO.save`.
  Raises `ValueError` for any other extension.

#### Utility

```python
emg.copy()
```

* **`.copy() -> EMGFile`**
  Returns a deep copy of the entire `EMGFile` (data, metadata, grids).

---

## Low-Level MATLAB I/O

```python
from hdsemg_shared.fileio.matlab_file_io import MatFileIO
```

* **`MatFileIO.load(file_path: str) -> tuple`**
  Loads a `.mat` and returns exactly
  `(data, time, description, sampling_frequency, file_name, file_size)`.

* **`MatFileIO.save(save_path: str, data, time, description, sampling_frequency)`**
  Saves the provided arrays/metadata to a `.mat` file.

---

## Under the Hood

* **Format dispatch** in `EMGFile.load`:

  * MATLAB (`.mat`) → `MatFileIO.load`
  * OTB+ / OTB (`.otb+`, `.otb`) → `otb_plus_file_io.load_otb_file`
  * OTB4 (`.otb4`) → `otb_4_file_io.load_otb4_file`
* **Sanitization**: ensures `data` is 2-D (samples × channels) and `time` is 1-D, swapping axes if needed.
* **Grid JSON cache**: fetched from Google Drive once per week, stored in `~/.hdsemg_cache/`.

---

## Quick Example

```python
# Load and inspect
emg = EMGFile.load("myrecording.otb+")
print(emg.data.shape, emg.sampling_frequency)

# List grids
for grid in emg.grids:
    print(f"{grid.grid_key}: {len(grid.emg_indices)} EMG, {len(grid.ref_indices)} refs")

# Find a specific grid
g2x8 = emg.get_grid(grid_key="2x8")

# Save a selection back to .mat
emg.save("selected_subset.mat")
```

---
## API Documentation
::: hdsemg_shared.fileio.file_io
    handler: python
    options:
      heading_level: 3
