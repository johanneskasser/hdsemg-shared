# File I/O & Grid Extraction

The **`hdsemg_shared.fileio`** module provides a interface to:

* Load HD-sEMG data from MATLAB (`.mat`), OTB+ (`.otb+`, `.otb`) or OTB4 (`.otb4`) files
* Support for **Novecento+** device files with multi-track recordings and control signals
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
emg.file_type          # "mat" | "otb" | "otb4" | "edf"
emg.channel_count      # int, number of channels (= data.shape[1])
emg.unit               # "V" | "mV" | "uV" | "a.u." | None – unit of the EMG channels
```

#### Signal Unit

`emg.unit` says what the **EMG grid channels** are measured in, taken from what
the file declares — never guessed from the magnitude of the data. It is `None`
when the format declares nothing, which is deliberate: an unknown unit is
recoverable, a wrong default is not.

Reference and auxiliary channels (force, requested/performed path, AUX) are
**not** in this unit — they carry their own scales.

| Format | Source of the unit | Typical result |
|---|---|---|
| `.otb4` | `<UnitOfMeasurement>` of each EMG grid track | `"mV"` |
| `.otb+` / `.otb` | Fixed by the loader's scaling (ADC counts → volts × 1000) | `"mV"` |
| `.mat` | Optional `Unit` variable, written by `EMGFile.save` | `None` for plain MATLAB exports |
| `.edf` | Physical dimension in the EDF header | `"uV"`, or `None` when blank |

When the EMG tracks/signals of one file disagree on a unit, `unit` is `None`
and a warning is logged.

```python
emg.scale_to("uV")     # factor, e.g. 1000.0; raises ValueError when unit is None
uv = emg.to_unit("uV") # a converted copy – EMG channels only, refs untouched
```

`to_unit` never modifies the original. `"a.u."` is never convertible.

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
  Loads a `.mat` and returns
  `(data, time, description, sampling_frequency, file_name, file_size, unit)`.

* **`MatFileIO.save(save_path: str, data, time, description, sampling_frequency, unit=None)`**
  Saves the provided arrays/metadata to a `.mat` file. `unit` is written only
  when it is not `None`, so a round-trip preserves it without inventing one.

!!! note "Loader return tuple"
    All four loaders now return the declared unit as a seventh element.
    `EMGFile.load` also accepts the older six-element form, in which case
    `emg.unit` is `None`.

---

## Novecento+ Support

The library fully supports **Novecento+** (OTBiolab) files in `.otb4` format. These files have unique characteristics:

### Multi-Track Signal Files

Novecento+ recordings store multiple data tracks in the same signal file (`.sig`), distinguished by:

* **`ChannelOffsetInSubPacket`**: Starting position of each track's channels within the file
* **`IsControl`**: Flag indicating control/reference signals (quaternions, buffer, ramp, etc.)
* **Grid metadata** in XML `<Description>` elements (`<Name>`, `<NRow>`, `<NColumn>`, `<IED>`)

### Grid Extraction

The loader automatically:

1. Parses grid information from `Tracks_000.xml` file
2. Extracts **EMG channels** (channels with valid grid patterns like `HD08MM1305`)
3. Identifies **reference channels** (control signals, quaternions, buffer, ramp)
4. Assigns reference channels to the appropriate EMG grid based on their position in the file

### Channel Organization

For Novecento+ files, channels are organized as:

* **EMG channels** (`emg_indices`): Main electrode grid signals with `HD{IED}MM{rows}{cols}` pattern
* **Reference channels** (`ref_indices`): Control signals, quaternions, buffer, ramp, and auxiliary inputs marked with `REF` in descriptions

### Example: Novecento+ File

```python
# Load Novecento+ OTB4 file
emg = EMGFile.load("recording.otb4")

# Main EMG grid (e.g., 5x13 = 65 electrodes, 64 active)
main_grid = emg.grids[0]
print(f"EMG Grid: {main_grid.rows}x{main_grid.cols}, IED={main_grid.ied_mm}mm")
print(f"EMG channels: {len(main_grid.emg_indices)}")
print(f"Reference channels: {len(main_grid.ref_indices)}")

# Access EMG data only
emg_data = emg.data[:, main_grid.emg_indices]

# Access reference signals (quaternions, control signals, etc.)
ref_data = emg.data[:, main_grid.ref_indices]

# Check which channels are references
for idx in main_grid.ref_indices:
    desc = emg.description[idx]
    print(f"Ref channel {idx}: {desc}")
```

### Supported Signal Types

Novecento+ files may contain:

* **HD-sEMG grids**: Main EMG electrode arrays (e.g., HD08MM1305, HD08MM0513)
* **Quaternions**: IMU orientation data (4 channels, 2x2 layout)
* **Buffer/Ramp**: Device control signals (1 channel each)
* **AUX inputs**: Auxiliary analog inputs (16 channels, 2kHz)
* **Control signals**: Additional device status channels (8 channels, 8kHz)
* **Load cells**: Force/torque sensors (1+ channels)

### Technical Details

* **Data format**: int32 (Novecento+) converted to float64 with appropriate scaling
* **Conversion**: `ADC_Range / (2^ADC_Nbits) * 1000 / Gain`
* **Fortran-order reshaping**: Data is reshaped as `(channels, samples, order='F')`
* **Mixed sampling rates**: Main EMG at 2kHz, control signals at 8kHz (automatically trimmed to match)

---

## Under the Hood

* **Format dispatch** in `EMGFile.load`:

  * MATLAB (`.mat`) → `MatFileIO.load`
  * OTB+ / OTB (`.otb+`, `.otb`) → `otb_plus_file_io.load_otb_file`
  * OTB4 (`.otb4`) → `otb_4_file_io.load_otb4_file`
    * **Novecento+** device detection and multi-track loading
    * Grid metadata extraction from XML
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
