# File I/O

The `fileio` module in `hdsemg-shared` provides unified, high-level functions for loading and saving high-density surface EMG (HD-sEMG) data from various file formats. It abstracts away the details of different vendor formats and provides a consistent interface for downstream processing.

---

## Supported File Types

- **MATLAB files** (`.mat`):
  - Loads data, time, description, and sampling frequency from MATLAB files.
- **OTB+ / OTB files** (`.otb+`, `.otb`):
  - Handles OTB+ archives, including extraction, XML parsing, and signal scaling.
- **OTB4 files** (`.otb4`):
  - Supports OTB4 archives, including multi-track and device-specific handling.

---

## Main API: `load_file`

```python
from hdsemg_shared.fileio import load_file

data, time, description, sampling_frequency, file_name, file_size = load_file(filepath)
```

- **`filepath`**: Path to the file to load (supports `.mat`, `.otb+`, `.otb`, `.otb4`).
- **Returns**: Tuple with data, time, description, sampling frequency, file name, and file size.

The function automatically detects the file type and dispatches to the appropriate loader.

---

## Data Structure

- **data**: `np.ndarray` (nSamples x nChannels), always returned as float for further processing.
- **time**: `np.ndarray` (nSamples,), time vector.
- **description**: Channel descriptions (array or list).
- **sampling_frequency**: Sampling frequency in Hz.
- **file_name**: Name of the loaded file.
- **file_size**: File size in bytes.

---

## Example Usage

```python
from hdsemg_shared.fileio import load_file

# Load any supported file
file_path = "my_emg_data.otb+"
data, time, description, fs, fname, fsize = load_file(file_path)

print(f"Loaded {fname} with shape {data.shape} and fs={fs} Hz")
```

---

## Internal Structure

- The `fileio` module delegates to specialized loaders:
  - `matlab_file_io.py` for `.mat` files
  - `otb_plus_file_io.py` for `.otb+` and `.otb` files
  - `otb_4_file_io.py` for `.otb4` files
- All loaders ensure data is returned in a consistent format and shape.
- Data is always converted to float for compatibility with downstream processing.

---

## Error Handling

- Raises `ValueError` for unsupported file types.
- Raises errors if file content is incompatible or missing required fields.

---

## Saving Data

Saving to `.mat` files is supported via:

```python
from hdsemg_shared.fileio.matlab_file_io import save_selection_to_mat

save_selection_to_mat(save_file_path, data, time, description, sampling_frequency, file_name, grid_info)
```

---

> For more details, see the API documentation or the source code in `src/hdsemg_shared/fileio/`.

---

## API Documentation

::: hdsemg_shared.fileio
    handler: python
    options:
      heading_level: 3


