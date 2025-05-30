# Grid Utilities

The `grid` module provides utilities for handling HD-sEMG electrode grid configurations and their associated data. It includes functionality for loading grid specifications, extracting grid information from data files, and managing grid data caching.

## Core Functions

### Grid Data Management

```python
def grid_json_setup():
    """Initialize the global grid_data variable by loading data from a JSON file."""
```

This function initializes the grid configuration data from a remote source and caches it locally. It's automatically called when needed.

### Grid Information Retrieval

```python
def get_electrodes_from_grid_name(grid_name: str) -> int:
    """Get the number of electrodes for a given grid name."""
```

#### Parameters
- **grid_name** (`str`): The name of the grid (e.g., "HD08MM0804")

#### Returns
- **int**: Number of electrodes if found, None otherwise

#### Example
```python
n_electrodes = get_electrodes_from_grid_name("HD08MM0804")
print(f"Number of electrodes: {n_electrodes}")
```

### Grid Information Extraction

```python
def extract_grid_info(description: list) -> dict:
    """Extract grid dimensions, indices, and reference signals from the description."""
```

#### Parameters
- **description** (`list`): List of descriptions containing grid information

#### Returns
- **dict**: Dictionary containing:
  - Grid dimensions (rows × columns)
  - Electrode indices
  - Inter-electrode distance (IED)
  - Number of electrodes
  - Reference signal indices
  - Path indices (requested and performed)

#### Example
```python
grid_info = extract_grid_info(description_list)
for grid_key, info in grid_info.items():
    print(f"Grid {grid_key}:")
    print(f"- Dimensions: {info['rows']}x{info['cols']}")
    print(f"- IED: {info['ied_mm']}mm")
    print(f"- Electrodes: {info['electrodes']}")
```

### File Loading

```python
def load_single_grid_file(file_path: str) -> list:
    """Load and process a single file to extract grid information."""
```

#### Parameters
- **file_path** (`str`): Path to the data file

#### Returns
- **list**: List of dictionaries, each containing:
  - File information (path, name)
  - Grid data and time vectors
  - Channel descriptions
  - Sampling frequency
  - Grid configuration (dimensions, IED)
  - Unique grid identifier

#### Example
```python
grids = load_single_grid_file("path/to/your/emg_data.mat")
for grid in grids:
    print(f"Grid {grid['grid_key']}:")
    print(f"- Shape: {grid['rows']}x{grid['cols']}")
    print(f"- Sampling rate: {grid['sf']} Hz")
    print(f"- Data shape: {grid['data'].shape}")
```

## Implementation Details

### Grid Data Caching
- Grid configurations are cached locally (~/.hdsemg_cache/grid_data_cache.json)
- Cache expires after one week
- Automatic fallback to remote data if cache is invalid or expired

### Grid Name Pattern
- Grid names follow the pattern: `HDxxMMxxxx`
  - First `xx`: Inter-electrode distance in mm
  - Second `xx`: Number of rows
  - Third `xx`: Number of columns

### Error Handling
- Robust error handling for network issues
- Fallback mechanisms for cache failures
- Type checking for input parameters

## Best Practices

1. **Grid Configuration**:
   - Always verify grid dimensions match your electrode setup
   - Check inter-electrode distances for spatial analysis
   - Validate channel counts against expected values

2. **Data Loading**:
   - Use load_single_grid_file for consistent data structure
   - Verify sampling frequencies match your recording setup
   - Check reference signals are correctly identified

3. **Cache Management**:
   - Let the system manage the cache automatically
   - Cache will refresh weekly to ensure up-to-date grid configurations
   - Manual cache clearing possible by deleting ~/.hdsemg_cache

## Common Use Cases

- Loading multi-grid recordings
- Extracting spatial information from EMG data
- Managing reference signals and path information
- Consistent handling of different grid configurations

## Source Code

The implementation can be found in `src/hdsemg_shared/grid.py`. The code includes type hints and extensive error handling for robust operation.
