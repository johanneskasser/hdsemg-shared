# Filters

The `filters` package provides digital signal processing filters specifically designed for HD-sEMG signal processing. All filters are implemented with a focus on phase preservation and numerical stability.

## Bandpass Filter

The package provides a zero-phase Butterworth bandpass filter implementation that is particularly suited for EMG signal processing. The filter is implemented using SciPy's `butter` and `sosfiltfilt` functions for optimal numerical stability.

### API Reference

```python
def bandpass_filter(data: np.ndarray, order: int, lowcut: float, highcut: float, fs: float) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to 1D data."""
```

#### Parameters

- **data** (`np.ndarray`): 
    - 1D array of signal samples to be filtered
    - Shape: (n_samples,)
    - Type: float

- **order** (`int`): 
    - Filter order
    - Higher orders give sharper cutoffs but may introduce more numerical artifacts
    - Typical values: 2-8

- **lowcut** (`float`): 
    - Lower cutoff frequency in Hz
    - Typical EMG values: 10-30 Hz

- **highcut** (`float`): 
    - Upper cutoff frequency in Hz
    - Typical EMG values: 400-500 Hz

- **fs** (`float`): 
    - Sampling frequency in Hz
    - Must be at least 2x the highest frequency component (Nyquist criterion)

#### Returns

- **filtered_data** (`np.ndarray`):
    - Filtered signal
    - Same shape as input data
    - Zero-phase (no temporal shifting)

### Implementation Details

The filter uses:
- Butterworth filter design for maximally flat frequency response
- Second-order sections (SOS) form for improved numerical stability
- Zero-phase filtering via forward-backward application (sosfiltfilt)
- Automatic scaling of frequencies to Nyquist frequency

### Example Usage

```python
import numpy as np
from hdsemg_shared.filters.bandpass import bandpass_filter

# Generate sample EMG data
fs = 2000  # Sample rate: 2kHz
t = np.linspace(0, 1, fs)  # 1 second of data
emg = np.random.randn(fs)  # Simulated noise-like EMG

# Apply bandpass filter (20-450 Hz, 4th order)
filtered_emg = bandpass_filter(
    data=emg,
    order=4,
    lowcut=20,    # Remove low-frequency drift
    highcut=450,  # Remove high-frequency noise
    fs=fs
)
```

### Best Practices

1. **Filter Order Selection**:
   - Start with order=4 for most applications
   - Increase order if you need sharper cutoffs
   - Decrease order if you observe ringing artifacts

2. **Frequency Selection**:
   - For surface EMG: lowcut=20Hz, highcut=450Hz is typical
   - Adjust based on your specific application and noise conditions
   - Ensure highcut < fs/2 (Nyquist frequency)

3. **Edge Effects**:
   - The filter may introduce edge effects at the start/end of the signal
   - Consider padding your signal or discarding edge regions in critical analyses

### Common Use Cases

- Removing power line interference and baseline drift
- Isolating the main EMG frequency band (20-450 Hz)
- Pre-processing before amplitude analysis or feature extraction

## Future Extensions

The filters package is designed to be extensible. Planned or possible additions include:

- Notch filters for power line interference
- Adaptive filters for specific noise types
- Wavelet-based filtering approaches
- Multi-channel filtering utilities

## Source Code

The implementation can be found in `src/hdsemg_shared/filters/bandpass.py`. The code is well-documented and follows numpy docstring conventions for easy integration with IDEs and documentation tools.
