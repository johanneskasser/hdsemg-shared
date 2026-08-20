# Filters

The `filters` package provides digital signal processing filters specifically designed for HD-sEMG signal processing. All filters are implemented with a focus on phase preservation and numerical stability.

## Bandpass Filter

The package provides a zero-phase Butterworth bandpass filter implementation that is particularly suited for EMG signal processing. It is an exact replica of Ton van den Bogert's MATLAB `bandpassfilter` ([reference](https://biomch-l.isbweb.org/archive/index.php/t-26625.html)): the requested order is halved, the halved order is used in the pre-warping constant, and the filter is then applied forwards and backwards so the effective order is the requested one again.

### API Reference

```python
def bandpass_filter(data: np.ndarray, N: int, fcl: float, fch: float, fs: float) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""
```

#### Parameters

- **data** (`np.ndarray`): 
    - Signal samples to be filtered
    - Shape: (n_samples,), or (n_channels, n_samples) — filtering runs along the last axis, so each channel row is filtered independently
    - Type: float

- **N** (`int`): 
    - *Total* filter order, must be **even** and >= 2
    - Halved internally, then doubled again by the forward/backward application
    - Typical values: 2-8

- **fcl** (`float`): 
    - Lower cutoff frequency in Hz
    - Typical EMG values: 10-30 Hz

- **fch** (`float`): 
    - Upper cutoff frequency in Hz
    - Typical EMG values: 400-500 Hz

- **fs** (`float`): 
    - Sampling frequency in Hz
    - Because of the pre-warping, the Nyquist criterion alone is **not** sufficient: `fs` must satisfy `fs > 2*fch/beta` with `beta = (sqrt(2)-1)**(1/N)`. For `N=2` that is `fs > 3.11*fch`, not `fs > 2*fch`

#### Returns

- **filtered_data** (`np.ndarray`):
    - Filtered signal
    - Same shape as input data
    - Zero-phase (no temporal shifting)

#### Raises

- **ValueError**:
    - If `N` is not an even integer >= 2
    - If the cutoffs do not satisfy `0 < fcl < fch`
    - If `fs` is too small for the requested cutoffs (see above). MATLAB errors on exactly the same inputs; before 2026-08 this function clipped the normalised cutoffs instead and returned a finite but **wrong** result

### Implementation Details

The filter uses:
- Butterworth filter design for maximally flat frequency response
- Transfer-function (`b`, `a`) form with `filtfilt`, using MATLAB's default padding (`padtype='odd'`, `padlen=3*(max(len(a),len(b))-1)`) rather than SciPy's defaults, which measurably diverge from MATLAB on short signals
- Zero-phase filtering via forward-backward application (`filtfilt`)
- Pre-warped cutoff scaling so the *double-pass* response has its -3 dB points at the requested frequencies

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
    N=4,        # total order, must be even
    fcl=20,     # Remove low-frequency drift
    fch=450,    # Remove high-frequency noise
    fs=fs
)
```

### Best Practices

1. **Filter Order Selection**:
   - Start with N=4 for most applications
   - N must be even (it is halved internally for the bi-directional pass)
   - Increase N if you need sharper cutoffs
   - Decrease N if you observe ringing artifacts

2. **Frequency Selection**:
   - For surface EMG: fcl=20Hz, fch=450Hz is typical
   - Adjust based on your specific application and noise conditions
   - Ensure `fs > 2*fch/(sqrt(2)-1)**(1/N)`, which is stricter than the plain Nyquist criterion `fch < fs/2` — the function raises otherwise

3. **Edge Effects**:
   - The filter may introduce edge effects at the start/end of the signal
   - Consider padding your signal or discarding edge regions in critical analyses

### Common Use Cases

- Removing power line interference and baseline drift
- Isolating the main EMG frequency band (20-450 Hz)
- Pre-processing before amplitude analysis or feature extraction

## Bandpass Filter with Exact Corners

`bandpass_filter_exact_corners(data, N, fcl, fch, fs)` band-passes each channel
with its realised -3 dB points landing **exactly** on `fcl` and `fch`.

### Why both band-passes exist

`bandpass_filter` is an exact replica of Ton van den Bogert's MATLAB original,
whose `beta` pre-warp is derived for a **single** corner. A band-pass
Butterworth has no such closed form, so dividing **both** corners by
`beta < 1` pushes both up by `1/beta = 1.554`. Measured at `fs = 2048 Hz`,
`N = 2`:

| requested | realised by `bandpass_filter` | realised by `bandpass_filter_exact_corners` |
|---|---|---|
| 15–450 Hz | 35.3–574.9 Hz | 15.0–450.0 Hz |
| 30–450 Hz | 68.8–582.3 Hz | 30.0–450.0 Hz |

`bandpass_filter` is deliberately **not** changed — every current caller
depends on those numbers, and it is what reproduces MATLAB output. Use the
exact-corner variant when the documented cut-offs must be the real ones.

```python
from hdsemg_shared.filters.bandpass import bandpass_filter_exact_corners

filtered = bandpass_filter_exact_corners(emg, N=2, fcl=15.0, fch=450.0, fs=2048.0)
```

The solved upper corner sits *above* `fch` for a two-pass filter, so keep
`fch` well below `fs/2`.

## Lowpass Filter

`lowpass_filter(data, N, fc, fs)` — the same pre-warp and the same
bi-directional application as `bandpass_filter`, for a single corner. That is
the case the pre-warp *is* exact for, so the realised -3 dB point lands on
`fc` and no exact-corner variant is needed (verified: 15.000 Hz for `N = 2`
and `N = 4`).

```python
from hdsemg_shared.filters.lowpass import lowpass_filter

envelope = lowpass_filter(np.abs(filtered), N=2, fc=15.0, fs=2048.0)
```

## Smoothing

`moving_average(data, fs, window_s=None, fc=None, kernel='bidirectional')` is
the second way to smooth an envelope: a zero-phase boxcar instead of a
Butterworth. Given the same `fc` the two attenuate a sine at `fc` identically,
so the choice changes kernel shape, never bandwidth.

- `'bidirectional'` (default) — two boxcar passes, i.e. a triangular kernel.
- `'rectangular'` — one plain boxcar. With a window as long as the signal
  every sample returns the plain unweighted mean over the whole signal, which
  the triangular kernel does not. Use it wherever exact agreement with an
  epoch mean is the point.

`window_length_for_cutoff(fc, fs, kernel)` returns the window a given `fc`
implies — roughly `0.319 * fs / fc` for the default kernel, so 44 samples at
`fs = 2048 Hz`, `fc = 15 Hz`.

## Padding

Filtering a finite signal disturbs both ends, and `filtfilt` pads by only 3
samples. `reflect_pad(data, pad)` / `trim_pad(data, pad)` extend each channel
with an **even** reflection of its own leading and trailing samples and remove
it again afterwards. `pad_samples(n_samples, fs, pad_s)` gives the sample count
both agree on.

Measured over 300 white-noise realisations, bias of the leading 64 samples:

| padding | band-pass | lowpass |
|---|---|---|
| none | +8.54 % | -0.66 % |
| odd reflection (what `filtfilt` does) | +9.16 % | -0.84 % |
| even reflection (here) | -0.28 % | -0.01 % |

Odd reflection does not fix the band-pass at all, so this is not a redundant
second pad. `global_amplitude` applies it through its `pad_s` argument.

## Future Extensions

The filters package is designed to be extensible. Planned or possible additions include:

- Notch filters for power line interference
- Adaptive filters for specific noise types
- Wavelet-based filtering approaches

## Source Code

> The implementations can be found in `src/hdsemg_shared/filters/` —
> `bandpass.py`, `lowpass.py`, `smoothing.py` and `padding.py`.

## API Dokumentation

::: hdsemg_shared.filters.bandpass
    handler: python
    options:
      heading_level: 3

::: hdsemg_shared.filters.lowpass
    handler: python
    options:
      heading_level: 3

::: hdsemg_shared.filters.smoothing
    handler: python
    options:
      heading_level: 3

::: hdsemg_shared.filters.padding
    handler: python
    options:
      heading_level: 3
