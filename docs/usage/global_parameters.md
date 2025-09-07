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

## Envelope Estimation

### 7. EMG Envelope (ARV, RMS, Median)

**Description:**

* Smoothed versions of ARV/RMS/Median using low-pass filtering.

**Python Implementation:**

```python
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return filtfilt(b, a, signal)

def lowpass_filter(signal, fs, cutoff, order=4):
    b, a = butter(order, cutoff, btype='low', fs=fs)
    return filtfilt(b, a, signal)

def compute_emg_envelopes(emg: np.ndarray, fs: float, bpf: dict, lpf: dict):
    emg_filtered = np.array([
        bandpass_filter(ch, fs, bpf['fcl'], bpf['fch'], bpf['N']) for ch in emg
    ])
    arv = np.nanmean(np.abs(emg_filtered), axis=0)
    rms = np.sqrt(np.nanmean(emg_filtered ** 2, axis=0))
    mdn = np.nanmedian(np.abs(emg_filtered), axis=0)
    arv_env = lowpass_filter(arv, fs, lpf['fc'], lpf['N'])
    rms_env = lowpass_filter(rms, fs, lpf['fc'], lpf['N'])
    mdn_env = lowpass_filter(mdn, fs, lpf['fc'], lpf['N'])
    return arv_env, rms_env, mdn_env
```

**References:**

* Merletti & Farina (2016)
* CEDE Best Practice: Amplitude Matrix
