# EMG Global Parameter Functions

This repository contains modular Python implementations of key global surface EMG parameters. These functions are designed for reuse in signal processing pipelines and adhere to recommendations from scientific literature, particularly the CEDE (Consensus for Experimental Design in Electromyography) project and related EMG publications.

---

## 📦 Implemented Parameters

### 1. Root Mean Square (RMS)
- **Description**: Measures signal energy.
- **Formula**: `sqrt(mean(xᵢ²))`
- **Reference**: Merletti & Farina (2016), CEDE Clancy et al. (2023)

### 2. Average Rectified Value (ARV)
- **Description**: Mean of absolute values, linear estimate of EMG activity.
- **Formula**: `mean(|xᵢ|)`
- **Reference**: CEDE Amplitude Normalization

### 3. Integrated EMG (IEMG)
- **Description**: Total amplitude over time, indicates intensity.
- **Formula**: `sum(|xᵢ|)`
- **Reference**: Merletti & Farina (2016), CEDE

### 4. Mean Frequency (MNF)
- **Description**: Spectral centroid of the EMG power.
- **Formula**: `sum(fᵢ * Pᵢ) / sum(Pᵢ)`
- **Reference**: CEDE Force Estimation, Phinyomark et al. (2012)

### 5. Median Frequency (MDF)
- **Description**: Frequency dividing power spectrum in two.
- **Reference**: CEDE, Farina & Merletti

### 6. Permutation Entropy
- **Description**: Quantifies signal complexity.
- **Reference**: Bandt & Pompe (2002), CEDE SMU Matrix

### 7. Global Amplitude
- **Description**: Reduces a whole HDsEMG grid to one amplitude over time.
- **Processing**: Map → MP/SD/DD → Bandpass → Square (or rectify) → Smooth → Mean over channels → Root **last**
- **Formula**: `A(t) = sqrt(mean_ch(smooth(xᵢ(t)²)))`, ARV drops the root
- **Reference**: Merletti & Cerone eq. 5.1/5.2; Del Vecchio et al. (2025)

---

## 🛠️ Usage

Everything here needs only NumPy and SciPy.

```python
from hdsemg_shared.global_parameters.RMS import root_mean_square
value = root_mean_square(signal)

from hdsemg_shared.global_parameters import global_amplitude
out = global_amplitude(emg.T, emg_map, fs, method='RMS')
```

---

## 📚 Literature

- Clancy EA et al. (2023). *Amplitude Best Practices in EMG*. CEDE Project.
- Merletti, Farina. *Surface EMG: Physiology, Engineering and Applications.*
- Dideriksen JL et al. (2023). *CEDE Amplitude Normalization Matrix*.
- Farina D et al. (2023). *EMG for Muscle Force Estimation*. CEDE.
- Bandt C, Pompe B (2002). *Permutation Entropy*. Phys. Rev. Lett.

---

## 🧪 License
MIT License. Attribution appreciated if used for scientific work.

---
