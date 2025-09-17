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

### 7. Envelope (ARV, RMS, MDN)
- **Description**: Smoothed EMG amplitude using low-pass filtering.
- **Processing**: Bandpass → Rectification → Low-pass
- **Reference**: CEDE-Check, Merletti & Farina

---

## 🛠️ Usage

Each function is fully modular and requires only NumPy (plus SciPy for envelope and Welch analysis). Simply copy the relevant `.py` file and import the function.

```python
from rms import root_mean_square
value = root_mean_square(signal)
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
