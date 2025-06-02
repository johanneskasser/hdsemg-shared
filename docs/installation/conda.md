# Installation with Conda

To install `hdsemg-shared` using Conda, follow these steps:

## 1. Create a new environment (optional)

```bash
conda create -n hdsemg python=3.8
conda activate hdsemg
```

## 2. Install dependencies

```bash
conda install numpy scipy
```

## 3. Install `hdsemg-shared` from PyPI

```bash
pip install hdsemg-shared
```

> **Note:** The main library is installed via PyPI, as there is no Conda-Forge package. Core dependencies are installed via Conda for best compatibility.

---
