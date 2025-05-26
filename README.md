<div align="center">
<br>
  <img src="src/hdsemg_shared/resources/icon.png" alt="App Icon" width="100" height="100"><br>
    <h2 align="center">📦 hdsemg-shared 📦</h2>
    <h3 align="center">HDsEMG toolbox</h3>
</div>

[![PyPI Version](https://img.shields.io/pypi/v/hdsemg-shared.svg?style=flat-square)](https://pypi.org/project/hdsemg-shared/)
[![Python Versions](https://img.shields.io/pypi/pyversions/hdsemg-shared.svg?style=flat-square)](https://pypi.org/project/hdsemg-shared/)


Reusable Python components and utilities for high-density surface EMG (HD-sEMG) signal processing and input/output (I/O).

This module provides shared logic for HD-sEMG signal processing and file handling, used across multiple related projects, such as `hdsemg-pipe` and `hdsemg-select`. It is installable as a standalone Python package and is designed to simplify working with HD-sEMG data.

---

## 📦 Installation

This package lives inside a subdirectory (`src/shared_logic`) of a larger monorepo. It includes its own `setup.py` and can be installed directly via `pip`.

```bash
    python.exe -m pip install --upgrade pip 
    pip install hdsemg-shared
```

---

## 🧪 Local Development

If you're actively developing or testing the module locally, you can install it in editable mode:

```bash
cd path/to/hdsemg-pipe/src/shared_logic
pip install -e .
```

This will allow you to make code changes without reinstalling the package.

---

## 🧰 Requirements

This module requires:

- Python ≥ 3.7
- `numpy`
- `scipy`

These will be installed automatically via `install_requires` if not already present in your environment.
