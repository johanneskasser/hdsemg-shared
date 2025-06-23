<div align="center">
<br>
  <img src="src/hdsemg_shared/resources/icon.png" alt="App Icon" width="100" height="100"><br>
    <h2 align="center">📦 hdsemg-shared 📦</h2>
    <h3 align="center">HDsEMG toolbox</h3>
</div>

[![PyPI Version](https://img.shields.io/pypi/v/hdsemg-shared.svg?style=flat-square)](https://pypi.org/project/hdsemg-shared/)
[![Python Versions](https://img.shields.io/pypi/pyversions/hdsemg-shared.svg?style=flat-square)](https://pypi.org/project/hdsemg-shared/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-hdsemg--shared-blue?logo=github&style=flat-square)](https://github.com/johanneskasser/hdsemg-shared)
[![codecov](https://codecov.io/gh/johanneskasser/hdsemg-shared/branch/main/graph/badge.svg)](https://codecov.io/gh/johanneskasser/hdsemg-shared)
[![Tested on Windows](https://img.shields.io/badge/tested%20on-windows-blue?logo=windows&style=flat-square)](#)



Reusable Python components and utilities for high-density surface EMG (HD-sEMG) signal processing and input/output (I/O).

This module provides shared logic for HD-sEMG signal processing and file handling, used across multiple related projects, such as `hdsemg-pipe` and `hdsemg-select`. It is installable as a standalone Python package and is designed to simplify working with HD-sEMG data.

---

## 📦 Installation

This library can be installed via PyPI or from source. It is compatible with Python 3.7 and later.
### Via PyPI
```bash
    python.exe -m pip install --upgrade pip 
    pip install hdsemg-shared
```
### From Source
```bash
    git clone https://github.com/johanneskasser/hdsemg-shared.git
    cd hdsemg-shared
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    pip install -e .
```
    

---

## 🧪 Documentation

> Please refer to the [documentation](https://johanneskasser.github.io/hdsemg-shared/) for detailed usage instructions, examples, and API references.