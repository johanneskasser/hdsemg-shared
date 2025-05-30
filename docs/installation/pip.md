# Installation with Pip

To install `hdsemg-shared` using pip, follow these steps:

## 1. (Optional) Create a virtual environment

It is recommended to use a virtual environment to avoid dependency conflicts:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 2. Install the package from PyPI

```bash
pip install hdsemg-shared
```

This will automatically install all required dependencies (`numpy`, `scipy`, etc.).

## 3. (Optional) Install in editable mode for development
If you are developing or testing the module locally, you can install it in editable mode:
```bash
git clone https://github.com/johanneskasser/hdsemg-shared.git
cd hdsemg-shared
pip install -e .
```

This allows you to make changes to the code without needing to reinstall the package each time.
For more information on editable installs, see the [pip documentation](https://pip.pypa.io/en/stable/topics/local-project-installs/).

```bash
---

For more information and troubleshooting, see the [documentation](../index.md).

