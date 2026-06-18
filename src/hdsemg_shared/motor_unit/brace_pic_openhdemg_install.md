# Brace PIC openhdemg/Bambi Environment

These files recreate the notebook environment used for the brace PIC
`openhdemg` sample workflow and Bambi sensitivity model.

## Conda Install

Run from the repository root:

```powershell
conda env create -f src\hdsemg_shared\motor_unit\brace_pic_openhdemg_environment.yml
conda activate bambi-openhdemg
python -m pip install -e .
python -m ipykernel install --user --name bambi-openhdemg --display-name "Python (bambi-openhdemg)"
jupyter lab --notebook-dir D:\Git\hdsemg-shared
```

## Pip Fallback

Use this only when conda is unavailable:

```powershell
python -m venv .venv-brace-pic
.\.venv-brace-pic\Scripts\Activate.ps1
python -m pip install -r src\hdsemg_shared\motor_unit\brace_pic_openhdemg_requirements.txt
python -m pip install -e .
python -m ipykernel install --user --name bambi-openhdemg --display-name "Python (bambi-openhdemg)"
```

## Notes

- The package itself is installed from the local repository with `pip install -e .`
  rather than pinned to the published `hdsemg-shared` wheel.
- The conda YAML pins the direct packages used by the notebook. It does not
  include machine-specific absolute paths from `pip freeze`.
- If PyTensor cannot write to its default compile cache, set a writable cache
  before fitting Bambi models:

```powershell
$env:PYTENSOR_FLAGS = "base_compiledir=D:\Data_local\Test_pic\.pytensor"
```
