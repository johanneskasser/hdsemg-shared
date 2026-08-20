"""Grid dimension normalization against the OTB product catalog (issue #50)."""
import json
import numpy as np
import pytest

import hdsemg_shared.fileio.file_io as FIOmod
from hdsemg_shared.fileio.file_io import EMGFile

# Real catalog entries: HD10MM0408 does not exist, HD10MM0804 does.
CATALOG = [
    {"product": "HD10MM0804", "electrodes": 32},
    {"product": "GR08MM1305", "electrodes": 64},
]


def _emg(monkeypatch, tmp_path, model_code, catalog=CATALOG):
    cache = tmp_path / "grid.json"
    cache.write_text(json.dumps(catalog))
    monkeypatch.setattr(EMGFile, "CACHE_PATH", str(cache))
    monkeypatch.setattr(EMGFile, "_grid_cache", None)
    monkeypatch.setattr(FIOmod.requests, "get", lambda *a, **k: pytest.fail("no HTTP"))

    desc = [f"Novecento+ (141 - 172) {model_code} ch1 [MUSCLE:Vastus Medialis Muscle Right]"]
    monkeypatch.setattr(
        FIOmod.MatFileIO, "load",
        staticmethod(lambda path: (np.zeros((2, 1)), np.arange(2), desc, 2000, "f.mat", 1)),
    )
    return EMGFile.load("f.mat").grids[0]


def test_reversed_digits_are_normalized_to_the_real_product(monkeypatch, tmp_path):
    g = _emg(monkeypatch, tmp_path, "HD10MM0408")
    assert (g.rows, g.cols) == (8, 4)
    assert g.grid_key == "10mm_8x4"
    assert g.electrodes == 32
    assert g.model_code == "HD10MM0408"  # provenance preserved


def test_correct_digits_are_left_alone(monkeypatch, tmp_path):
    g = _emg(monkeypatch, tmp_path, "HD10MM0804")
    assert (g.rows, g.cols) == (8, 4)
    assert g.model_code == "HD10MM0804"


def test_gr_products_are_recognized(monkeypatch, tmp_path):
    g = _emg(monkeypatch, tmp_path, "GR08MM1305")
    assert (g.rows, g.cols, g.ied_mm) == (13, 5, 8)
    assert g.electrodes == 64


def test_unknown_product_keeps_parsed_order(monkeypatch, tmp_path):
    g = _emg(monkeypatch, tmp_path, "HD99MM0203", catalog=[])
    assert (g.rows, g.cols) == (2, 3)
    assert g.electrodes == 6


# --- OTB4 model code source (issue #50 root cause) --------------------------
from hdsemg_shared.fileio.otb_4_file_io import grid_pattern_from_info


def test_model_code_comes_from_grid_name_not_nrow_ncolumn():
    # Real GridInfo from tests/data/*.otb4: NRow/NColumn are in the opposite
    # orientation to the digits in the product code.
    gi = {"Name": "HD08MM1305", "NRow": 5, "NColumn": 13, "IED": 8}
    assert grid_pattern_from_info(gi) == "HD08MM1305"  # not HD08MM0513


def test_non_product_names_fall_back_to_synthesized_code():
    gi = {"Name": "Control Signal", "NRow": 1, "NColumn": 1, "IED": 1}
    assert grid_pattern_from_info(gi) == "HD01MM0101"


def test_otb4_grid_has_catalog_orientation():
    from hdsemg_shared.fileio.file_io import EMGFile
    g = EMGFile.load("tests/data/CE13_TibAnt_AM_04062025_Trap3.otb4").grids[0]
    assert (g.rows, g.cols) == (13, 5)
    assert g.grid_key == "8mm_13x5"
    assert g.model_code == "HD08MM1305"
    assert len(g.emg_indices) == 64
