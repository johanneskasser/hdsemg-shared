import os
import json
import time
import uuid
import numpy as np
import pytest
from pathlib import Path

import hdsemg_shared.fileio.file_io as FIOmod
from hdsemg_shared.fileio.file_io import EMGFile, Grid, MatFileIO

class DummyResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._data

@pytest.fixture(autouse=True)
def patch_loaders(monkeypatch):
    # Always reset the in-memory cache
    EMGFile._grid_cache = None

    def fake_mat_load(path):
        # 2×3 int16 data, 3×1 time, 5 entries of description
        data = np.array([[1,2,3],[4,5,6]], dtype=np.int16)
        time_arr = np.array([[0],[1],[2]])
        desc = [
            "HD10MM0203 - header",  # header as plain str
            np.array([["chan1"]], dtype=object),  # a normal channel
            np.array([["requested path info"]], dtype=object),
            np.array([["performed path info"]], dtype=object),
            np.array([["ref_signal"]], dtype=object)
        ]
        sf = 1000
        fn = Path(path).name
        fs = 1234
        return data, time_arr, desc, sf, fn, fs

    monkeypatch.setattr(MatFileIO, "load", fake_mat_load)
    monkeypatch.setattr(FIOmod, "load_otb_file", lambda p: fake_mat_load(p))
    monkeypatch.setattr(FIOmod, "load_otb4_file", lambda p: fake_mat_load(p))

    yield

def test_unsupported_extension_raises(tmp_path):
    with pytest.raises(ValueError):
        EMGFile.load(str(tmp_path / "file.unknown"))

def test_int16_cast_and_sanitize_transpose_and_time_swap():
    emg = EMGFile.load("some.mat")
    # dtype converted
    assert emg.data.dtype == np.float32
    # shape was 2×3 → transpose to 3×2
    assert emg.data.shape == (3, 2)
    # time squeezed to (3,)
    assert emg.time.shape == (3,)
    # rows match
    assert emg.time.shape[0] == emg.data.shape[0]

def test_sanitize_incompatible_time_raises():
    data = np.zeros((4,2))
    bad_time = np.arange(5)
    with pytest.raises(ValueError):
        EMGFile._sanitize(data, bad_time)

def test_grids_extraction(monkeypatch, tmp_path):
    # write a valid cache so HTTP is never needed
    fake_grid_json = [{"product":"HD10MM0203","electrodes":100}]
    cache_file = tmp_path / "grid.json"
    cache_file.write_text(json.dumps(fake_grid_json))

    # point the class at our cache, disable HTTP
    monkeypatch.setattr(EMGFile, "CACHE_PATH", str(cache_file))
    monkeypatch.setattr(EMGFile, "GRID_JSON_URL", "http://nope")
    monkeypatch.setattr(FIOmod.requests, "get", lambda *a, **k: pytest.skip("Shouldn't fetch"))

    emg = EMGFile.load("dummy.mat")
    grids = emg.grids  # triggers cache read

    assert len(grids) == 1
    g = grids[0]
    assert isinstance(g, Grid)

    # Header only at idx 0
    assert g.emg_indices == [0]

    # All other entries become refs
    assert g.ref_indices == [1, 2, 3, 4]

    # path indices captured
    assert g.requested_path_idx == 2
    assert g.performed_path_idx == 3

    # geometry parsed correctly
    assert (g.rows, g.cols, g.ied_mm) == (2, 3, 10)
    assert g.electrodes == 100
    assert g.grid_key == "2x3"
    uuid.UUID(g.grid_uid)  # valid UUID

def test_grid_cache_refresh(monkeypatch, tmp_path):
    # create an expired cache
    cache = tmp_path / "old.json"
    old_data = [{"product":"X","electrodes":1}]
    cache.write_text(json.dumps(old_data))
    old_mtime = time.time() - 8*24*3600
    os.utime(cache, (old_mtime, old_mtime))

    # patch CACHE_PATH, reset in-memory cache, and fake HTTP
    monkeypatch.setattr(EMGFile, "CACHE_PATH", str(cache))
    monkeypatch.setattr(EMGFile, "_grid_cache", None)
    new_data = [{"product":"HD10MM0203","electrodes":42}]
    monkeypatch.setattr(FIOmod.requests, "get", lambda *a, **k: DummyResponse(new_data))
    monkeypatch.setattr(EMGFile, "GRID_JSON_URL", "http://fake")

    emg = EMGFile.load("f.mat")
    # trigger a refresh
    _ = emg.grids

    # now the cache file should have been rewritten
    reloaded = json.loads(cache.read_text())
    assert reloaded == new_data

    # and the new electrode count is used
    assert emg.grids[0].electrodes == 42
