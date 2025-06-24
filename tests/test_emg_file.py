import os
import json
import time
import uuid
import numpy as np
import pytest
from pathlib import Path

import hdsemg_shared.fileio.file_io as FIO
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
    # Return a small dummy dataset for any loader
    def fake_mat_load(path):
        data = np.array([[1,2,3],[4,5,6]], dtype=np.int16)
        time_arr = np.array([[0],[1],[2]])        # 3×1 shape
        desc = ["HD10MM0203 - chan1",                      # grid header
                np.array([["chan1"]], dtype=object),
                ["requested path info"],
                ["performed path info"],
                "ref_signal"]
        sf = 1000
        fn = Path(path).name
        fs = 1234
        return data, time_arr, desc, sf, fn, fs

    monkeypatch.setattr(MatFileIO, "load", fake_mat_load)
    monkeypatch.setattr(FIO, "load_otb_file", lambda p: fake_mat_load(p))
    monkeypatch.setattr(FIO, "load_otb4_file", lambda p: fake_mat_load(p))

    yield

def test_unsupported_extension_raises(tmp_path):
    bogus = tmp_path / "file.xyz"
    with pytest.raises(ValueError):
        EMGFile.load(str(bogus))

def test_int16_cast_and_sanitize_transpose_and_time_swap():
    # fake_mat_load returns data shape (2×3) dtype=int16 and time shape (3×1)
    emg = EMGFile.load("some.mat")
    # data was int16 → float32
    assert emg.data.dtype == np.float32
    # sanitize: original data 2×3, so transposed → 3×2
    assert emg.data.shape == (3,2)
    # time squeezed from (3,1) → (3,)
    assert emg.time.shape == (3,)
    # time matches data rows
    assert emg.time.shape[0] == emg.data.shape[0]

def test_sanitize_incompatible_time_raises():
    # manually call _sanitize with mismatched shapes
    data = np.zeros((4,2))
    time = np.arange(5)
    with pytest.raises(ValueError):
        EMGFile._sanitize(data, time)

def test_grids_extraction(monkeypatch, tmp_path):
    # prepare a fake grid-data JSON cache
    fake_grid_json = [
        {"product":"HD10MM0203","electrodes":100}
    ]
    cache_file = tmp_path / "cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(fake_grid_json))

    # monkeypatch CACHE_PATH and disable real HTTP
    monkeypatch.setattr(EMGFile, "CACHE_PATH", str(cache_file))
    monkeypatch.setattr(EMGFile, "GRID_JSON_URL", "http://doesnot.exist")
    # ensure HTTP not called
    monkeypatch.setattr(FIO.requests, "get", lambda *args, **kw: pytest.skip("Shouldn't fetch"))

    emg = EMGFile.load("dummy.mat")
    grids = emg.grids
    # we had one header → one Grid
    assert len(grids) == 1
    g = grids[0]
    assert isinstance(g, Grid)
    # header at index 0 → emg_indices == [0]
    assert g.emg_indices == [0]
    # reference signals are entries after header minus requested/performed
    # in our desc: indices 1 (chan1) & 4 (ref_signal)
    assert set(g.ref_indices) == {1, 4}
    assert g.rows == 2    # from "MM02"
    assert g.cols == 3    # from "MM0203"
    assert g.ied_mm == 10 # from "HD10"
    # from fake JSON
    assert g.electrodes == 100
    # grid_key should be "2x3"
    assert g.grid_key == "2x3"
    # uuid auto‐generated
    uuid.UUID(g.grid_uid)

def test_grid_cache_refresh(monkeypatch, tmp_path):
    # make CACHE_PATH in tmp, write expired cache
    cache = tmp_path / "cache2.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    old = [{"product":"X","electrodes":1}]
    cache.write_text(json.dumps(old))
    # set mtime to 8 days ago
    old_time = time.time() - 8*24*3600
    os.utime(cache, (old_time, old_time))

    # monkeypatch CACHE_PATH and HTTP get
    monkeypatch.setattr(EMGFile, "CACHE_PATH", str(cache))
    new_data = [{"product":"HD10MM0203","electrodes":42}]
    monkeypatch.setattr(FIO.requests, "get", lambda *args, **kw: DummyResponse(new_data))
    monkeypatch.setattr(EMGFile, "GRID_JSON_URL", "http://fake")

    # after load, cache should be rewritten
    emg = EMGFile.load("f.mat")
    # read back cache
    reloaded = json.loads(cache.read_text())
    assert reloaded == new_data
    # grids should use electrodes=42
    assert emg.grids[0].electrodes == 42

