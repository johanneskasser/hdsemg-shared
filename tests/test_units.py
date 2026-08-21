"""Signal-unit declaration on EMGFile (issue #53)."""
import numpy as np
import pytest

import hdsemg_shared.fileio.file_io as FIOmod
from hdsemg_shared.fileio.file_io import EMGFile
from hdsemg_shared.fileio.edf_file_io import _header_unit
from hdsemg_shared.fileio.otb_4_file_io import emg_unit_from_tracks
from hdsemg_shared.fileio.units import conversion_factor, normalize_unit


# ---------------------------------------------------------------- normalize
@pytest.mark.parametrize("raw,expected", [
    ("mV", "mV"), ("MV", "mV"), (" millivolt ", "mV"),
    ("uV", "uV"), ("µV", "uV"), ("μV", "uV"), ("microvolts", "uV"),
    ("V", "V"), ("volt", "V"),
    ("A.U.", "a.u."), ("a.u.", "a.u."), ("au", "a.u."),
    ("", None), (None, None), ("newton", None), ("kV", None),
])
def test_normalize_unit(raw, expected):
    assert normalize_unit(raw) == expected


# ---------------------------------------------------------------- conversion
def test_conversion_factor():
    assert conversion_factor("mV", "uV") == pytest.approx(1000.0)
    assert conversion_factor("uV", "mV") == pytest.approx(1e-3)
    assert conversion_factor("V", "uV") == pytest.approx(1e6)
    assert conversion_factor("mV", "mV") == pytest.approx(1.0)


def test_conversion_factor_rejects_unknown_and_arbitrary():
    with pytest.raises(ValueError, match="unknown"):
        conversion_factor(None, "uV")
    with pytest.raises(ValueError, match="Cannot convert"):
        conversion_factor("a.u.", "uV")
    with pytest.raises(ValueError, match="Cannot convert"):
        conversion_factor("mV", "a.u.")


# ---------------------------------------------------------------- otb4 tracks
def _track(unit, *, rows=13, cols=5, ied=8, control=False):
    return {
        "UnitOfMeasurement": unit,
        "IsControl": control,
        "GridInfo": {"Name": "HD08MM1305", "NRow": rows, "NColumn": cols, "IED": ied},
    }


def test_otb4_unit_ignores_control_and_pseudo_grids():
    tracks = [
        _track("mV"),
        _track("A.U.", control=True),                    # control signal
        _track("A.U.", rows=1, cols=4, ied=1),           # quaternion pseudo-grid
        {"UnitOfMeasurement": "V", "IsControl": False, "GridInfo": None},  # AUX
    ]
    assert emg_unit_from_tracks(tracks) == "mV"


def test_otb4_unit_none_when_grid_tracks_disagree():
    assert emg_unit_from_tracks([_track("mV"), _track("uV")]) is None


def test_otb4_unit_none_when_nothing_declared():
    assert emg_unit_from_tracks([_track("")]) is None


# ---------------------------------------------------------------- EMGFile
def _fake_load(unit):
    """A 2-channel EMG grid plus one reference channel, in `unit`."""
    def loader(path):
        data = np.array([[1.0, 2.0, 100.0],
                         [3.0, 4.0, 200.0],
                         [5.0, 6.0, 300.0],
                         [7.0, 8.0, 400.0]])
        time_arr = np.arange(4.0)
        desc = ["HD10MM0203 ch1", "HD10MM0203 ch2", "performed path"]
        return data, time_arr, desc, 1000, "f.mat", 42, unit
    return loader


@pytest.fixture(autouse=True)
def _no_catalog_fetch(monkeypatch):
    monkeypatch.setattr(EMGFile, "_grid_cache", [])


def test_unit_from_loader(monkeypatch):
    monkeypatch.setattr(FIOmod.MatFileIO, "load", _fake_load("mV"))
    assert EMGFile.load("f.mat").unit == "mV"


def test_unit_is_none_when_loader_declares_nothing(monkeypatch):
    monkeypatch.setattr(FIOmod.MatFileIO, "load", _fake_load(None))
    emg = EMGFile.load("f.mat")
    assert emg.unit is None
    with pytest.raises(ValueError):
        emg.scale_to("uV")


def test_load_tolerates_legacy_six_tuple_loader(monkeypatch):
    six = lambda p: _fake_load("mV")(p)[:6]
    monkeypatch.setattr(FIOmod.MatFileIO, "load", six)
    assert EMGFile.load("f.mat").unit is None


def test_to_unit_scales_emg_only_and_leaves_original(monkeypatch):
    monkeypatch.setattr(FIOmod.MatFileIO, "load", _fake_load("mV"))
    emg = EMGFile.load("f.mat")
    grid = emg.grids[0]
    assert grid.emg_indices == [0, 1] and grid.ref_indices == [2]

    converted = emg.to_unit("uV")

    assert converted.unit == "uV"
    np.testing.assert_allclose(converted.data[:, grid.emg_indices],
                               emg.data[:, grid.emg_indices] * 1000)
    # the reference channel keeps its own (unknown) unit
    np.testing.assert_allclose(converted.data[:, grid.ref_indices],
                               emg.data[:, grid.ref_indices])
    # source is untouched
    assert emg.unit == "mV"
    np.testing.assert_allclose(emg.data[0], [1.0, 2.0, 100.0])


def test_mat_round_trip_preserves_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(FIOmod.MatFileIO, "load", _fake_load("mV"))
    emg = EMGFile.load("f.mat")
    monkeypatch.undo()

    out = tmp_path / "out.mat"
    emg.save(str(out))
    assert EMGFile.load(str(out)).unit == "mV"


def test_mat_without_unit_stays_none(tmp_path, monkeypatch):
    monkeypatch.setattr(FIOmod.MatFileIO, "load", _fake_load(None))
    emg = EMGFile.load("f.mat")
    monkeypatch.undo()

    out = tmp_path / "out.mat"
    emg.save(str(out))
    assert EMGFile.load(str(out)).unit is None


# ---------------------------------------------------------------- edf header
def _edf_header(dims):
    return {"phys_dims": dims}


def test_edf_unit_from_physical_dimension():
    assert _header_unit(_edf_header(["uV", "uV", "uV"]), set()) == "uV"


def test_edf_unit_ignores_annotation_signals():
    assert _header_unit(_edf_header(["uV", ""]), {1}) == "uV"


def test_edf_unit_none_when_blank_or_conflicting():
    assert _header_unit(_edf_header(["", ""]), set()) is None
    assert _header_unit(_edf_header(["uV", "mV"]), set()) is None
