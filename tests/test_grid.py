import numpy as np
import pytest
from hdsemg_shared.grid import handle_entry, extract_grid_info, get_electrodes_from_grid_name


def test_handle_entry_string():
    assert handle_entry("test-entry") == "test-entry"


def test_handle_entry_numpy():
    desc = np.array([["wrapped-entry"]], dtype=object)
    assert handle_entry(desc) == "wrapped-entry"


def test_handle_entry_invalid():
    with pytest.raises(ValueError):
        handle_entry(12345)  # Not str or np.ndarray


def test_extract_grid_info_basic(monkeypatch):
    # Mock grid data to skip network call
    monkeypatch.setattr("hdsemg_shared.grid.grid_data", [
        {"product": "HD10MM0808", "electrodes": 64}
    ])

    description = [
        np.array([["HD10MM0808"]], dtype=object),                    # grid match
        np.array([["adapter-control-signal"]], dtype=object),        # ref
        np.array([["requested path MVC=55"]], dtype=object),         # request idx
        np.array([["performed path MVC=55"]], dtype=object),         # performed idx
    ]

    info = extract_grid_info(description)
    assert "8x8" in info
    grid = info["8x8"]
    assert grid["rows"] == 8
    assert grid["cols"] == 8
    assert grid["ied_mm"] == 10
    assert grid["requested_path_idx"] == 2
    assert grid["performed_path_idx"] == 3
    assert grid["reference_signals"][0]["index"] == 1

def test_get_electrodes_known(monkeypatch):
    # pretend grid_data has one entry
    monkeypatch.setenv("HOME", ".")  # ensure no real cache is used
    monkeypatch.setattr("hdsemg_shared.grid.grid_data", [
        {"product": "HD05MM0404", "electrodes": 16}
    ])
    assert get_electrodes_from_grid_name("HD05MM0404") == 16
    # case-insensitive match
    assert get_electrodes_from_grid_name("hd05mm0404") == 16

def test_get_electrodes_unknown():
    # grid_data untouched or empty
    from hdsemg_shared.grid import grid_data as gd
    # ensure unknown name returns None
    assert get_electrodes_from_grid_name("NO_SUCH_GRID") is None

def test_extract_grid_info_no_match(monkeypatch):
    # monkeypatch grid_data so extract_grid_info won't attempt to reload
    monkeypatch.setattr("hdsemg_shared.grid.grid_data", [])
    # no grid-like entries in description
    desc = [np.array([["foobar"]], dtype=object), np.array([["something else"]], dtype=object)]
    info = extract_grid_info(desc)
    assert isinstance(info, dict)
    assert info == {}

def test_extract_grid_info_two_grids(monkeypatch):
    monkeypatch.setattr("hdsemg_shared.grid.grid_data", [
        {"product": "HD10MM0402", "electrodes": 8},
        {"product": "HD08MM0303", "electrodes": 9},
    ])
    # simulate two grids and some refs and paths
    desc = [
        np.array([["HD10MM0402"]], dtype=object),  # first grid
        np.array([["grid1-ref"]], dtype=object),
        np.array([["requested path A"]], dtype=object),
        np.array([["HD08MM0303"]], dtype=object),  # second grid
        np.array([["grid2-ref"]], dtype=object),
        np.array([["performed path B"]], dtype=object),
    ]
    info = extract_grid_info(desc)
    # Expect two keys "4x2" and "3x3"
    assert set(info.keys()) == {"4x2", "3x3"}
    # First grid
    g1 = info["4x2"]
    assert g1["rows"] == 4 and g1["cols"] == 2
    assert g1["indices"] == [0]
    assert g1["reference_signals"][0]["index"] == 1
    assert g1["requested_path_idx"] == 2
    # Second grid
    g2 = info["3x3"]
    assert g2["rows"] == 3 and g2["cols"] == 3
    assert g2["indices"] == [3]
    assert g2["reference_signals"][0]["index"] == 4
    assert g2["performed_path_idx"] == 5
