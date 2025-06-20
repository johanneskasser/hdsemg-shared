import numpy as np
import pytest
from hdsemg_shared.grid import handle_entry, extract_grid_info


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
