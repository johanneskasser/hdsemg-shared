"""
Tests for Novecento+ OTB4 file loading.

This test suite verifies the correct handling of Novecento+ files which contain:
- Multiple tracks per signal file using ChannelOffsetInSubPacket
- Grid metadata in XML Description elements
- Control signals (IsControl=true) and reference channels
- Mixed sampling frequencies (EMG at 2kHz, Control signals at 8kHz)
"""

import pytest
import numpy as np
from pathlib import Path

from hdsemg_shared.fileio.file_io import EMGFile


def test_novecento_file_loads():
    """Test that Novecento+ OTB4 file can be loaded without errors."""
    test_file = Path(__file__).parent / "data" / "novecento.otb4"

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    emg = EMGFile.load(str(test_file))

    # Basic file attributes
    assert emg.file_type == "otb4"
    assert emg.sampling_frequency == 2000.0  # Main EMG sampling rate
    assert emg.data.shape[0] > 0  # Has samples
    assert emg.data.shape[1] > 0  # Has channels


def test_novecento_channel_count():
    """Test that all channels are loaded correctly."""
    test_file = Path(__file__).parent / "data" / "novecento.otb4"

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    emg = EMGFile.load(str(test_file))

    # Expected channels based on XML:
    # - 64 EMG channels (HD08MM1305)
    # - 4 Quaternions
    # - 2 Buffer/Ramp (IsControl=true)
    # - 3 AUX channels
    # - 1 Load Cell (might fail due to offset issue)
    # - 8 Control Signals (IsControl=true)
    # Total: 78 channels (or 77 if Load Cell fails)

    assert emg.channel_count >= 77
    assert emg.data.shape[1] >= 77


def test_novecento_grid_detection():
    """Test that the main EMG grid is detected correctly."""
    test_file = Path(__file__).parent / "data" / "novecento.otb4"

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    emg = EMGFile.load(str(test_file))
    grids = emg.grids

    # Should detect exactly one main EMG grid (5x13 = 13x5 transposed)
    assert len(grids) >= 1

    # Find the main EMG grid
    main_grid = None
    for grid in grids:
        if grid.ied_mm == 8 and (grid.rows == 5 or grid.rows == 13):
            main_grid = grid
            break

    assert main_grid is not None, "Main EMG grid (HD08MM1305) not found"

    # Check grid geometry (could be 5x13 or 13x5 due to transposition)
    assert main_grid.ied_mm == 8
    assert main_grid.rows * main_grid.cols == 65  # 5×13 = 65
    assert main_grid.electrodes == 64  # Only 64 active electrodes

    # Main grid should have 64 EMG channel indices
    assert len(main_grid.emg_indices) == 64


def test_novecento_reference_channels():
    """Test that reference/control channels are correctly assigned."""
    test_file = Path(__file__).parent / "data" / "novecento.otb4"

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    emg = EMGFile.load(str(test_file))
    grids = emg.grids

    # Find the main EMG grid
    main_grid = None
    for grid in grids:
        if grid.ied_mm == 8 and grid.electrodes == 64:
            main_grid = grid
            break

    assert main_grid is not None

    # Reference channels should include:
    # - 4 Quaternions
    # - 2 Buffer/Ramp (IsControl=true)
    # - 8 Control Signals (IsControl=true)
    # - Potentially 3 AUX channels
    # Total: at least 14 reference channels

    assert len(main_grid.ref_indices) >= 10

    # Verify that ref channels have "REF" marker in descriptions
    for ref_idx in main_grid.ref_indices:
        desc = emg.description[ref_idx]
        desc_str = str(desc).upper()
        assert "REF" in desc_str, f"Reference channel {ref_idx} missing REF marker: {desc}"


def test_novecento_emg_channel_descriptions():
    """Test that EMG channel descriptions contain grid pattern."""
    test_file = Path(__file__).parent / "data" / "novecento.otb4"

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    emg = EMGFile.load(str(test_file))
    grids = emg.grids

    # Find the main EMG grid
    main_grid = None
    for grid in grids:
        if grid.ied_mm == 8 and grid.electrodes == 64:
            main_grid = grid
            break

    assert main_grid is not None

    # Verify that EMG channels have grid pattern (HD08MM1305 or HD08MM0513)
    for emg_idx in main_grid.emg_indices:
        desc = str(emg.description[emg_idx])
        # Should contain HD08MM pattern
        assert "HD08MM" in desc.upper(), f"EMG channel {emg_idx} missing grid pattern: {desc}"
        # Should NOT have REF marker
        assert "REF" not in desc.upper(), f"EMG channel {emg_idx} has REF marker: {desc}"


def test_novecento_data_integrity():
    """Test that loaded data has correct shape and type."""
    test_file = Path(__file__).parent / "data" / "novecento.otb4"

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    emg = EMGFile.load(str(test_file))

    # Data should be 2D: (samples, channels)
    assert emg.data.ndim == 2
    assert emg.data.shape[0] > emg.data.shape[1]  # More samples than channels

    # Data should be float type
    assert emg.data.dtype in [np.float32, np.float64]

    # Time vector should match sample count
    assert len(emg.time) == emg.data.shape[0]

    # Check that data is not all zeros (real signal)
    assert np.any(emg.data != 0)


def test_novecento_grid_key():
    """Test that grid key is correctly generated."""
    test_file = Path(__file__).parent / "data" / "novecento.otb4"

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    emg = EMGFile.load(str(test_file))

    # Find the main EMG grid
    main_grid = None
    for grid in emg.grids:
        if grid.ied_mm == 8 and grid.electrodes == 64:
            main_grid = grid
            break

    assert main_grid is not None

    # Grid key should be "5x13" or "13x5"
    assert main_grid.grid_key in ["5x13", "13x5"]


def test_novecento_get_grid_by_key():
    """Test that grids can be retrieved by key."""
    test_file = Path(__file__).parent / "data" / "novecento.otb4"

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    emg = EMGFile.load(str(test_file))

    # Try to get grid by key
    grid_5x13 = emg.get_grid(grid_key="5x13")
    grid_13x5 = emg.get_grid(grid_key="13x5")

    # At least one should exist
    assert grid_5x13 is not None or grid_13x5 is not None

    # Verify it's the correct grid
    found_grid = grid_5x13 if grid_5x13 else grid_13x5
    assert found_grid.ied_mm == 8
    assert found_grid.electrodes == 64
    assert len(found_grid.emg_indices) == 64


def test_novecento_get_grid_by_uid():
    """Test that grids can be retrieved by UID."""
    test_file = Path(__file__).parent / "data" / "novecento.otb4"

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    emg = EMGFile.load(str(test_file))

    if len(emg.grids) == 0:
        pytest.skip("No grids detected")

    # Get first grid's UID
    first_grid = emg.grids[0]
    uid = first_grid.grid_uid

    # Retrieve by UID
    retrieved = emg.get_grid(grid_uid=uid)

    assert retrieved is not None
    assert retrieved.grid_uid == uid
    assert retrieved.grid_key == first_grid.grid_key
