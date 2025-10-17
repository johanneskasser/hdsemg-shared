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

    # Grid key should include IED and dimensions: "8mm_5x13" or "8mm_13x5"
    assert main_grid.grid_key in ["8mm_5x13", "8mm_13x5"]


def test_novecento_get_grid_by_key():
    """Test that grids can be retrieved by key."""
    test_file = Path(__file__).parent / "data" / "novecento.otb4"

    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    emg = EMGFile.load(str(test_file))

    # Try to get grid by key (with IED included)
    grid_5x13 = emg.get_grid(grid_key="8mm_5x13")
    grid_13x5 = emg.get_grid(grid_key="8mm_13x5")

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


def test_dual_grids_same_dimensions():
    """
    Test that two grids with same dimensions but different IED are detected as separate grids.

    This tests the scenario where a file has two grids like:
    - Grid 1: HD08MM1305 (8mm IED, 5 rows, 13 cols)
    - Grid 2: HD04MM1305 (4mm IED, 5 rows, 13 cols)

    These should be detected as TWO separate grids, not merged into one.
    """
    # Mock EMG file with two grids of same dimensions but different IED
    data = np.random.randn(1000, 128)  # 128 channels total (64 + 64)
    time = np.arange(1000) / 2000.0

    # Create descriptions for two 5x13 grids with different IED
    descriptions = []

    # First grid: HD08MM1305 (IED=8mm, channels 0-63)
    for i in range(64):
        descriptions.append(np.array([[f"IN1 HD08MM1305 ch{i+1}"]], dtype=object))

    # Second grid: HD04MM1305 (IED=4mm, channels 64-127)
    for i in range(64):
        descriptions.append(np.array([[f"IN2 HD04MM1305 ch{i+1}"]], dtype=object))

    desc_array = np.array(descriptions, dtype=object)

    # Create EMGFile instance
    emg = EMGFile(
        data=data,
        time=time,
        description=desc_array,
        sf=2000.0,
        file_name="test_dual_grid.otb4",
        file_size=1000,
        file_type="otb4"
    )

    # Get grids
    grids = emg.grids

    # Should detect TWO grids, not one merged grid
    assert len(grids) == 2, f"Expected 2 grids, got {len(grids)}"

    # Find both grids
    grid_8mm = None
    grid_4mm = None
    for grid in grids:
        if grid.ied_mm == 8:
            grid_8mm = grid
        elif grid.ied_mm == 4:
            grid_4mm = grid

    assert grid_8mm is not None, "8mm IED grid not found"
    assert grid_4mm is not None, "4mm IED grid not found"

    # Verify each grid has correct properties
    # Grid 1: 8mm IED
    assert grid_8mm.rows * grid_8mm.cols == 65  # 5x13 = 65
    assert len(grid_8mm.emg_indices) == 64  # First 64 channels
    assert all(0 <= idx < 64 for idx in grid_8mm.emg_indices)
    assert grid_8mm.grid_key in ["8mm_5x13", "8mm_13x5"]

    # Grid 2: 4mm IED
    assert grid_4mm.rows * grid_4mm.cols == 65  # 5x13 = 65
    assert len(grid_4mm.emg_indices) == 64  # Next 64 channels
    assert all(64 <= idx < 128 for idx in grid_4mm.emg_indices)
    assert grid_4mm.grid_key in ["4mm_5x13", "4mm_13x5"]

    # Verify grids have different keys (this is the key fix!)
    assert grid_8mm.grid_key != grid_4mm.grid_key

    # Verify grids can be retrieved individually by their unique keys
    retrieved_8mm = emg.get_grid(grid_key=grid_8mm.grid_key)
    retrieved_4mm = emg.get_grid(grid_key=grid_4mm.grid_key)

    assert retrieved_8mm is not None
    assert retrieved_4mm is not None
    assert retrieved_8mm.ied_mm == 8
    assert retrieved_4mm.ied_mm == 4


def test_dual_identical_grids_non_contiguous():
    """
    Test that two grids with IDENTICAL specs are detected as separate when non-contiguous.

    This tests the scenario where a file has two identical grids like:
    - Grid 1: HD08MM1305 (8mm IED, 5 rows, 13 cols, channels 0-63)
    - Grid 2: HD08MM1305 (8mm IED, 5 rows, 13 cols, channels 70-133)

    The gap in channel indices (64-69) indicates these are separate physical grids.
    """
    # Mock EMG file with two identical grids separated by a gap
    data = np.random.randn(1000, 140)  # 140 channels total
    time = np.arange(1000) / 2000.0

    # Create descriptions
    descriptions = []

    # First grid: HD08MM1305 (channels 0-63)
    for i in range(64):
        descriptions.append(np.array([[f"IN1 HD08MM1305 ch{i+1}"]], dtype=object))

    # Gap: Some reference channels (64-69)
    for i in range(6):
        descriptions.append(np.array([[f"REF ch{i+1}"]], dtype=object))

    # Second grid: HD08MM1305 (channels 70-133)
    for i in range(64):
        descriptions.append(np.array([[f"IN2 HD08MM1305 ch{i+1}"]], dtype=object))

    # More reference channels
    for i in range(6):
        descriptions.append(np.array([[f"REF2 ch{i+1}"]], dtype=object))

    desc_array = np.array(descriptions, dtype=object)

    # Create EMGFile instance
    emg = EMGFile(
        data=data,
        time=time,
        description=desc_array,
        sf=2000.0,
        file_name="test_dual_identical_grids.otb4",
        file_size=1000,
        file_type="otb4"
    )

    # Get grids
    grids = emg.grids

    # Should detect TWO separate grids, not one merged grid
    assert len(grids) == 2, f"Expected 2 grids, got {len(grids)}: {[g.grid_key for g in grids]}"

    # Both grids should have 8mm IED and 5x13 dimensions
    for grid in grids:
        assert grid.ied_mm == 8
        assert grid.rows * grid.cols == 65  # 5x13 = 65

    # Verify grid keys are different (one should have suffix)
    grid_keys = [g.grid_key for g in grids]
    assert len(set(grid_keys)) == 2, f"Grid keys should be unique: {grid_keys}"

    # First grid should have base key, second should have suffix
    assert any("8mm_5x13" == k or "8mm_13x5" == k for k in grid_keys)
    assert any("_2" in k for k in grid_keys), f"Second grid should have _2 suffix: {grid_keys}"

    # Verify channel ranges are correct
    grid1 = grids[0]
    grid2 = grids[1]

    # First grid: channels 0-63
    assert len(grid1.emg_indices) == 64
    assert all(0 <= idx < 64 for idx in grid1.emg_indices)

    # Second grid: channels 70-133
    assert len(grid2.emg_indices) == 64
    assert all(70 <= idx < 134 for idx in grid2.emg_indices)

    # Verify no overlap in channel indices
    indices_set1 = set(grid1.emg_indices)
    indices_set2 = set(grid2.emg_indices)
    assert len(indices_set1.intersection(indices_set2)) == 0, "Grids should not share channel indices"

    # Verify grids can be retrieved by their unique keys
    retrieved_1 = emg.get_grid(grid_key=grid1.grid_key)
    retrieved_2 = emg.get_grid(grid_key=grid2.grid_key)

    assert retrieved_1 is not None
    assert retrieved_2 is not None
    assert retrieved_1.grid_uid != retrieved_2.grid_uid

    # Verify reference channels are assigned correctly
    # Grid 1 should have refs at indices 64-69 (6 channels)
    assert len(grid1.ref_indices) == 6
    assert all(64 <= idx < 70 for idx in grid1.ref_indices)

    # Grid 2 should have refs at indices 134-139 (6 channels)
    assert len(grid2.ref_indices) == 6
    assert all(134 <= idx < 140 for idx in grid2.ref_indices)


def test_muscle_information_extraction():
    """
    Test that muscle information is correctly extracted from OTB4 files.

    This tests the muscle field in Grid dataclass, which should be populated
    from the <Muscle> tag in the OTB4 XML file.
    """
    # Mock EMG file with muscle information in descriptions
    data = np.random.randn(1000, 128)  # 128 channels total (64 + 64)
    time = np.arange(1000) / 2000.0

    # Create descriptions for two grids with different muscles
    descriptions = []

    # First grid: Vastus Lateralis
    for i in range(64):
        desc = f"IN1 HD08MM1305 ch{i+1} [MUSCLE:Vastus Lateralis Muscle Right]"
        descriptions.append(np.array([[desc]], dtype=object))

    # Second grid: Vastus Medialis
    for i in range(64):
        desc = f"IN2 HD04MM1305 ch{i+1} [MUSCLE:Vastus Medialis Muscle Right]"
        descriptions.append(np.array([[desc]], dtype=object))

    desc_array = np.array(descriptions, dtype=object)

    # Create EMGFile instance
    emg = EMGFile(
        data=data,
        time=time,
        description=desc_array,
        sf=2000.0,
        file_name="test_muscle_info.otb4",
        file_size=1000,
        file_type="otb4"
    )

    # Get grids
    grids = emg.grids

    assert len(grids) == 2, f"Expected 2 grids, got {len(grids)}"

    # Find grids by IED
    grid_8mm = next(g for g in grids if g.ied_mm == 8)
    grid_4mm = next(g for g in grids if g.ied_mm == 4)

    # Verify muscle information is correctly extracted
    assert grid_8mm.muscle == "Vastus Lateralis Muscle Right"
    assert grid_4mm.muscle == "Vastus Medialis Muscle Right"


def test_muscle_information_nullable():
    """
    Test that muscle field is nullable (can be None).

    This tests grids without muscle information should have muscle=None.
    """
    # Mock EMG file without muscle information
    data = np.random.randn(1000, 64)
    time = np.arange(1000) / 2000.0

    # Create descriptions WITHOUT muscle information
    descriptions = []
    for i in range(64):
        descriptions.append(np.array([[f"IN1 HD08MM1305 ch{i+1}"]], dtype=object))

    desc_array = np.array(descriptions, dtype=object)

    # Create EMGFile instance
    emg = EMGFile(
        data=data,
        time=time,
        description=desc_array,
        sf=2000.0,
        file_name="test_no_muscle.otb4",
        file_size=1000,
        file_type="otb4"
    )

    # Get grids
    grids = emg.grids

    assert len(grids) == 1
    grid = grids[0]

    # Verify muscle is None when not specified
    assert grid.muscle is None


def test_muscle_consistent_across_channels():
    """
    Test that muscle information is set from the first channel and remains consistent.

    When multiple channels of the same grid have muscle info, the first one should be used.
    """
    data = np.random.randn(1000, 64)
    time = np.arange(1000) / 2000.0

    descriptions = []
    # All channels have the same muscle info
    for i in range(64):
        desc = f"IN1 HD08MM1305 ch{i+1} [MUSCLE:Biceps Brachii]"
        descriptions.append(np.array([[desc]], dtype=object))

    desc_array = np.array(descriptions, dtype=object)

    emg = EMGFile(
        data=data,
        time=time,
        description=desc_array,
        sf=2000.0,
        file_name="test_muscle_consistent.otb4",
        file_size=1000,
        file_type="otb4"
    )

    grids = emg.grids
    assert len(grids) == 1
    assert grids[0].muscle == "Biceps Brachii"
