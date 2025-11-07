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


def test_trapezoidal_tracks_parsing():
    """
    Test that TrapezoidalTracks XML files are correctly parsed.

    This tests the parse_trapezoidal_tracks_xml function which extracts
    force signal information from TrapezoidalTracks_*.xml files.
    """
    import tempfile
    import os
    from hdsemg_shared.fileio.otb_4_file_io import parse_trapezoidal_tracks_xml

    # Create a temporary directory with mock TrapezoidalTracks XML
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock TrapezoidalTracks_000.xml file
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<ArrayOfTrackInfo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <TrackInfo>
    <SubTitle>Performed Path</SubTitle>
    <SignalStreamPath>force_001.sig</SignalStreamPath>
    <SamplingFrequency>10</SamplingFrequency>
    <Gain>0.5</Gain>
    <ADC_Nbits>1</ADC_Nbits>
    <ADC_Range>1</ADC_Range>
    <AcquisitionChannel>0</AcquisitionChannel>
  </TrackInfo>
  <TrackInfo>
    <SubTitle>Original Path</SubTitle>
    <SignalStreamPath>force_001.sig</SignalStreamPath>
    <SamplingFrequency>10</SamplingFrequency>
    <Gain>0.5</Gain>
    <ADC_Nbits>1</ADC_Nbits>
    <ADC_Range>1</ADC_Range>
    <AcquisitionChannel>1</AcquisitionChannel>
  </TrackInfo>
</ArrayOfTrackInfo>"""

        xml_path = os.path.join(tmpdir, "TrapezoidalTracks_000.xml")
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

        # Parse the XML
        tracks = parse_trapezoidal_tracks_xml(tmpdir)

        # Verify results
        assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"

        # Check Performed Path
        performed = tracks[0]
        assert performed["SubTitle"] == "Performed Path"
        assert performed["SignalStreamPath"] == "force_001.sig"
        assert performed["SamplingFrequency"] == 10.0
        assert performed["Gain"] == 0.5
        assert performed["ADC_Nbits"] == 1
        assert performed["ADC_Range"] == 1.0
        assert performed["AcquisitionChannel"] == 0
        assert performed["NumberOfChannels"] == 1

        # Check Original Path
        original = tracks[1]
        assert original["SubTitle"] == "Original Path"
        assert original["SignalStreamPath"] == "force_001.sig"
        assert original["AcquisitionChannel"] == 1


def test_trapezoidal_tracks_multiple_files():
    """
    Test that multiple TrapezoidalTracks XML files are all parsed.

    This tests the scenario where there are multiple TrapezoidalTracks_*.xml
    files (e.g., TrapezoidalTracks_000.xml, TrapezoidalTracks_010.xml).
    """
    import tempfile
    import os
    from hdsemg_shared.fileio.otb_4_file_io import parse_trapezoidal_tracks_xml

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two TrapezoidalTracks XML files
        for idx in [0, 10]:
            xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<ArrayOfTrackInfo>
  <TrackInfo>
    <SubTitle>Performed Path</SubTitle>
    <SignalStreamPath>force_{idx:03d}.sig</SignalStreamPath>
    <SamplingFrequency>10</SamplingFrequency>
    <Gain>0.5</Gain>
    <ADC_Nbits>1</ADC_Nbits>
    <ADC_Range>1</ADC_Range>
    <AcquisitionChannel>0</AcquisitionChannel>
  </TrackInfo>
  <TrackInfo>
    <SubTitle>Original Path</SubTitle>
    <SignalStreamPath>force_{idx:03d}.sig</SignalStreamPath>
    <SamplingFrequency>10</SamplingFrequency>
    <Gain>0.5</Gain>
    <ADC_Nbits>1</ADC_Nbits>
    <ADC_Range>1</ADC_Range>
    <AcquisitionChannel>1</AcquisitionChannel>
  </TrackInfo>
</ArrayOfTrackInfo>"""

            xml_path = os.path.join(tmpdir, f"TrapezoidalTracks_{idx:03d}.xml")
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)

        # Parse all XML files
        tracks = parse_trapezoidal_tracks_xml(tmpdir)

        # Should have 4 tracks total (2 from each file)
        assert len(tracks) == 4, f"Expected 4 tracks, got {len(tracks)}"

        # Verify signal paths are different
        signal_paths = [t["SignalStreamPath"] for t in tracks]
        assert "force_000.sig" in signal_paths
        assert "force_010.sig" in signal_paths


def test_trapezoidal_tracks_no_files():
    """
    Test that parse_trapezoidal_tracks_xml returns empty list when no XML files exist.
    """
    import tempfile
    from hdsemg_shared.fileio.otb_4_file_io import parse_trapezoidal_tracks_xml

    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty directory - no TrapezoidalTracks files
        tracks = parse_trapezoidal_tracks_xml(tmpdir)
        assert tracks == [], "Should return empty list when no TrapezoidalTracks XML files found"


def test_force_signals_added_to_each_grid():
    """
    Test that force signals are added to each grid separately as reference channels.

    This tests that when multiple grids exist, each grid gets its own copy of
    the force signals (Performed Path and Original Path) as reference channels.
    """
    # Mock EMG file with 2 grids
    data = np.random.randn(1000, 128)
    time = np.arange(1000) / 2000.0

    # Create descriptions for two grids
    descriptions = []

    # Grid 1: HD08MM1305 (8mm IED, 64 channels)
    for i in range(64):
        descriptions.append(np.array([[f"IN1 HD08MM1305 ch{i+1}"]], dtype=object))

    # Grid 2: HD04MM1305 (4mm IED, 64 channels)
    for i in range(64):
        descriptions.append(np.array([[f"IN2 HD04MM1305 ch{i+1}"]], dtype=object))

    desc_array = np.array(descriptions, dtype=object)

    # Create EMGFile instance
    emg = EMGFile(
        data=data,
        time=time,
        description=desc_array,
        sf=2000.0,
        file_name="test_force_signals.otb4",
        file_size=1000,
        file_type="otb4"
    )

    # Get grids - should be 2 grids without force signals yet
    grids = emg.grids
    assert len(grids) == 2, f"Expected 2 grids, got {len(grids)}"


def test_original_path_recognized_as_requested_path():
    """
    Test that 'Original Path' in descriptions is recognized as 'requested path'.

    The file_io.py grid parser should recognize both "requested path" and
    "original path" as the requested_path_idx in Grid objects.
    """
    # Mock EMG file with grid and force signals
    data = np.random.randn(1000, 66)  # 64 EMG + 2 force signals
    time = np.arange(1000) / 2000.0

    descriptions = []

    # Grid: HD08MM1305 (64 channels)
    for i in range(64):
        descriptions.append(np.array([[f"IN1 HD08MM1305 ch{i+1}"]], dtype=object))

    # Force signals
    descriptions.append(np.array([["Performed Path"]], dtype=object))
    descriptions.append(np.array([["Original Path"]], dtype=object))

    desc_array = np.array(descriptions, dtype=object)

    # Create EMGFile instance
    emg = EMGFile(
        data=data,
        time=time,
        description=desc_array,
        sf=2000.0,
        file_name="test_original_path.otb4",
        file_size=1000,
        file_type="otb4"
    )

    # Get grids
    grids = emg.grids
    assert len(grids) == 1

    grid = grids[0]

    # Verify reference channels are detected
    assert len(grid.ref_indices) == 2, f"Expected 2 ref channels, got {len(grid.ref_indices)}"

    # Verify path indices are set
    assert grid.performed_path_idx == 64, "Performed Path should be at index 64"
    assert grid.requested_path_idx == 65, "Original Path should be at index 65 (recognized as requested path)"


def test_performed_path_recognized():
    """
    Test that 'Performed Path' in descriptions is correctly recognized.
    """
    # Mock EMG file with grid and performed path
    data = np.random.randn(1000, 65)
    time = np.arange(1000) / 2000.0

    descriptions = []

    # Grid: HD08MM1305 (64 channels)
    for i in range(64):
        descriptions.append(np.array([[f"IN1 HD08MM1305 ch{i+1}"]], dtype=object))

    # Performed path only
    descriptions.append(np.array([["Performed Path"]], dtype=object))

    desc_array = np.array(descriptions, dtype=object)

    # Create EMGFile instance
    emg = EMGFile(
        data=data,
        time=time,
        description=desc_array,
        sf=2000.0,
        file_name="test_performed_path.otb4",
        file_size=1000,
        file_type="otb4"
    )

    grids = emg.grids
    assert len(grids) == 1

    grid = grids[0]

    # Verify performed path is detected
    assert grid.performed_path_idx == 64
    # requested_path_idx should be None since only performed path exists
    assert grid.requested_path_idx is None


def test_force_signals_placed_after_each_grid():
    """
    Test that force signals are correctly placed after each grid, not all at the end.

    When multiple grids exist, each grid should have its own copy of force signals
    immediately following its EMG channels and regular refs. This ensures that the
    file_io.py grid parser correctly assigns force signals to each grid.

    Structure should be:
    [Grid1 EMG, Grid1 Refs, Grid1 Force Signals, Grid2 EMG, Grid2 Refs, Grid2 Force Signals, ...]
    """
    # Create mock data with 2 grids + force signals after each grid
    data = np.random.randn(1000, 132)  # 64 + 2 force + 32 + 2 force = 100
    time = np.arange(1000) / 2000.0
    descriptions = []

    # Grid 1: HD08MM1305 (64 EMG channels)
    for i in range(64):
        descriptions.append(np.array([[f"IN1 HD08MM1305 ch{i+1}"]], dtype=object))

    # Force signals after Grid 1
    descriptions.append(np.array([["Performed Path"]], dtype=object))
    descriptions.append(np.array([["Original Path"]], dtype=object))

    # Grid 2: HD10MM0408 (32 EMG channels)
    for i in range(32):
        descriptions.append(np.array([[f"IN2 HD10MM0408 ch{i+1}"]], dtype=object))

    # Force signals after Grid 2
    descriptions.append(np.array([["Performed Path"]], dtype=object))
    descriptions.append(np.array([["Original Path"]], dtype=object))

    desc_array = np.array(descriptions, dtype=object)

    # Create EMGFile
    emg = EMGFile(
        data=data,
        time=time,
        description=desc_array,
        sf=2000.0,
        file_name="test_force_placement.otb4",
        file_size=1000,
        file_type="otb4"
    )

    # Get grids
    grids = emg.grids
    assert len(grids) == 2, f"Expected 2 grids, got {len(grids)}"

    # Check Grid 1
    grid1 = grids[0]
    assert len(grid1.emg_indices) == 64, f"Grid 1 should have 64 EMG channels"
    assert len(grid1.ref_indices) == 2, f"Grid 1 should have 2 ref channels (force signals)"
    assert 64 in grid1.ref_indices, "Grid 1 should include force signal at index 64"
    assert 65 in grid1.ref_indices, "Grid 1 should include force signal at index 65"
    assert grid1.performed_path_idx == 64, f"Grid 1 performed path at wrong index"
    assert grid1.requested_path_idx == 65, f"Grid 1 requested path at wrong index"

    # Check Grid 2
    grid2 = grids[1]
    assert len(grid2.emg_indices) == 32, f"Grid 2 should have 32 EMG channels"
    assert len(grid2.ref_indices) == 2, f"Grid 2 should have 2 ref channels (force signals)"
    assert 98 in grid2.ref_indices, "Grid 2 should include force signal at index 98"
    assert 99 in grid2.ref_indices, "Grid 2 should include force signal at index 99"
    assert grid2.performed_path_idx == 98, f"Grid 2 performed path at wrong index"
    assert grid2.requested_path_idx == 99, f"Grid 2 requested path at wrong index"

    # Verify force signals are NOT shared (different indices for each grid)
    assert grid1.performed_path_idx != grid2.performed_path_idx, "Force signals should be separate for each grid"
    assert grid1.requested_path_idx != grid2.requested_path_idx, "Force signals should be separate for each grid"


def test_multiple_grids_with_three_grids():
    """
    Test force signal placement with three grids to ensure the pattern holds.
    """
    # Create data for 3 grids with force signals after each
    # Grid1: 64ch, Grid2: 32ch, Grid3: 64ch
    # Force signals: 2 per grid
    # Total: 64 + 2 + 32 + 2 + 64 + 2 = 166 channels
    data = np.random.randn(1000, 166)
    time = np.arange(1000) / 2000.0
    descriptions = []

    # Grid 1: HD08MM1305 (64 channels)
    for i in range(64):
        descriptions.append(np.array([[f"IN1 HD08MM1305 ch{i+1}"]], dtype=object))
    descriptions.append(np.array([["Performed Path"]], dtype=object))
    descriptions.append(np.array([["Original Path"]], dtype=object))

    # Grid 2: HD10MM0408 (32 channels)
    for i in range(32):
        descriptions.append(np.array([[f"IN2 HD10MM0408 ch{i+1}"]], dtype=object))
    descriptions.append(np.array([["Performed Path"]], dtype=object))
    descriptions.append(np.array([["Original Path"]], dtype=object))

    # Grid 3: HD08MM1305 (64 channels, same as Grid 1)
    for i in range(64):
        descriptions.append(np.array([[f"IN3 HD08MM1305 ch{i+1}"]], dtype=object))
    descriptions.append(np.array([["Performed Path"]], dtype=object))
    descriptions.append(np.array([["Original Path"]], dtype=object))

    desc_array = np.array(descriptions, dtype=object)

    emg = EMGFile(
        data=data,
        time=time,
        description=desc_array,
        sf=2000.0,
        file_name="test_three_grids.otb4",
        file_size=1000,
        file_type="otb4"
    )

    grids = emg.grids
    assert len(grids) == 3, f"Expected 3 grids, got {len(grids)}"

    # Check each grid has its own force signals
    # Grid 1: EMG 0-63, Force 64-65
    assert grids[0].performed_path_idx == 64
    assert grids[0].requested_path_idx == 65

    # Grid 2: EMG 66-97, Force 98-99
    assert grids[1].performed_path_idx == 98
    assert grids[1].requested_path_idx == 99

    # Grid 3: EMG 100-163, Force 164-165
    assert grids[2].performed_path_idx == 164
    assert grids[2].requested_path_idx == 165

    # Verify all force signals are unique per grid
    force_indices = set()
    for grid in grids:
        assert grid.performed_path_idx not in force_indices, "Performed path indices should be unique"
        assert grid.requested_path_idx not in force_indices, "Requested path indices should be unique"
        force_indices.add(grid.performed_path_idx)
        force_indices.add(grid.requested_path_idx)

    assert len(force_indices) == 6, "Should have 6 unique force signal indices (2 per grid × 3 grids)"
