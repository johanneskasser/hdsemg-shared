import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

import xml.etree.ElementTree as ET


def load_otb4_file(file_path):
    """
    Load and process an OTB4 file.

    This function handles the extraction of the OTB4 file, validates its contents,
    parses the associated XML file, and processes signal data. It returns the
    processed data, time array, descriptions, sampling frequency, file name, and
    file size.

    Args:
        file_path (str): Path to the OTB4 file to be loaded.

    Returns:
        tuple: A tuple containing:
            - data (numpy.ndarray): Processed signal data.
            - time (numpy.ndarray): Time array corresponding to the data.
            - description_array (numpy.ndarray): Array of channel descriptions.
            - sampling_frequency (float): Sampling frequency of the signals.
            - file_name (str): Name of the OTB4 file.
            - file_size (int): Size of the OTB4 file in bytes.

    Raises:
        ValueError: If the OTB4 file format is unrecognized or if no track info is found.
        FileNotFoundError: If the required XML file is missing.
        RuntimeError: If the extraction of the OTB4 file fails.
    """
    logger.debug(f"load_otb4_file called with file_path={file_path}")
    file_path_obj = Path(file_path)
    file_name = file_path_obj.name
    file_size = os.path.getsize(file_path)

    tmpdir = tempfile.mkdtemp(prefix="otb4_tmp_")
    logger.debug(f"Created temp directory: {tmpdir}")

    # Attempt to extract .otb4 (which presumably is a tar or zip)
    try:
        if tarfile.is_tarfile(file_path):
            with tarfile.open(file_path, 'r') as t:
                t.extractall(path=tmpdir)
        elif zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as z:
                z.extractall(path=tmpdir)
        else:
            logger.warning("OTB4 file is neither recognized as tar nor zip.")
            raise ValueError("Unrecognized OTB4 archive format.")
    except Exception as exc:
        logger.error(f"Extraction failed for {file_path}: {exc}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"Failed to extract OTB4 file: {file_path}") from exc

    # Look for "Tracks_000.xml" - as in your .m code
    tracks_xml = os.path.join(tmpdir, "Tracks_000.xml")
    if not os.path.exists(tracks_xml):
        logger.warning(f"Missing 'Tracks_000.xml' in extracted OTB4 folder: {tmpdir}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise FileNotFoundError("No Tracks_000.xml found for OTB4 data.")

    logger.debug(f"Parsing OTB4 XML: {tracks_xml}")
    track_info_list = parse_otb4_tracks_xml(tracks_xml)  # We'll define below

    # Parse TrapezoidalTracks XML files for force signals
    trapezoidal_tracks = parse_trapezoidal_tracks_xml(tmpdir)
    logger.debug(f"Found {len(trapezoidal_tracks)} trapezoidal track files with force signals")

    # We'll also list .sig files in tmpdir
    signals = []
    for root, dirs, files in os.walk(tmpdir):
        for f in files:
            if f.lower().endswith(".sig"):
                signals.append(os.path.join(root, f))

    logger.debug(f"Found .sig files for OTB4: {signals}")

    # Now, let's gather info from track_info_list. e.g. sum up channels, detect device type, etc.
    # The .m code references `device = textscan(...)` from the first track. We'll replicate that:
    if len(track_info_list) == 0:
        logger.error("No <TrackInfo> found in Tracks_000.xml. Can't proceed.")
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise ValueError("No track info in OTB4 xml.")
    device = track_info_list[0]["Device"]
    logger.debug(f"OTB4 device name: {device}")

    # Sum total channels from track_info
    total_channels = sum(tr["NumberOfChannels"] for tr in track_info_list)
    logger.debug(f"Total channels from tracks: {total_channels}")

    # Log grid information summary
    grids_with_info = [tr for tr in track_info_list if tr.get("GridInfo")]
    logger.debug(f"Tracks with GridInfo: {len(grids_with_info)}/{len(track_info_list)}")
    for tr in grids_with_info:
        gi = tr["GridInfo"]
        logger.debug(f"  {tr['Device']} - {tr['SubTitle']}: {gi['Name']} ({gi['NRow']}x{gi['NColumn']}, IED={gi['IED']}mm)")

    if device == "Novecento+":
        logger.debug("Using Novecento+ reader")
        data, descriptions, fs_main = read_novecento_plus(signals, track_info_list, trapezoidal_tracks, tmpdir)
    else:
        logger.debug("Using standard OTB4 reader")
        data, descriptions, fs_main = read_standard_otb4(signals, track_info_list, trapezoidal_tracks, tmpdir)

    description_array = np.array(
        [[np.array([desc], dtype=f'<U{len(desc)}')] for desc in descriptions],
        dtype=object
    )

    sampling_frequency = fs_main

    n_samples = data.shape[1]
    time = np.arange(n_samples) / sampling_frequency

    # clean up
    shutil.rmtree(tmpdir, ignore_errors=True)
    logger.info(f"OTB4 file loaded successfully: {file_name}")

    # robust shape match
    if data.shape[1] == len(time):
        pass  # already aligned
    elif data.shape[0] == len(time):
        logger.debug("Transposing data to align time dimension")
        data = data.T
    else:
        raise ValueError(f"Could not align data ({data.shape}) with time ({time.shape})")

    return data, time, description_array, sampling_frequency, file_name, file_size


def parse_otb4_tracks_xml(xml_file):
    """
    Parse 'Tracks_000.xml' to gather arrayOfTrackInfo.
    Returns a list of dictionaries, z. B.:
      [{
          "Device": "Novecento+",
          "Gain": ...,
          "ADC_Nbits": ...,
          "ADC_Range": ...,
          "SamplingFrequency": ...,
          "SignalStreamPath": "...",
          "NumberOfChannels": ...,
          "AcquisitionChannel": ...,
          "SubTitle": ...   # <-- Grid identifier
          "GridInfo": {      # <-- Grid metadata from Description element
              "Name": "HD08MM1305",
              "NRow": 5,
              "NColumn": 13,
              "IED": 8
          } or None
        }, ... ]
    """
    logger.debug(f"Parsing OTB4 XML file: {xml_file}")
    tree = ET.parse(xml_file)
    track_info_parent = tree.getroot()

    if track_info_parent is None:
        logger.warning("XML root is None, returning empty track list")
        return []

    # Finde die <TrackInfo>-Elemente
    track_info_elems = track_info_parent.findall("TrackInfo")
    logger.debug(f"Found {len(track_info_elems)} TrackInfo elements in XML")
    results = []
    for idx, tr_el in enumerate(track_info_elems):
        logger.debug(f"Processing TrackInfo #{idx}")
        device_str = _get_text_save(tr_el, "Device", default="Unknown", do_strip= True)
        gain_str = _get_text_save(tr_el, "Gain", default="1", do_strip= False)
        bits_str = _get_text_save(tr_el, "ADC_Nbits", default="16", do_strip= False)
        rng_str = _get_text_save(tr_el, "ADC_Range", default="5", do_strip= False)
        fs_str = _get_text_save(tr_el, "SamplingFrequency", default="2000", do_strip= False)
        path_str = _get_text_save(tr_el, "SignalStreamPath", default="", do_strip= True)
        nchan_str = _get_text_save(tr_el, "NumberOfChannels", default="0", do_strip= False)
        acq_str = _get_text_save(tr_el, "AcquisitionChannel", default="0", do_strip= False)
        subtitle_str = _get_text_save(tr_el, "SubTitle", default="Unknown", do_strip= True)
        is_control_str = _get_text_save(tr_el, "IsControl", default="false", do_strip= True)
        chan_offset_str = _get_text_save(tr_el, "ChannelOffsetInSubPacket", default="0", do_strip= False)
        muscle_str = _get_text_save(tr_el, "Muscle", default="", do_strip= True)

        # Konvertierungen
        gain_val = float(gain_str)
        bits_val = int(bits_str)
        rng_val = float(rng_str)
        fs_val = float(fs_str)
        nchan_val = int(nchan_str)
        acq_val = int(acq_str)
        is_control_val = is_control_str.lower() == "true"
        chan_offset_val = int(chan_offset_str)

        logger.debug(f"  Device: {device_str}, SubTitle: {subtitle_str}, Channels: {nchan_val}, Path: {path_str}, IsControl: {is_control_val}, Offset: {chan_offset_val}, Muscle: {muscle_str}")

        # Extract grid information from Description element if present
        grid_info = None
        desc_el = tr_el.find("Description")
        if desc_el is not None:
            logger.debug(f"  Found Description element for TrackInfo #{idx}")
            grid_name = _get_text_save(desc_el, "Name", default="", do_strip=True)
            nrow_str = _get_text_save(desc_el, "NRow", default="", do_strip=False)
            ncol_str = _get_text_save(desc_el, "NColumn", default="", do_strip=False)
            ied_str = _get_text_save(desc_el, "IED", default="", do_strip=False)

            logger.debug(f"    Name: {grid_name!r}, NRow: {nrow_str!r}, NColumn: {ncol_str!r}, IED: {ied_str!r}")

            # Only create grid_info if we have valid grid data
            if grid_name and nrow_str and ncol_str and ied_str:
                try:
                    grid_info = {
                        "Name": grid_name,
                        "NRow": int(nrow_str),
                        "NColumn": int(ncol_str),
                        "IED": int(ied_str)
                    }
                    logger.debug(f"    Created GridInfo: {grid_info}")
                except ValueError as e:
                    logger.warning(f"    Failed to parse grid dimensions: {e}")
                    grid_info = None
            else:
                logger.debug(f"    Incomplete grid data, skipping GridInfo creation")
        else:
            logger.debug(f"  No Description element found for TrackInfo #{idx}")

        track_dict = {
            "Device": device_str,
            "Gain": gain_val,
            "ADC_Nbits": bits_val,
            "ADC_Range": rng_val,
            "SamplingFrequency": fs_val,
            "SignalStreamPath": path_str,
            "NumberOfChannels": nchan_val,
            "AcquisitionChannel": acq_val,
            "SubTitle": subtitle_str,  # Grid identifier
            "GridInfo": grid_info,     # Grid metadata
            "IsControl": is_control_val,  # Whether this is a control/reference signal
            "ChannelOffset": chan_offset_val,  # Offset within signal file
            "Muscle": muscle_str if muscle_str else None  # Muscle name
        }
        results.append(track_dict)
        logger.debug(f"  Added track to results list (total: {len(results)})")

    logger.info(f"Parsed {len(results)} tracks from XML, {sum(1 for r in results if r['GridInfo'] is not None)} with grid info")
    return results


def parse_trapezoidal_tracks_xml(tmpdir):
    """
    Parse TrapezoidalTracks_*.xml files to extract force signal information.

    These files contain "Performed Path" and "Original Path" force signals that should
    be added as reference signals to all grids.

    Args:
        tmpdir: Directory containing extracted OTB4 files

    Returns:
        list: List of dictionaries containing force signal track information:
            [{
                "SubTitle": "Performed Path" or "Original Path",
                "SignalStreamPath": "filename.sig",
                "SamplingFrequency": float,
                "Gain": float,
                "ADC_Nbits": int,
                "ADC_Range": float,
                "AcquisitionChannel": int,
                "NumberOfChannels": int (always 1 for force signals)
            }, ...]
    """
    logger.debug(f"Searching for TrapezoidalTracks XML files in {tmpdir}")
    trapezoidal_tracks = []

    # Find all TrapezoidalTracks_*.xml files
    for root, dirs, files in os.walk(tmpdir):
        for f in files:
            if f.startswith("TrapezoidalTracks_") and f.endswith(".xml"):
                xml_path = os.path.join(root, f)
                logger.debug(f"Parsing trapezoidal tracks XML: {xml_path}")

                try:
                    tree = ET.parse(xml_path)
                    track_info_parent = tree.getroot()

                    if track_info_parent is None:
                        logger.warning(f"XML root is None in {xml_path}")
                        continue

                    # Find all TrackInfo elements
                    track_info_elems = track_info_parent.findall("TrackInfo")
                    logger.debug(f"Found {len(track_info_elems)} TrackInfo elements in {f}")

                    for idx, tr_el in enumerate(track_info_elems):
                        subtitle_str = _get_text_save(tr_el, "SubTitle", default="Unknown", do_strip=True)
                        path_str = _get_text_save(tr_el, "SignalStreamPath", default="", do_strip=True)
                        fs_str = _get_text_save(tr_el, "SamplingFrequency", default="10", do_strip=False)
                        gain_str = _get_text_save(tr_el, "Gain", default="0.5", do_strip=False)
                        bits_str = _get_text_save(tr_el, "ADC_Nbits", default="1", do_strip=False)
                        rng_str = _get_text_save(tr_el, "ADC_Range", default="1", do_strip=False)
                        acq_str = _get_text_save(tr_el, "AcquisitionChannel", default="0", do_strip=False)

                        # Convert to appropriate types
                        fs_val = float(fs_str)
                        gain_val = float(gain_str)
                        bits_val = int(bits_str)
                        rng_val = float(rng_str)
                        acq_val = int(acq_str)

                        track_dict = {
                            "SubTitle": subtitle_str,
                            "SignalStreamPath": path_str,
                            "SamplingFrequency": fs_val,
                            "Gain": gain_val,
                            "ADC_Nbits": bits_val,
                            "ADC_Range": rng_val,
                            "AcquisitionChannel": acq_val,
                            "NumberOfChannels": 1  # Force signals are always 1 channel
                        }

                        trapezoidal_tracks.append(track_dict)
                        logger.debug(f"  Added force signal: {subtitle_str} from {path_str} (channel {acq_val})")

                except Exception as e:
                    logger.warning(f"Failed to parse {xml_path}: {e}")
                    continue

    logger.info(f"Parsed {len(trapezoidal_tracks)} force signal tracks from TrapezoidalTracks XML files")
    return trapezoidal_tracks


def _get_text_save(parent, tag, default="", do_strip=True, requred_none_null=False):
    """
       Find the first child <tag> under `parent`, return its .text (stripped if requested),
       or return `default` if either the element is missing or its .text is None.

       parent     : an Element (e.g. tr_el)
       tag        : string name of the sub‐element to find
       default    : fallback string if no element or element.text is None
       do_strip   : whether to apply .strip() to the result
       """
    el = parent.find(tag)
    if el is None and requred_none_null:
        logger.error(f"Required tag '{tag}' not found in {parent.tag}.")
        raise ValueError(f"Required tag '{tag}' not found in {parent.tag}.")
    raw = el.text if (el is not None and el.text is not None) else default
    return raw.strip() if do_strip else raw



def read_novecento_plus(signals, track_info_list, trapezoidal_tracks, tmpdir):
    """
    For the 'Novecento+' device, reads data from signal files using int32 format.
    Handles multiple tracks per signal file using ChannelOffset to extract correct channels.
    Also processes force signals from TrapezoidalTracks XML and adds them to all grids.
    Returns final (data, descriptions, fs).
    """
    logger.debug("read_novecento_plus routine")
    logger.debug(f"Processing {len(signals)} signal files with {len(track_info_list)} tracks")
    logger.debug(f"Processing {len(trapezoidal_tracks)} force signal tracks")

    # First pass: identify valid EMG grids (not control signals)
    # We'll use the last valid EMG grid pattern for control signals
    last_valid_grid_pattern = None
    for tr in track_info_list:
        grid_info = tr.get("GridInfo")
        is_control = tr.get("IsControl", False)

        # Valid EMG grid: has GridInfo, not a control signal, IED > 1, and > 4 electrodes
        if grid_info and not is_control:
            ied = grid_info.get("IED", 1)
            rows = grid_info.get("NRow", 1)
            cols = grid_info.get("NColumn", 1)
            electrodes = rows * cols

            # Only use grids with IED > 1 or many electrodes (real EMG grids)
            if ied > 1 or electrodes > 4:
                last_valid_grid_pattern = f"HD{ied:02d}MM{rows:02d}{cols:02d}"
                logger.debug(f"Found valid EMG grid pattern: {last_valid_grid_pattern}")

    # We'll create a big list of channel blocks
    data_blocks = []
    descriptions = []
    fs_main = None

    for sig_idx, sig_path in enumerate(signals):
        logger.debug(f"Processing signal file #{sig_idx}: {os.path.basename(sig_path)}")

        # Find ALL tracks that reference this signal file
        matched_tracks = []
        for tr_idx, tr in enumerate(track_info_list):
            if tr["SignalStreamPath"] == os.path.basename(sig_path):
                matched_tracks.append((tr_idx, tr))

        if not matched_tracks:
            logger.warning(f"  No matching tracks found for signal file {os.path.basename(sig_path)}, skipping")
            continue

        logger.debug(f"  Found {len(matched_tracks)} tracks using this signal file")

        # Calculate total channels in this file to determine file size
        total_channels = sum(tr["NumberOfChannels"] for _, tr in matched_tracks)
        logger.debug(f"  Total channels in file: {total_channels}")

        # Read entire signal file as int32
        raw = np.fromfile(sig_path, dtype=np.int32)
        samples = raw.size // total_channels
        if raw.size % total_channels != 0:
            logger.warning(f"  File size mismatch: {raw.size} values not divisible by {total_channels} channels")
            # Trim to nearest complete sample
            raw = raw[:samples * total_channels]

        logger.debug(f"  Read {raw.size} int32 values, {samples} samples × {total_channels} channels")

        # Reshape to (total_channels, samples) using Fortran order
        all_data = raw.reshape((total_channels, samples), order='F')
        logger.debug(f"  Reshaped data: {all_data.shape} (channels × samples)")

        # Process each track that uses this signal file
        for tr_idx, matched_track in matched_tracks:
            n_ch = matched_track["NumberOfChannels"]
            offset = matched_track["ChannelOffset"]
            is_control = matched_track["IsControl"]

            logger.debug(f"  Processing track #{tr_idx}: {matched_track['Device']} - {matched_track['SubTitle']}")
            logger.debug(f"    Channels: {n_ch}, Offset: {offset}, IsControl: {is_control}")

            # Extract channels for this track using offset
            if offset + n_ch > total_channels:
                logger.error(f"    Channel offset {offset} + {n_ch} exceeds total channels {total_channels}")
                continue

            block = all_data[offset:offset + n_ch, :]
            logger.debug(f"    Extracted block shape: {block.shape}")

            # Apply gain/scaling conversion
            conv = matched_track["ADC_Range"] / (2 ** matched_track["ADC_Nbits"]) * 1000 / matched_track["Gain"]
            block = block.astype(np.float64) * conv
            logger.debug(f"    Applied conversion factor: {conv:.6f}")

            data_blocks.append(block)

            # Track sampling frequency
            if fs_main is None:
                fs_main = matched_track["SamplingFrequency"]
                logger.debug(f"    Set sampling frequency: {fs_main} Hz")
            else:
                if abs(fs_main - matched_track["SamplingFrequency"]) > 1e-9:
                    logger.warning(f"    Inconsistent sampling freq: expected {fs_main}, got {matched_track['SamplingFrequency']}")

            # Build descriptions with grid information
            grid_info = matched_track.get("GridInfo")
            if grid_info:
                logger.debug(f"    Using GridInfo: {grid_info['Name']} ({grid_info['NRow']}x{grid_info['NColumn']}, IED={grid_info['IED']}mm)")
            else:
                logger.debug(f"    No GridInfo, using fallback description format")

            # Get muscle information for this track
            muscle = matched_track.get("Muscle")

            for c in range(n_ch):
                grid_id = matched_track.get("SubTitle", "Unknown")

                # Determine which grid pattern to use
                # IMPORTANT: REF channels should NOT have a grid pattern in their description
                # so that file_io.py's grid parser adds them to refs instead of indices

                if is_control:
                    # Control signals: NO grid pattern, just REF marker
                    desc = f"{grid_id} REF ch{c+1}"
                elif grid_info:
                    # Check if this is a valid EMG grid or a control grid
                    ied = grid_info["IED"]
                    rows = grid_info["NRow"]
                    cols = grid_info["NColumn"]
                    electrodes = rows * cols

                    if ied == 1 and electrodes <= 4:
                        # Small control grid (quaternions, etc): NO pattern, just REF
                        desc = f"{grid_id} REF ch{c+1}"
                    else:
                        # Valid EMG grid: use its own pattern (no REF marker)
                        grid_pattern = f"HD{ied:02d}MM{rows:02d}{cols:02d}"
                        desc = f"{grid_id} {grid_pattern} ch{c+1}"
                        # Add muscle information if available
                        if muscle:
                            desc += f" [MUSCLE:{muscle}]"
                else:
                    # No grid info: just description without pattern
                    desc = f"{grid_id} ch{c+1}"

                descriptions.append(desc)

            logger.debug(f"    Generated {n_ch} descriptions (total so far: {len(descriptions)})")

    # Process force signals from TrapezoidalTracks XML files
    force_signal_data = []
    force_signal_descriptions = []

    if trapezoidal_tracks:
        logger.debug(f"Processing {len(trapezoidal_tracks)} force signal tracks")

        # Group force signals by their signal file
        force_signals_by_file = {}
        for ft in trapezoidal_tracks:
            sig_file = ft["SignalStreamPath"]
            if sig_file not in force_signals_by_file:
                force_signals_by_file[sig_file] = []
            force_signals_by_file[sig_file].append(ft)

        # Process each force signal file
        for sig_file, force_tracks in force_signals_by_file.items():
            sig_path = os.path.join(tmpdir, sig_file)

            if not os.path.exists(sig_path):
                logger.warning(f"  Force signal file not found: {sig_file}")
                continue

            logger.debug(f"  Reading force signal file: {sig_file}")

            # Calculate total channels in this force signal file
            total_force_channels = sum(ft["NumberOfChannels"] for ft in force_tracks)

            # Read force signal file as float64 (8 bytes per sample)
            raw_force = np.fromfile(sig_path, dtype=np.float64)
            force_samples = raw_force.size // total_force_channels

            if raw_force.size % total_force_channels != 0:
                logger.warning(f"    Force file size mismatch: {raw_force.size} values not divisible by {total_force_channels} channels")
                raw_force = raw_force[:force_samples * total_force_channels]

            logger.debug(f"    Read {raw_force.size} float64 values, {force_samples} samples × {total_force_channels} channels")

            # Reshape to (total_channels, samples) using Fortran order
            force_data = raw_force.reshape((total_force_channels, force_samples), order='F')
            logger.debug(f"    Reshaped force data: {force_data.shape}")

            # Extract each force track
            for ft in force_tracks:
                acq_channel = ft["AcquisitionChannel"]
                subtitle = ft["SubTitle"]

                if acq_channel >= total_force_channels:
                    logger.warning(f"    Force signal channel {acq_channel} out of range (max {total_force_channels-1})")
                    continue

                # Extract the specific channel
                channel_data = force_data[acq_channel:acq_channel+1, :]
                logger.debug(f"    Extracted force channel {acq_channel}: {subtitle}, shape: {channel_data.shape}")

                # Apply gain/scaling conversion
                conv = ft["ADC_Range"] / (2 ** ft["ADC_Nbits"]) * 1000 / ft["Gain"]
                channel_data = channel_data.astype(np.float64) * conv

                force_signal_data.append(channel_data)
                force_signal_descriptions.append(f"{subtitle}")

                logger.debug(f"    Added force signal: {subtitle}")

    # Combine EMG data and force signals
    all_data_blocks = data_blocks.copy()
    all_descriptions = descriptions.copy()

    # Calculate target sample count from EMG data
    target_samples = data_blocks[0].shape[1] if data_blocks else 0

    # Resample force signals to match EMG sampling rate
    if force_signal_data and target_samples > 0:
        logger.debug(f"Resampling {len(force_signal_data)} force signals to match EMG data")

        from scipy.interpolate import interp1d

        for idx, force_block in enumerate(force_signal_data):
            force_samples = force_block.shape[1]

            if force_samples != target_samples:
                logger.debug(f"  Resampling force signal {idx}: {force_samples} -> {target_samples} samples")

                # Create interpolation function
                old_time = np.linspace(0, 1, force_samples)
                new_time = np.linspace(0, 1, target_samples)

                resampled_data = np.zeros((1, target_samples))
                interpolator = interp1d(old_time, force_block[0, :], kind='linear', fill_value='extrapolate')
                resampled_data[0, :] = interpolator(new_time)

                force_signal_data[idx] = resampled_data
                logger.debug(f"    Resampled shape: {resampled_data.shape}")

        # Add force signals as shared reference signals for all grids
        # These should be added ONCE and will be referenced by all grids
        logger.debug(f"Adding {len(force_signal_data)} force signals as shared reference signals")

        for force_idx, force_data_block in enumerate(force_signal_data):
            all_data_blocks.append(force_data_block)
            force_desc = force_signal_descriptions[force_idx]
            all_descriptions.append(force_desc)
            logger.debug(f"  Added force signal: '{force_desc}'")

    lens = [b.shape[1] for b in all_data_blocks]
    if not all(l == lens[0] for l in lens):
        # mismatch sample counts => handle or raise
        logger.warning(f"Different sample lengths among signals: {lens}. Picking min length.")
        min_len = min(lens)
        for i in range(len(all_data_blocks)):
            all_data_blocks[i] = all_data_blocks[i][:, :min_len]
        logger.debug(f"Trimmed all blocks to {min_len} samples")

    # vstack them
    final_data = np.vstack(all_data_blocks) if all_data_blocks else np.zeros((0, 0))
    logger.info(f"read_novecento_plus: Final data shape: {final_data.shape}, {len(all_descriptions)} descriptions, fs={fs_main or 2000} Hz")
    return final_data, all_descriptions, fs_main or 2000


def read_standard_otb4(signals, track_info_list, trapezoidal_tracks, tmpdir):
    """
    We read the *first* .sig using 'short' if so indicated,
    then apply Gains for each track in partial channel intervals.
    Also processes force signals from TrapezoidalTracks XML and adds them to all grids.
    """
    logger.debug("read_standard_otb4 routine")
    logger.debug(f"Processing {len(signals)} signal files with {len(track_info_list)} tracks")
    logger.debug(f"Processing {len(trapezoidal_tracks)} force signal tracks")

    if not signals:
        logger.warning("No signal files provided, returning empty data")
        return np.zeros((0, 0)), [], 2000.0

    sig_path = signals[0]  # the .m code uses the first .sig
    logger.debug(f"Reading from signal file: {os.path.basename(sig_path)}")

    # sum up totalCh
    totalCh = sum(tr["NumberOfChannels"] for tr in track_info_list)
    logger.debug(f"Total channels across all tracks: {totalCh}")

    # read 'short' => int16
    raw = np.fromfile(sig_path, dtype=np.int16)
    expected_channels = totalCh
    n_values = raw.size
    logger.debug(f"Read {n_values} int16 values from file")

    if n_values % expected_channels != 0:
        logger.error(f"Raw data length {n_values} is not divisible by {expected_channels} channels")
        raise ValueError(f"Invalid signal data length for {expected_channels} channels")

    samples = n_values // expected_channels
    data_2d = raw[:samples * expected_channels].reshape((expected_channels, samples), order='F').astype(np.float64)
    logger.debug(f"Reshaped data to: {data_2d.shape} (channels x samples)")

    indexes = []
    running_sum = 0
    for tr in track_info_list:
        running_sum += tr["NumberOfChannels"]
        indexes.append(running_sum)
    logger.debug(f"Channel index boundaries: {indexes}")

    start_idx = 0
    for i, trackinfo in enumerate(track_info_list):
        end_idx = indexes[i]
        psup = trackinfo["ADC_Range"]
        nbits = trackinfo["ADC_Nbits"]
        gainv = trackinfo["Gain"]
        conv = psup / (2 ** nbits) * 1000.0 / gainv
        logger.debug(f"Track #{i} ({trackinfo['Device']} - {trackinfo['SubTitle']}): channels {start_idx}:{end_idx}, conv={conv:.6f}")
        for ch in range(start_idx, end_idx):
            data_2d[ch, :] *= conv
        start_idx = end_idx

    # build description strings with grid information if available
    descriptions = []
    desc_idx = 0
    for tr_idx, tr in enumerate(track_info_list):
        dev = tr["Device"]
        path = tr["SignalStreamPath"]
        grid_id = tr.get("SubTitle", "Unknown")
        nchan = tr["NumberOfChannels"]
        grid_info = tr.get("GridInfo")
        muscle = tr.get("Muscle")

        if grid_info:
            logger.debug(f"Track #{tr_idx}: Using GridInfo for {nchan} channels: {grid_info['Name']} ({grid_info['NRow']}x{grid_info['NColumn']}, IED={grid_info['IED']}mm)")
        else:
            logger.debug(f"Track #{tr_idx}: No GridInfo, using fallback for {nchan} channels")

        for c in range(nchan):
            # If grid info is available, format description to match expected pattern
            if grid_info:
                # Format: HD{IED}MM{rows:02d}{cols:02d}
                ied = grid_info["IED"]
                rows = grid_info["NRow"]
                cols = grid_info["NColumn"]
                grid_pattern = f"HD{ied:02d}MM{rows:02d}{cols:02d}"
                desc = f"{dev} {grid_pattern} ch{c+1}"
                # Add muscle information if available
                if muscle:
                    desc += f" [MUSCLE:{muscle}]"
            else:
                desc = f"{dev}-{path}-{grid_id}-ch{c}"
            descriptions.append(desc)
            desc_idx += 1

        logger.debug(f"  Generated {nchan} descriptions (total so far: {desc_idx})")

    # Process force signals from TrapezoidalTracks XML files
    # Note: For standard OTB4, we currently don't have force signal processing implemented
    # as the typical use case is Novecento+. If needed in the future, similar logic to
    # read_novecento_plus can be added here.
    if trapezoidal_tracks:
        logger.warning(f"Found {len(trapezoidal_tracks)} trapezoidal tracks, but force signal processing is not yet implemented for standard OTB4 devices")

    # the .m code picks the sample freq from e.g. Fsample{nSig}, presumably from the first track
    fs_main = track_info_list[0]["SamplingFrequency"] if track_info_list else 2000.0

    logger.info(f"read_standard_otb4: Final data shape: {data_2d.shape}, {len(descriptions)} descriptions, fs={fs_main} Hz")
    return data_2d, descriptions, fs_main
