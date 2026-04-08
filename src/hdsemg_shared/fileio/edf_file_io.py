import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_ANNOTATION_LABEL = "EDF Annotations"


def load_edf_file(edf_path: str):
    """
    Load an EDF / EDF+C file together with its BIDS sidecar files.

    Sidecar lookup (all optional, same directory as the EDF):
      *_channels.tsv  – per-channel type, group, description, muscle
      *_emg.json      – grid geometry (EMGElectrodeGroups or model name)

    EDF+C annotation signals ("EDF Annotations") are detected and excluded
    from the returned data array automatically.

    Returns
    -------
    data : ndarray, shape (n_channels, n_samples), float64 – physical units
    time : ndarray, shape (n_samples,)
    description : list[str], length n_channels
    sampling_frequency : float
    file_name : str
    file_size : int – bytes
    """
    edf_path = Path(edf_path)
    file_name = edf_path.name
    file_size = os.path.getsize(edf_path)

    with open(edf_path, "rb") as fh:
        raw = fh.read()

    header = _parse_edf_header(raw)

    # EDF+C annotation signals are not numeric data – exclude them
    annotation_indices = {
        i for i, lbl in enumerate(header["labels"])
        if lbl == _ANNOTATION_LABEL
    }

    data = _read_edf_data(raw, header, annotation_indices)   # (n_data_ch, n_samples)

    sf = header["sf"]
    time = np.arange(data.shape[1]) / sf

    sidecar = _find_sidecar(edf_path)
    channels_tsv_path = _find_channels_tsv(edf_path)

    n_data_channels = header["n_signals"] - len(annotation_indices)
    description = _build_descriptions(n_data_channels, sidecar, channels_tsv_path)

    logger.info(
        "EDF loaded: %s | channels=%d | samples=%d | sf=%.1f Hz",
        file_name, data.shape[0], data.shape[1], sf,
    )
    return data, time, description, sf, file_name, file_size


# ---------------------------------------------------------------------------
# EDF header parsing
# ---------------------------------------------------------------------------

def _parse_edf_header(raw: bytes) -> dict:
    """
    Parse the fixed-width EDF / EDF+ header.

    General header (bytes 0-255):
      [  0:  8]  version
      [  8: 88]  patient info
      [ 88:168]  recording info
      [168:176]  startdate
      [176:184]  starttime
      [184:192]  number of header bytes
      [192:236]  reserved  (contains "EDF+C" or "EDF+D" for EDF+ files)
      [236:244]  number of data records
      [244:252]  duration of each record in seconds
      [252:256]  number of signals ns

    Signal header (bytes 256 … 256 + ns×256):
      label        16 bytes × ns
      transducer   80 bytes × ns
      phys_dim      8 bytes × ns
      phys_min      8 bytes × ns
      phys_max      8 bytes × ns
      dig_min       8 bytes × ns
      dig_max       8 bytes × ns
      prefilter    80 bytes × ns
      n_samps       8 bytes × ns
      reserved     32 bytes × ns
    """
    n_signals       = int(raw[252:256])
    n_records       = int(raw[236:244])
    record_duration = float(raw[244:252])
    header_bytes    = int(raw[184:192])
    ns = n_signals

    def _read_fields(start: int, width: int) -> list:
        return [
            raw[start + i * width: start + (i + 1) * width].decode("ascii").strip()
            for i in range(ns)
        ]

    base = 256
    labels     = _read_fields(base, 16);  base += ns * 16
    base      += ns * 80                  # skip transducers
    phys_dims  = _read_fields(base,  8);  base += ns *  8
    phys_mins  = _read_fields(base,  8);  base += ns *  8
    phys_maxs  = _read_fields(base,  8);  base += ns *  8
    dig_mins   = _read_fields(base,  8);  base += ns *  8
    dig_maxs   = _read_fields(base,  8);  base += ns *  8
    base      += ns * 80                  # skip prefilters
    n_samps    = _read_fields(base,  8);  base += ns *  8

    n_samps   = [int(x)   for x in n_samps]
    phys_mins = [float(x) for x in phys_mins]
    phys_maxs = [float(x) for x in phys_maxs]
    dig_mins  = [float(x) for x in dig_mins]
    dig_maxs  = [float(x) for x in dig_maxs]

    sf = n_samps[0] / record_duration if record_duration > 0 else float(n_samps[0])

    return {
        "n_signals":       ns,
        "n_records":       n_records,
        "record_duration": record_duration,
        "header_bytes":    header_bytes,
        "labels":          labels,
        "n_samps":         n_samps,
        "phys_mins":       phys_mins,
        "phys_maxs":       phys_maxs,
        "dig_mins":        dig_mins,
        "dig_maxs":        dig_maxs,
        "sf":              sf,
    }


# ---------------------------------------------------------------------------
# EDF data reading
# ---------------------------------------------------------------------------

def _read_edf_data(raw: bytes, header: dict, annotation_indices: set) -> np.ndarray:
    """
    Read all data records and return float64 array of shape (n_data_channels, n_samples).

    Signals whose index is in ``annotation_indices`` are skipped entirely —
    EDF+C annotation signals contain TAL-encoded text, not numeric samples.

    Scaling:  physical = gain × digital + offset
      gain   = (phys_max - phys_min) / (dig_max - dig_min)
      offset = phys_min - gain × dig_min
    """
    ns        = header["n_signals"]
    n_records = header["n_records"]
    n_samps   = header["n_samps"]
    offset    = header["header_bytes"]

    phys_mins = np.asarray(header["phys_mins"], dtype=np.float64)
    phys_maxs = np.asarray(header["phys_maxs"], dtype=np.float64)
    dig_mins  = np.asarray(header["dig_mins"],  dtype=np.float64)
    dig_maxs  = np.asarray(header["dig_maxs"],  dtype=np.float64)

    denom = dig_maxs - dig_mins
    denom = np.where(denom == 0, 1.0, denom)
    gains   = (phys_maxs - phys_mins) / denom
    offsets = phys_mins - gains * dig_mins

    data_indices = [i for i in range(ns) if i not in annotation_indices]
    total_per_ch = {i: n_samps[i] * n_records for i in data_indices}
    buffers    = {i: np.empty(total_per_ch[i], dtype=np.float64) for i in data_indices}
    write_ptrs = {i: 0 for i in data_indices}

    data_raw = np.frombuffer(raw[offset:], dtype="<i2")
    read_ptr = 0

    for _ in range(n_records):
        for sig_i in range(ns):
            n = n_samps[sig_i]
            if sig_i not in annotation_indices:
                chunk = data_raw[read_ptr: read_ptr + n]
                wp = write_ptrs[sig_i]
                buffers[sig_i][wp: wp + n] = chunk
                write_ptrs[sig_i] = wp + n
            read_ptr += n

    for sig_i in data_indices:
        buffers[sig_i] = buffers[sig_i] * gains[sig_i] + offsets[sig_i]

    return np.vstack([buffers[i] for i in data_indices])


# ---------------------------------------------------------------------------
# BIDS sidecar helpers
# ---------------------------------------------------------------------------

def _find_sidecar(edf_path: Path) -> Optional[dict]:
    """
    Find the BIDS *_emg.json sidecar (same base name, .edf → .json).
    Returns parsed dict or None.
    """
    sidecar_path = edf_path.with_suffix(".json")
    if not sidecar_path.exists():
        logger.warning("BIDS JSON sidecar not found for %s.", edf_path.name)
        return None
    with open(sidecar_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_channels_tsv(edf_path: Path) -> Optional[Path]:
    """
    Find the BIDS *_channels.tsv sidecar.
    Naming convention: replace ``_emg.edf`` with ``_channels.tsv``.
    """
    tsv_path = edf_path.with_name(
        edf_path.name[: -len("_emg.edf")] + "_channels.tsv"
        if edf_path.name.endswith("_emg.edf")
        else edf_path.stem + "_channels.tsv"
    )
    if tsv_path.exists():
        return tsv_path
    logger.debug("No _channels.tsv found for %s.", edf_path.name)
    return None


def _parse_channels_tsv(tsv_path: Path) -> list:
    """Return list of dicts, one per channel row, from a BIDS *_channels.tsv."""
    with open(tsv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [dict(row) for row in reader]


# ---------------------------------------------------------------------------
# Grid geometry helpers
# ---------------------------------------------------------------------------

def _parse_ied_mm(value) -> int:
    """Extract IED (mm) from '8 mm', '8mm', or numeric."""
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else 8


def _parse_model_name(model_name: str) -> Optional[tuple]:
    """
    Parse grid geometry from a manufacturer model name that encodes
    ied/rows/cols, e.g.:
      "GR04MM1305"  →  (4, 13, 5)   OTBioelettronica convention
      "HD08MM1032"  →  (8, 10, 32)  already in HD format

    Returns (ied_mm, rows, cols) or None if not parseable.
    """
    match = re.search(r"(\d{2})MM(\d{2})(\d{2})", str(model_name))
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def _extract_muscle(sidecar: dict, group_info: dict) -> Optional[str]:
    """
    Resolve muscle name for an electrode group.

    Priority:
    1. group_info["Muscle"]  — explicit per-group field
    2. sidecar["TaskName"]   — parsed for "Muscle: <name>"
    """
    muscle = group_info.get("Muscle")
    if muscle:
        return str(muscle).strip()

    task_name = sidecar.get("TaskName", "")
    match = re.search(r"(?i)muscle[:\s]+([A-Za-z0-9_]+)", task_name)
    if match:
        return match.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# Description builder
# ---------------------------------------------------------------------------

def _build_descriptions(
    n_channels: int,
    sidecar: Optional[dict],
    channels_tsv_path: Optional[Path],
) -> list:
    """
    Build a list of ``n_channels`` description strings.

    Strategy (first match wins):
    1. BIDS *_channels.tsv — most detailed; derives grid geometry from
       the sidecar model name or IED; uses the TSV ``description`` column
       directly for non-EMG channels (enabling "Requested Path" /
       "Performed Path" detection by ``EMGFile.grids``).
    2. ``EMGElectrodeGroups`` in the *_emg.json sidecar — used when the
       synthetic dataset provides electrode group info directly.
    3. Plain fallback: ``EDF_ch{i}`` for every channel.

    EMG channel descriptions encode grid geometry as
    ``HD{ied:02d}MM{rows:02d}{cols:02d}`` (e.g. ``HD04MM1305``),
    followed by the group name, channel index, and optional
    ``[MUSCLE:{name}]`` tag — matching the patterns expected by
    ``EMGFile.grids``.
    """
    fallback = [f"EDF_ch{i}" for i in range(n_channels)]

    # --- Strategy 1: channels.tsv ---
    if channels_tsv_path is not None:
        try:
            return _build_from_channels_tsv(
                n_channels, channels_tsv_path, sidecar or {}
            )
        except Exception as exc:
            logger.warning("Could not build descriptions from channels.tsv: %s", exc)

    if sidecar is None:
        return fallback

    # --- Strategy 2: EMGElectrodeGroups in JSON sidecar ---
    groups = sidecar.get("EMGElectrodeGroups", {})
    if not groups:
        return fallback

    descriptions = list(fallback)
    ch_idx = 0
    for group_name, info in groups.items():
        ied   = _parse_ied_mm(info.get("InterelectrodeDistance", 8))
        shape = info.get("GridShape", [])
        if len(shape) >= 2:
            rows, cols = int(shape[0]), int(shape[1])
        else:
            rows, cols = 1, n_channels - ch_idx

        prefix = f"HD{ied:02d}MM{rows:02d}{cols:02d}"
        n_elec = rows * cols
        muscle = _extract_muscle(sidecar, info)
        muscle_tag = f"[MUSCLE:{muscle}]" if muscle else ""

        for k in range(n_elec):
            if ch_idx >= n_channels:
                break
            descriptions[ch_idx] = f"{prefix}-{group_name}-ch{k}{muscle_tag}"
            ch_idx += 1

        if ch_idx >= n_channels:
            break

    return descriptions


def _build_from_channels_tsv(
    n_channels: int,
    tsv_path: Path,
    sidecar: dict,
) -> list:
    """
    Build descriptions from a BIDS *_channels.tsv.

    For EMG channels the grid-geometry prefix is derived (in priority order):
      1. ``ElectrodeManufacturersModelName`` in the JSON sidecar
         (e.g. "GR04MM1305" → ied=4, rows=13, cols=5)
      2. ``InterelectrodeDistance`` in the JSON sidecar + channel count
         per group to infer the grid shape.

    For non-EMG channels (MISC, etc.) the ``description`` column from the
    TSV is used verbatim — this preserves keywords like "Requested Path"
    and "Performed Path" so ``EMGFile.grids`` can set the path indices.
    """
    rows_tsv = _parse_channels_tsv(tsv_path)

    if len(rows_tsv) != n_channels:
        raise ValueError(
            f"channels.tsv has {len(rows_tsv)} rows but EDF has {n_channels} "
            "data channels (after excluding annotation signals)."
        )

    # Resolve grid geometry from model name in sidecar
    model_name    = sidecar.get("ElectrodeManufacturersModelName", "")
    model_geom    = _parse_model_name(model_name)  # (ied, rows, cols) or None
    file_level_ied = _parse_ied_mm(sidecar.get("InterelectrodeDistance", 8))

    # Count channels per group to fall back on if model name is absent
    group_channel_count: dict = {}
    for row in rows_tsv:
        if row.get("type", "").strip().upper() == "EMG":
            grp = row.get("group", "").strip()
            group_channel_count[grp] = group_channel_count.get(grp, 0) + 1

    group_counters: dict = {}
    descriptions: list = []

    for row in rows_tsv:
        ch_type  = row.get("type",        "").strip().upper()
        desc_txt = row.get("description", "").strip()
        group    = row.get("group",       "").strip()
        muscle   = row.get("target_muscle", "").strip()
        ied_raw  = row.get("interelectrode_distance", "")

        if ch_type == "EMG":
            if model_geom:
                ied, grid_rows, grid_cols = model_geom
            else:
                # Derive IED: prefer per-channel TSV value, else sidecar
                try:
                    ied = int(float(ied_raw)) if ied_raw and ied_raw != "n/a" else file_level_ied
                except ValueError:
                    ied = file_level_ied

                # Infer grid shape from channel count per group
                n_in_group = group_channel_count.get(group, 1)
                grid_rows, grid_cols = _infer_grid_shape(n_in_group)

            prefix = f"HD{ied:02d}MM{grid_rows:02d}{grid_cols:02d}"
            k = group_counters.get(group, 0)
            group_counters[group] = k + 1

            muscle_tag = (
                f"[MUSCLE:{muscle}]"
                if muscle and muscle.lower() != "n/a"
                else ""
            )
            descriptions.append(f"{prefix}-{group}-ch{k}{muscle_tag}")

        else:
            # MISC, TRIG, etc. — use the description column verbatim so that
            # "Requested Path" and "Performed Path" are preserved for grid
            # parser detection.
            descriptions.append(desc_txt if desc_txt else f"{ch_type}_ch{len(descriptions)}")

    return descriptions


def _infer_grid_shape(n_channels: int) -> tuple:
    """
    Infer (rows, cols) for a grid with ``n_channels`` active EMG channels
    when the model name is unavailable.

    Preference: known OTBioelettronica grid sizes, then nearest-square.
    """
    known = {
        64:  (13, 5),   # GR04MM1305 / GR08MM1305  (65 electrodes, 1 ref)
        32:  (8,  4),   # GR04MM0804
        16:  (4,  4),   # GR10MM0404
        128: (16, 8),
        256: (16, 16),
    }
    if n_channels in known:
        return known[n_channels]

    # Nearest-square fallback
    cols = int(np.ceil(np.sqrt(n_channels)))
    rows = int(np.ceil(n_channels / cols))
    return rows, cols
