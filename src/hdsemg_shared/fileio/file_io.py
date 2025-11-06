from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import os, time, json, re, uuid, requests

from .matlab_file_io import MatFileIO
from .otb_plus_file_io import load_otb_file
from .otb_4_file_io import load_otb4_file

# -----------------------------------------------------------------------------
# Grid dataclass
# -----------------------------------------------------------------------------
@dataclass
class Grid:
    """
    Represents a high-density EMG electrode grid.

    Attributes:
        emg_indices: List of channel indices for EMG electrodes
        ref_indices: List of channel indices for reference electrodes
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        ied_mm: Inter-electrode distance in millimeters
        electrodes: Total number of active electrodes
        grid_key: Unique identifier key (format: "{ied}mm_{rows}x{cols}" or with "_N" suffix)
        grid_uid: Unique UUID for this grid instance
        muscle: Optional muscle name where grid is placed (extracted from OTB4 XML <Muscle> tag)
        requested_path_idx: Optional index of "requested path" description entry
        performed_path_idx: Optional index of "performed path" description entry
    """
    emg_indices: list[int]
    ref_indices: list[int]
    rows: int
    cols: int
    ied_mm: int
    electrodes: int
    grid_key: str
    grid_uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    muscle: Optional[str] = None
    requested_path_idx: Optional[int] = None
    performed_path_idx: Optional[int] = None

# -----------------------------------------------------------------------------
# EMGFile: unified loader + grid extractor
# -----------------------------------------------------------------------------
class EMGFile:
    GRID_JSON_URL = (
        "https://drive.google.com/uc?export=download&"
        "id=1FqR6-ZlT1U74PluFEjCSeIS7NXJQUT-v"
    )
    CACHE_PATH = os.path.join(
        os.path.expanduser("~"), ".hdsemg_cache", "grid_data_cache.json"
    )
    _grid_cache: list[dict] | None = None

    def __init__(self, data, time, description, sf, file_name, file_size, file_type):
        self.data = data
        self.time = time
        self.description = description
        self.sampling_frequency = sf
        self.file_name = file_name
        self.file_size = file_size
        self.file_type = file_type
        self.channel_count = data.shape[1] if data.ndim > 1 else 1

        # parse out grids *once* on demand
        self._grids: list[Grid] | None = None

    @classmethod
    def load(cls, filepath: str) -> "EMGFile":
        """Factory: pick the right underlying loader, sanitize, and return EMGFile."""
        suffix = Path(filepath).suffix.lower()
        if suffix == ".mat":
            raw = MatFileIO.load(filepath)
            file_type = "mat"
        elif suffix in {".otb+", ".otb"}:
            raw = load_otb_file(filepath)
            file_type = "otb"
        elif suffix == ".otb4":
            raw = load_otb4_file(filepath)
            file_type = "otb4"
        else:
            raise ValueError(f"Unsupported file type: {suffix!r}")

        data, time, desc, sf, fn, fs = raw

        if data.dtype == np.int16:
            data = data.astype(np.float32)

        data, time = cls._sanitize(data, time)
        return cls(data, time, desc, sf, fn, fs, file_type)

    @staticmethod
    def _sanitize(data: np.ndarray, time: np.ndarray):
        data = np.atleast_2d(data)
        if data.shape[0] < data.shape[1]:
            data = data.T

        time = np.squeeze(time)
        if time.ndim == 2:
            time = time[:, 0] if time.shape[1] == 1 else time[0, :]
        if time.ndim == 1 and time.shape[0] != data.shape[0]:
            if time.shape[0] == data.shape[1]:
                time = time.T
            else:
                raise ValueError(f"Incompatible time {time.shape} for data {data.shape}")
        return data, time

    @property
    def grids(self) -> list[Grid]:
        """
        Lazily extract grid metadata from `self.description` and return a list
        of Grid instances. Handles multiple grids with identical specifications
        by detecting non-contiguous channel indices.
        """
        if self._grids is not None:
            return self._grids

        desc = self.description
        pattern = re.compile(r"HD(\d{2})MM(\d{2})(\d{2})")
        muscle_pattern = re.compile(r"\[MUSCLE:(.*?)\]")

        # Instead of dict, use list to allow multiple grids with same specs
        grid_instances: list[dict] = []
        current_grid = None

        # pull in (or fetch) the grid-data cache
        grid_data = self._load_grid_data()

        def entry_text(e):
            # Handle NumPy arrays
            if isinstance(e, np.ndarray):
                if e.size == 1:
                    return entry_text(e.item())  # recurse into the item
                else:
                    return str(e)  # fallback

            # Handle bytes
            if isinstance(e, bytes):
                try:
                    return e.decode("utf-8")
                except UnicodeDecodeError:
                    return e.decode("latin1")

            # Handle regular string
            if isinstance(e, str):
                return e

            # Fallback for anything else
            try:
                return str(e[0][0])  # often used in nested arrays from .mat
            except Exception:
                return str(e)

        def is_contiguous(indices: list[int], new_idx: int, tolerance: int = 5) -> bool:
            """Check if new_idx is contiguous with existing indices."""
            if not indices:
                return True
            # Check if within tolerance of the last index
            return abs(new_idx - indices[-1]) <= tolerance

        def find_or_create_grid(scale: int, rows: int, cols: int, idx: int) -> dict:
            """Find existing grid with matching specs and contiguous indices, or create new one."""
            # Look for existing grid with same specs
            base_key = f"{scale}mm_{rows}x{cols}"

            for grid_inst in grid_instances:
                if (grid_inst["ied_mm"] == scale and
                    grid_inst["rows"] == rows and
                    grid_inst["cols"] == cols and
                    is_contiguous(grid_inst["indices"], idx)):
                    return grid_inst

            # No contiguous grid found, create new instance
            # Look up electrode count from cache
            prod = f"HD{scale:02d}MM{rows:02d}{cols:02d}".upper()
            # Create transposed pattern
            prod_transposed = f"HD{scale:02d}MM{cols:02d}{rows:02d}".upper()

            elec = None
            for g in grid_data:
                g_prod_upper = g["product"].upper()
                if g_prod_upper == prod or g_prod_upper == prod_transposed:
                    elec = g["electrodes"]
                    break

            if elec is None:
                elec = rows * cols

            # Create unique key with instance counter if needed
            instance_num = sum(1 for g in grid_instances
                             if g["ied_mm"] == scale and g["rows"] == rows and g["cols"] == cols)
            if instance_num > 0:
                grid_key = f"{base_key}_{instance_num + 1}"
            else:
                grid_key = base_key

            new_grid = {
                "rows": rows,
                "cols": cols,
                "ied_mm": scale,
                "electrodes": elec,
                "indices": [],
                "refs": [],
                "req_idx": None,
                "perf_idx": None,
                "grid_key": grid_key,
                "muscle": None
            }
            grid_instances.append(new_grid)
            return new_grid

        for idx, ent in enumerate(desc):
            txt = entry_text(ent)
            m = pattern.search(txt)
            if m:
                scale, rows, cols = map(int, m.groups())
                current_grid = find_or_create_grid(scale, rows, cols, idx)
                current_grid["indices"].append(idx)

                # Extract muscle information if present
                muscle_match = muscle_pattern.search(txt)
                if muscle_match and current_grid["muscle"] is None:
                    current_grid["muscle"] = muscle_match.group(1).strip()
            else:
                if current_grid:
                    # Support both "requested path" and "original path" for the requested/original path index
                    if "requested path" in txt.lower() or "original path" in txt.lower():
                        current_grid["req_idx"] = idx
                    if "performed path" in txt.lower():
                        current_grid["perf_idx"] = idx
                    current_grid["refs"].append((idx, txt))

        # build Grid objects
        self._grids = []
        for gi in grid_instances:
            grid = Grid(
                emg_indices=gi["indices"],
                ref_indices=[i for i, _ in gi["refs"]],
                rows=gi["rows"],
                cols=gi["cols"],
                ied_mm=gi["ied_mm"],
                electrodes=gi["electrodes"],
                grid_key=gi["grid_key"],
                muscle=gi.get("muscle"),
                requested_path_idx=gi.get("req_idx"),
                performed_path_idx=gi.get("perf_idx"),
            )
            self._grids.append(grid)

        return self._grids

    def save(self, save_path: str) -> None:
        if save_path.endswith(".mat"):
            MatFileIO.save(save_path, self.data, self.time, self.description, self.sampling_frequency)
        else:
            file_format = save_path.split('.')[-1].lower()
            raise ValueError(f"Unsupported save format: {file_format!r}")

    @classmethod
    def _load_grid_data(cls) -> list[dict]:
        """
        Load from cache if < 1 week old, else fetch from URL.
        """
        if cls._grid_cache is not None:
            return cls._grid_cache

        os.makedirs(os.path.dirname(cls.CACHE_PATH), exist_ok=True)
        one_week = 7 * 24 * 3600
        try:
            if os.path.exists(cls.CACHE_PATH):
                age = time.time() - os.path.getmtime(cls.CACHE_PATH)
                if age < one_week:
                    with open(cls.CACHE_PATH) as f:
                        cls._grid_cache = json.load(f)
                        return cls._grid_cache
        except Exception:
            pass

        try:
            r = requests.get(cls.GRID_JSON_URL, timeout=10)
            r.raise_for_status()
            cls._grid_cache = r.json()
            with open(cls.CACHE_PATH, "w") as f:
                json.dump(cls._grid_cache, f)
        except Exception:
            cls._grid_cache = []
        return cls._grid_cache

    def get_grid(self, *, grid_key: str = None, grid_uid: str = None) -> Grid | None:
        """
        Searches for a Grid by its key or UID.
        If both are None, returns None.
        """
        if self._grids is None:
            _ = self.grids  # Initialisiere Grids falls noch nicht geschehen
        if grid_key is not None:
            for g in self._grids:
                if g.grid_key == grid_key:
                    return g
        if grid_uid is not None:
            for g in self._grids:
                if g.grid_uid == grid_uid:
                    return g
        return None

    def copy(self):
        """
        Returns a deep copy of the EMGFile instance.
        """
        import copy
        return copy.deepcopy(self)
