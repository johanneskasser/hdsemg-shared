from pathlib import Path
from dataclasses import dataclass, field
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
    emg_indices: list[int]
    ref_indices: list[int]
    rows: int
    cols: int
    ied_mm: int
    electrodes: int
    grid_key: str
    grid_uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    requested_path_idx: int | None = None
    performed_path_idx: int | None = None

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
        of Grid instances.
        """
        if self._grids is not None:
            return self._grids

        desc = self.description
        pattern = re.compile(r"HD(\d{2})MM(\d{2})(\d{2})")
        info: dict[str, dict] = {}
        current_key = None

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

        for idx, ent in enumerate(desc):
            txt = entry_text(ent)
            m = pattern.search(txt)
            if m:
                scale, rows, cols = map(int, m.groups())
                key = f"{rows}x{cols}"
                if key not in info:
                    # look up in JSON cache
                    prod = m.group(0).upper()
                    elec = next(
                        (g["electrodes"] for g in grid_data if g["product"].upper() == prod),
                        rows * cols
                    )
                    info[key] = {
                        "rows": rows, "cols": cols, "ied_mm": scale,
                        "electrodes": elec, "indices": [], "refs": [],
                        "req_idx": None, "perf_idx": None
                    }
                info[key]["indices"].append(idx)
                current_key = key
            else:
                if current_key:
                    if "requested path" in txt.lower():
                        info[current_key]["requested_path_idx"] = idx
                    if "performed path" in txt.lower():
                        info[current_key]["performed_path_idx"] = idx
                    info[current_key]["refs"].append((idx, txt))

        # build Grid objects
        self._grids = []
        for key, gi in info.items():
            grid = Grid(
                emg_indices=gi["indices"],
                ref_indices=[i for i, _ in gi["refs"]],
                rows=gi["rows"],
                cols=gi["cols"],
                ied_mm=gi["ied_mm"],
                electrodes=gi["electrodes"],
                grid_key=key,
                requested_path_idx=gi.get("requested_path_idx"),
                performed_path_idx=gi.get("performed_path_idx"),
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
