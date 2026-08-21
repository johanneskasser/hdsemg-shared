from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import os, time, json, re, uuid, requests
import logging

from .matlab_file_io import MatFileIO
from .otb_plus_file_io import load_otb_file
from .otb_4_file_io import load_otb4_file
from .edf_file_io import load_edf_file
from .units import conversion_factor, normalize_unit

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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
        model_code: Raw electrode model code as written in the file (e.g. "HD10MM0408"),
            preserved even when rows/cols were normalized against the product catalog
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
    model_code: Optional[str] = None
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

    def __init__(self, data, time, description, sf, file_name, file_size, file_type,
                 unit: Optional[str] = None):
        self.data = data
        self.time = time
        self.description = description
        self.sampling_frequency = sf
        self.file_name = file_name
        self.file_size = file_size
        self.file_type = file_type
        #: Unit of the EMG channels, one of :data:`~hdsemg_shared.fileio.units.CANONICAL_UNITS`,
        #: or ``None`` when the file does not declare one. Reference/auxiliary
        #: channels (force, paths, AUX) are NOT in this unit — see :meth:`to_unit`.
        self.unit = normalize_unit(unit)
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
        elif suffix == ".edf":
            raw = load_edf_file(filepath)
            file_type = "edf"
        else:
            raise ValueError(f"Unsupported file type: {suffix!r}")

        # Loaders append the declared unit; tolerate the older 6-tuple so a
        # third-party loader or a monkeypatched one keeps working.
        data, time, desc, sf, fn, fs, *extra = raw
        unit = extra[0] if extra else None

        if data.dtype == np.int16:
            data = data.astype(np.float32)

        data, time = cls._sanitize(data, time)
        return cls(data, time, desc, sf, fn, fs, file_type, unit)

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
        pattern = re.compile(r"(HD|GR)(\d{2})MM(\d{2})(\d{2})")
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

        def find_or_create_grid(scale: int, rows: int, cols: int, idx: int,
                                muscle: Optional[str] = None,
                                elec: Optional[int] = None,
                                model_code: Optional[str] = None) -> dict:
            """Find existing grid with matching specs and contiguous indices, or create new one."""
            # Look for existing grid with same specs
            base_key = f"{scale}mm_{rows}x{cols}"

            for grid_inst in grid_instances:
                # Match by specs, contiguity, AND muscle (if available)
                specs_match = (grid_inst["ied_mm"] == scale and
                              grid_inst["rows"] == rows and
                              grid_inst["cols"] == cols)

                # If muscle info is available, use it to differentiate grids
                if muscle is not None and grid_inst.get("muscle") is not None:
                    muscle_match = grid_inst["muscle"] == muscle
                else:
                    muscle_match = True  # No muscle info, don't use for matching

                if specs_match and muscle_match and is_contiguous(grid_inst["indices"], idx):
                    return grid_inst

            # No contiguous grid found, create new instance
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
                "model_code": model_code,
                "muscle": None
            }
            grid_instances.append(new_grid)
            return new_grid

        for idx, ent in enumerate(desc):
            txt = entry_text(ent)
            m = pattern.search(txt)
            if m:
                prefix, scale, rows, cols = m.group(1).upper(), *map(int, m.groups()[1:])
                model_code = m.group(0).upper()
                rows, cols, elec = self._normalize_grid_dims(
                    prefix, scale, rows, cols, grid_data
                )

                # Extract muscle information if present (do this BEFORE find_or_create_grid)
                muscle = None
                muscle_match = muscle_pattern.search(txt)
                if muscle_match:
                    muscle = muscle_match.group(1).strip()

                # Pass muscle info to find_or_create_grid for proper differentiation
                current_grid = find_or_create_grid(scale, rows, cols, idx, muscle,
                                                   elec=elec, model_code=model_code)
                current_grid["indices"].append(idx)

                # Store muscle info in grid if not already set
                if muscle and current_grid["muscle"] is None:
                    current_grid["muscle"] = muscle
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
                model_code=gi.get("model_code"),
                muscle=gi.get("muscle"),
                requested_path_idx=gi.get("req_idx"),
                performed_path_idx=gi.get("perf_idx"),
            )
            self._grids.append(grid)

        return self._grids

    def save(self, save_path: str) -> None:
        if save_path.endswith(".mat"):
            MatFileIO.save(save_path, self.data, self.time, self.description,
                           self.sampling_frequency, self.unit)
        else:
            file_format = save_path.split('.')[-1].lower()
            raise ValueError(f"Unsupported save format: {file_format!r}")

    @classmethod
    def _normalize_grid_dims(cls, prefix: str, scale: int, rows: int, cols: int,
                             grid_data: list[dict]) -> tuple[int, int, Optional[int]]:
        """
        Validate the parsed row/column digits against the product catalog.

        OTBiolab sometimes writes the two 2-digit groups in reversed order
        (e.g. "HD10MM0408" for the matrix that only exists as HD10MM0804).
        The electrode count is identical either way, so the transposition is
        invisible downstream. If the parsed order is not a real product but the
        reversed order is, the reversed order wins and the swap is logged.

        Returns (rows, cols, electrodes) - electrodes is None when the product
        is unknown (empty/unavailable catalog), leaving rows/cols untouched.
        """
        catalog = {g["product"].upper(): g["electrodes"] for g in grid_data if "product" in g}
        prod = f"{prefix}{scale:02d}MM{rows:02d}{cols:02d}"
        prod_transposed = f"{prefix}{scale:02d}MM{cols:02d}{rows:02d}"

        if prod in catalog:
            return rows, cols, catalog[prod]

        if prod_transposed in catalog:
            logger.warning(
                "Electrode model %s does not exist in the product catalog; "
                "%s does. Normalizing grid to %dx%d (rows x cols).",
                prod, prod_transposed, cols, rows,
            )
            return cols, rows, catalog[prod_transposed]

        # Also hit by the pseudo-grids OTB4 emits for control/aux tracks, so debug.
        logger.debug(
            "Electrode model %s not found in the product catalog; "
            "using the row/column order as written in the file.", prod,
        )
        return rows, cols, None

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

    @classmethod
    def supported_file_types(cls) -> list[tuple[str, str]]:
        """
        Returns file type filters for use in file picker dialogs (e.g. tkinter filedialog).

        Each entry is a (description, pattern) tuple where the pattern is a
        space-separated list of glob expressions. Pass the return value directly
        to tkinter's ``filedialog.askopenfilename(filetypes=...)``.

        Example::

            from tkinter import filedialog
            path = filedialog.askopenfilename(
                filetypes=EMGFile.supported_file_types()
            )

        Returns
        -------
        list of (str, str)
            Ordered list of (description, glob_pattern) tuples.
        """
        return [
            ("EMG Files", "*.mat *.otb *.otb+ *.otb4 *.edf"),
            ("MATLAB Files", "*.mat"),
            ("OTB / OTB+ Files", "*.otb *.otb+"),
            ("OTB4 Files", "*.otb4"),
            ("EDF Files", "*.edf"),
            ("All Files", "*.*"),
        ]

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """
        Returns the list of file extensions that ``EMGFile.load`` accepts.

        Returns
        -------
        list of str
            Lower-case extensions including the leading dot.
        """
        return [".mat", ".otb", ".otb+", ".otb4", ".edf"]

    def scale_to(self, unit: str) -> float:
        """
        Factor that converts this file's EMG channels into ``unit``.

        Raises ``ValueError`` when :attr:`unit` is ``None`` (the file declared
        nothing) or when either unit is not convertible, e.g. ``"a.u."``.

        Example::

            emg.data[:, grid.emg_indices] * emg.scale_to("uV")
        """
        return conversion_factor(self.unit, unit)

    def to_unit(self, unit: str) -> "EMGFile":
        """
        Return a copy whose EMG channels are expressed in ``unit``.

        Only the grid EMG channels are scaled. Reference and auxiliary channels
        (force, requested/performed path, AUX) carry their own units and are
        left untouched. When no grids could be parsed, every channel is scaled,
        since there is nothing to distinguish EMG from the rest.
        """
        factor = self.scale_to(unit)
        data = self.data.astype(np.float64)          # always a fresh array
        emg_indices = sorted({i for g in self.grids for i in g.emg_indices})
        if emg_indices:
            data[:, emg_indices] *= factor
        else:
            logger.warning(
                "No grids parsed for %s; scaling every channel to %s.",
                self.file_name, unit,
            )
            data *= factor

        out = self.copy()
        out.data = data
        out.unit = unit
        return out

    def copy(self):
        """
        Returns a deep copy of the EMGFile instance.
        """
        import copy
        return copy.deepcopy(self)
