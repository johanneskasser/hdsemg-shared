"""
Canonical signal units for :class:`~hdsemg_shared.fileio.file_io.EMGFile`.

Loaders report what the file *declares*, normalized to one of
``"V"``, ``"mV"``, ``"uV"`` or ``"a.u."``. ``None`` means the format did not
say — that is deliberate: a consumer can decide what to do about an unknown
unit, but cannot recover from a default that looks authoritative.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: The units a loader may report. Anything else normalizes to ``None``.
CANONICAL_UNITS = ("V", "mV", "uV", "a.u.")

# Scale of one unit expressed in volts. "a.u." is deliberately absent:
# arbitrary units cannot be converted.
_VOLT_SCALE = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}

_ALIASES = {
    "v": "V", "volt": "V", "volts": "V",
    "mv": "mV", "millivolt": "mV", "millivolts": "mV",
    # µ (U+00B5 micro sign) and μ (U+03BC greek mu) both occur in OTB files.
    "uv": "uV", "µv": "uV", "μv": "uV",
    "microvolt": "uV", "microvolts": "uV",
    "a.u.": "a.u.", "au": "a.u.", "a.u": "a.u.",
    "arbitrary": "a.u.", "arbitrary units": "a.u.",
}


def normalize_unit(raw) -> Optional[str]:
    """
    Map a unit string as written in a file onto :data:`CANONICAL_UNITS`.

    Returns ``None`` for empty, missing or unrecognized values.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    canonical = _ALIASES.get(text.lower())
    if canonical is None:
        logger.debug("Unrecognized signal unit %r; reporting None.", text)
    return canonical


def conversion_factor(from_unit: Optional[str], to_unit: str) -> float:
    """
    Factor to multiply data in ``from_unit`` by to obtain ``to_unit``.

    Raises ``ValueError`` when either side is unknown or not convertible
    (``"a.u."`` never is).
    """
    if from_unit is None:
        raise ValueError(
            "Source unit is unknown (None); the file did not declare one."
        )
    if from_unit not in _VOLT_SCALE or to_unit not in _VOLT_SCALE:
        raise ValueError(
            f"Cannot convert {from_unit!r} to {to_unit!r}; "
            f"convertible units are {sorted(_VOLT_SCALE)}."
        )
    return _VOLT_SCALE[from_unit] / _VOLT_SCALE[to_unit]
