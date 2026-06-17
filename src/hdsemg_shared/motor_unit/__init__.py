"""
Motor-unit discharge analysis for HD-sEMG.

Currently provides the brace-height method of Beauchamp et al. (2023) for
estimating persistent inward current (PIC) amplification from a single
motor-unit (MU) discharge profile, together with the discharge-rate utilities it
builds upon.
"""

from .discharge_rate import (
    MIN_DISCHARGES,
    firing_times_from_binary,
    firing_times_from_indices,
    instantaneous_discharge_rate,
    smooth_discharge_rate_svr,
)
from .brace_height import (
    MAX_BRACE_HEIGHT_NORM,
    BraceHeightResult,
    brace_height_from_spike_train,
    compute_brace_height,
)

__all__ = [
    "MIN_DISCHARGES",
    "firing_times_from_binary",
    "firing_times_from_indices",
    "instantaneous_discharge_rate",
    "smooth_discharge_rate_svr",
    "MAX_BRACE_HEIGHT_NORM",
    "BraceHeightResult",
    "brace_height_from_spike_train",
    "compute_brace_height",
]
