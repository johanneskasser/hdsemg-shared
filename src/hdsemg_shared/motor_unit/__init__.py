"""
Motor-unit discharge analysis for HD-sEMG.

This subpackage provides utilities for converting decomposed motor-unit (MU)
spike trains into instantaneous and smoothed discharge-rate traces, and the
PIC-oriented brace-method implementation of Beauchamp et al. (2023).

>>> from hdsemg_shared.motor_unit import brace_pic
>>> result = brace_pic.compute_brace_pic(smooth_rate_pps, reference_percent_mvt)
"""

from .discharge_rate import (
    MIN_DISCHARGES,
    firing_times_from_binary,
    firing_times_from_indices,
    instantaneous_discharge_rate,
    smooth_discharge_rate_svr,
)
from .brace_pic import (
    MetricInterval,
    BracePICCI,
    CIOptions,
    BracePICResult,
    compute_brace_pic,
    brace_pic_from_spike_train,
    compute_brace_pic_openhdemg_all,
    plot_brace,
)

__all__ = [
    "MIN_DISCHARGES",
    "firing_times_from_binary",
    "firing_times_from_indices",
    "instantaneous_discharge_rate",
    "smooth_discharge_rate_svr",
    "MetricInterval",
    "BracePICCI",
    "CIOptions",
    "BracePICResult",
    "compute_brace_pic",
    "brace_pic_from_spike_train",
    "compute_brace_pic_openhdemg_all",
    "plot_brace",
]
