"""
Motor-unit discharge and PIC brace-method analysis for HD-sEMG.

This subpackage provides utilities for converting decomposed motor-unit (MU)
spike trains into instantaneous and smoothed discharge-rate traces, together
with a brace-method implementation for estimating PIC-related discharge
nonlinearity from a single MU.

The preferred brace-method API is the module namespace:

>>> from hdsemg_shared.motor_unit import brace_pic
>>> result = brace_pic.compute_brace_pic(smooth_rate_pps, reference_percent_mvt)
"""

# Module namespace exports:
#     from hdsemg_shared.motor_unit import discharge_rate
#     from hdsemg_shared.motor_unit import brace_pic
from . import discharge_rate
from . import brace_pic

from .discharge_rate import (
    MIN_DISCHARGES,
    firing_times_from_binary,
    firing_times_from_indices,
    instantaneous_discharge_rate,
    smooth_discharge_rate_svr,
)

from .brace_pic import (
    MAX_BRACE_HEIGHT_NORM,
    MetricInterval,
    BracePICCI,
    CIOptions,
    BracePICResult,
    compute_brace_pic,
    pics_brace,
    compute_pic_brace,
    brace_pic_from_spike_train,
    pics_brace_openhdemg_all,
    plot_brace,
)

__all__ = [
    # submodule namespaces
    "discharge_rate",
    "brace_pic",

    # discharge-rate utilities
    "MIN_DISCHARGES",
    "firing_times_from_binary",
    "firing_times_from_indices",
    "instantaneous_discharge_rate",
    "smooth_discharge_rate_svr",

    # brace-method PIC exports
    "MAX_BRACE_HEIGHT_NORM",
    "MetricInterval",
    "BracePICCI",
    "CIOptions",
    "BracePICResult",
    "compute_brace_pic",
    "pics_brace",
    "compute_pic_brace",
    "brace_pic_from_spike_train",
    "pics_brace_openhdemg_all",
    "plot_brace",
]
