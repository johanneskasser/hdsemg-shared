"""
Brace-height quantification of persistent inward currents (PICs) from a single
motor-unit (MU) discharge profile.

Beauchamp et al. (2023) proposed a pseudo-geometric proxy for PIC amplification:
during a linear (triangular) ramp contraction, a MU that behaves as a passive
integrator of synaptic drive would discharge linearly with the produced
force/torque. Intrinsic activation from PICs makes the discharge rate rise
steeply just after recruitment and then attenuate, bowing the discharge-vs-force
trace away from a straight line. The maximum deviation from that straight line —
the **brace height** — therefore serves as a single-unit estimate of PIC
amplification (and, by extension, neuromodulatory drive).

Geometry (paper §2.3.2)
-----------------------
For the ascending segment from MU recruitment to the instant of *peak discharge
rate*, with reference force/torque ``x`` and smoothed discharge rate ``y``:

* The "theoretical linear discharge" is the straight hypotenuse from
  ``(x_rec, y_rec)`` to ``(x_peak, y_peak)``.
* Brace height is the maximum orthogonal distance from that hypotenuse to the
  discharge trace; the location of that maximum is the **brace point**.
* It is normalized to the altitude of the right triangle whose hypotenuse is the
  same line, giving units of **percent of the right-triangle height (% rTri)** —
  the deviation that would occur if full PIC activation drove the MU to peak
  discharge immediately at recruitment.

Because discharge rate (pps) and reference force/torque are on different scales,
the orthogonal geometry is evaluated in axes normalized to the recruitment→peak
range. In that frame the hypotenuse runs from ``(0, 0)`` to ``(1, 1)`` and the
right-triangle altitude is ``1/sqrt(2)``; the ``sqrt(2)`` cancels, so the
normalized brace height reduces to the cleanly scale-invariant expression

    brace_height_norm = 100 * max( (y - y_rec)/(y_peak - y_rec)
                                   - (x - x_rec)/(x_peak - x_rec) )

and the brace point is the index attaining that maximum. The raw brace height is
reported as the equivalent vertical deviation in pps,
``(y_peak - y_rec) * max(...)``.

Supplemental metrics (computed in the raw ``x``/``y`` units, so the reference
should be in its natural units, e.g. % MVT):

* ``acceleration_slope`` — slope of the chord from recruitment to the brace
  point (the steep "secondary range").
* ``attenuation_slope`` — slope of the chord from the brace point to peak
  discharge (the attenuated "tertiary range").
* ``angle`` — the (reflex) vertex angle at the brace point of the polyline
  recruitment → brace → peak, in degrees (180 deg for a perfectly linear trace,
  increasing as the trace bows further from linear).

Exclusion criteria (paper §2.3.2): a MU is flagged invalid if the acceleration
slope is negative, the normalized brace height exceeds 200 % rTri, or peak
discharge occurs after peak force/torque.

References:
- Beauchamp et al. (2023), *J. Neural Eng.* 20 016034.

Usage:
>>> result = compute_brace_height(smooth_rate, torque)
>>> result.brace_height_norm     # % rTri
>>> result.acceleration_slope, result.attenuation_slope, result.angle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from .discharge_rate import (
    firing_times_from_binary,
    firing_times_from_indices,
    instantaneous_discharge_rate,
)

#: Normalized brace height above which a MU is flagged for irregularities.
MAX_BRACE_HEIGHT_NORM = 200.0


@dataclass
class BraceHeightResult:
    """Result of a brace-height quantification for a single MU.

    Attributes
    ----------
    brace_height : float
        Raw brace height as the maximum vertical deviation of discharge rate
        above the recruitment→peak line, in pps.
    brace_height_norm : float
        Brace height normalized to the right-triangle altitude, in percent of
        the right triangle (% rTri).
    acceleration_slope : float
        Slope of the recruitment→brace-point chord (pps per reference unit).
    attenuation_slope : float
        Slope of the brace-point→peak chord (pps per reference unit).
    angle : float
        Reflex vertex angle at the brace point of the recruitment→brace→peak
        polyline, in degrees.
    recruitment_idx, brace_idx, peak_idx : int
        Indices (into the input arrays) of recruitment, the brace point, and
        peak discharge rate.
    valid : bool
        ``False`` if any exclusion criterion was triggered.
    exclusion_reasons : list of str
        Human-readable reasons for exclusion (empty if ``valid``).
    x, y : np.ndarray
        The reference and discharge-rate values of the analysed
        recruitment→peak segment.
    """

    brace_height: float
    brace_height_norm: float
    acceleration_slope: float
    attenuation_slope: float
    angle: float
    recruitment_idx: int
    brace_idx: int
    peak_idx: int
    valid: bool
    exclusion_reasons: List[str] = field(default_factory=list)
    x: np.ndarray = field(default=None, repr=False)
    y: np.ndarray = field(default=None, repr=False)


def compute_brace_height(
    discharge_rate: np.ndarray,
    reference: np.ndarray,
    *,
    recruitment_idx: Optional[int] = None,
    peak_idx: Optional[int] = None,
    peak_reference_idx: Optional[int] = None,
) -> BraceHeightResult:
    """
    Compute brace height and its supplemental metrics for a single MU.

    Parameters
    ----------
    discharge_rate : np.ndarray
        Smoothed continuous discharge-rate trace (pps), sampled on the same time
        base as ``reference``.
    reference : np.ndarray
        Reference force/torque trace (e.g. % MVT) on the same time base.
    recruitment_idx : int, optional
        Index of MU recruitment. Defaults to ``0`` (the trace is assumed to
        start at recruitment, as produced by the smoothing step).
    peak_idx : int, optional
        Index of peak discharge rate. Defaults to ``argmax(discharge_rate)``
        within the recruitment→end window.
    peak_reference_idx : int, optional
        Index of peak reference force/torque, used for the "peak discharge after
        peak torque" exclusion. Defaults to ``argmax(reference)``.

    Returns
    -------
    BraceHeightResult

    Raises
    ------
    ValueError
        If the inputs are not 1D arrays of equal length, or the recruitment→peak
        segment is degenerate (fewer than three samples or zero span).
    """
    y = np.asarray(discharge_rate, dtype=np.float64)
    x = np.asarray(reference, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("discharge_rate and reference must be 1D arrays.")
    if x.shape != y.shape:
        raise ValueError("discharge_rate and reference must have equal length.")

    if recruitment_idx is None:
        recruitment_idx = 0
    if peak_reference_idx is None:
        peak_reference_idx = int(np.argmax(x))
    if peak_idx is None:
        # Peak discharge rate at or after recruitment.
        peak_idx = recruitment_idx + int(np.argmax(y[recruitment_idx:]))

    if not (0 <= recruitment_idx < peak_idx < x.size):
        raise ValueError(
            "Require 0 <= recruitment_idx < peak_idx < len(signal); "
            f"got recruitment_idx={recruitment_idx}, peak_idx={peak_idx}, "
            f"len={x.size}."
        )

    seg_x = x[recruitment_idx : peak_idx + 1]
    seg_y = y[recruitment_idx : peak_idx + 1]
    if seg_x.size < 3:
        raise ValueError("Recruitment-to-peak segment must contain >= 3 samples.")

    dx = seg_x[-1] - seg_x[0]
    dy = seg_y[-1] - seg_y[0]
    if dx == 0 or dy == 0:
        raise ValueError(
            "Degenerate segment: reference and discharge rate must each change "
            "between recruitment and peak."
        )

    # Deviation from the recruitment->peak line in axes normalized to the
    # recruitment->peak range. u, v in [0, 1] at the endpoints; the line is v = u
    # and (v - u) is the scale-invariant deviation (positive above the line).
    u = (seg_x - seg_x[0]) / dx
    v = (seg_y - seg_y[0]) / dy
    deviation = v - u

    local_brace = int(np.argmax(deviation))
    brace_idx = recruitment_idx + local_brace
    max_dev = float(deviation[local_brace])

    brace_height_norm = 100.0 * max_dev
    brace_height = dy * max_dev  # equivalent vertical deviation in pps

    # Supplemental metrics in raw units, from the two phase chords.
    x_rec, y_rec = seg_x[0], seg_y[0]
    x_brace, y_brace = seg_x[local_brace], seg_y[local_brace]
    x_peak, y_peak = seg_x[-1], seg_y[-1]

    acc_dx = x_brace - x_rec
    att_dx = x_peak - x_brace
    acceleration_slope = (y_brace - y_rec) / acc_dx if acc_dx != 0 else np.nan
    attenuation_slope = (y_peak - y_brace) / att_dx if att_dx != 0 else np.nan
    angle = _vertex_angle(
        (x_rec, y_rec), (x_brace, y_brace), (x_peak, y_peak)
    )

    # Exclusion criteria (flagged, not removed).
    reasons: List[str] = []
    if not np.isnan(acceleration_slope) and acceleration_slope < 0:
        reasons.append("negative acceleration slope")
    if brace_height_norm > MAX_BRACE_HEIGHT_NORM:
        reasons.append(f"normalized brace height > {MAX_BRACE_HEIGHT_NORM:.0f}% rTri")
    if peak_idx > peak_reference_idx:
        reasons.append("peak discharge after peak force/torque")

    return BraceHeightResult(
        brace_height=brace_height,
        brace_height_norm=brace_height_norm,
        acceleration_slope=acceleration_slope,
        attenuation_slope=attenuation_slope,
        angle=angle,
        recruitment_idx=recruitment_idx,
        brace_idx=brace_idx,
        peak_idx=peak_idx,
        valid=len(reasons) == 0,
        exclusion_reasons=reasons,
        x=seg_x,
        y=seg_y,
    )


def _vertex_angle(rec, brace, peak) -> float:
    """Reflex vertex angle (deg) at ``brace`` of the polyline rec->brace->peak.

    Returns 180 deg when the three points are collinear (linear discharge) and
    increases above 180 deg as the brace point bows away from the line.
    """
    ba = np.array([rec[0] - brace[0], rec[1] - brace[1]], dtype=np.float64)
    bc = np.array([peak[0] - brace[0], peak[1] - brace[1]], dtype=np.float64)
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return np.nan
    cos_theta = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))  # interior angle in [0, 180]
    return 360.0 - theta  # reflex angle, on the hypotenuse side of the bow


def brace_height_from_spike_train(
    spikes: np.ndarray,
    reference: np.ndarray,
    fsamp: float,
    *,
    kind: str = "auto",
    smoother=None,
    n_eval: int = 200,
) -> BraceHeightResult:
    """
    Convenience wrapper: brace height directly from a MUAP spike train.

    Ties together discharge-rate computation, smoothing, and brace-height
    quantification. The reference force/torque is resampled (by linear
    interpolation) onto the smoothed discharge-rate time base.

    Parameters
    ----------
    spikes : np.ndarray
        Either a binary spike train (``kind="binary"``) or discharge sample
        indices (``kind="indices"``). With ``kind="auto"`` (default), an array
        containing only 0/1 values is treated as a binary train; otherwise it is
        treated as discharge indices.
    reference : np.ndarray
        Reference force/torque trace sampled at ``fsamp`` (same recording time
        base as the spike train).
    fsamp : float
        Sampling frequency in Hz.
    kind : {"auto", "binary", "indices"}
        How to interpret ``spikes``.
    smoother : callable, optional
        ``smoother(times, rate, t_eval) -> (t_eval, smooth_rate)``. Defaults to
        :func:`~hdsemg_shared.motor_unit.discharge_rate.smooth_discharge_rate_svr`
        (requires scikit-learn).
    n_eval : int
        Number of points in the smoothed trace from recruitment to derecruitment.

    Returns
    -------
    BraceHeightResult

    Raises
    ------
    ValueError
        If ``kind`` is invalid or inputs are malformed.
    ImportError
        If the default SVR smoother is used without scikit-learn installed.
    """
    spikes = np.asarray(spikes)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 1:
        raise ValueError("reference must be a 1D array.")

    if kind == "auto":
        unique = np.unique(spikes)
        is_binary = np.all(np.isin(unique, (0, 1))) and spikes.size == reference.size
        kind = "binary" if is_binary else "indices"

    if kind == "binary":
        firing_times = firing_times_from_binary(spikes, fsamp)
    elif kind == "indices":
        firing_times = firing_times_from_indices(spikes, fsamp)
    else:
        raise ValueError("kind must be one of {'auto', 'binary', 'indices'}.")

    rate_times, rate = instantaneous_discharge_rate(firing_times)

    if smoother is None:
        from .discharge_rate import smooth_discharge_rate_svr as smoother

    t_eval = np.linspace(rate_times[0], rate_times[-1], n_eval)
    t_eval, smooth_rate = smoother(rate_times, rate, t_eval)

    # Resample the reference onto the smoothed discharge-rate time base.
    ref_time = np.arange(reference.size, dtype=np.float64) / float(fsamp)
    ref_on_rate = np.interp(t_eval, ref_time, reference)

    return compute_brace_height(smooth_rate, ref_on_rate)
