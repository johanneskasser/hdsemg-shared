
"""
PIC brace-method quantification for motor-unit discharge profiles.

This module supersedes ``brace_height.py`` while preserving its main public
entry points.  It implements the pseudo-geometric brace method of Beauchamp et
al. (2023) for estimating persistent inward-current (PIC) amplification in
individual motor units (MUs).

Method summary
--------------
The input to the geometry is a smoothed MU discharge-rate trace ``y`` in pulses
per second (pps) and the concurrent reference force/torque trace ``x``
typically expressed as percent maximum voluntary torque/force (%MVT).  On the
ascending segment from recruitment to peak discharge, a theoretical linear
discharge trace is the straight line from recruitment to peak discharge.
Brace height is the maximum orthogonal distance from that line to the smoothed
trace, normalized by the altitude of the corresponding right triangle.  In
right-triangle normalized coordinates this reduces to

    100 * max((y - y_rec) / (y_peak - y_rec)
              - (x - x_rec) / (x_peak - x_rec))

and is reported in percent right-triangle height (% rTri).

The same brace point segments the ascending discharge profile into an
acceleration phase and an attenuation phase.  The module reports their slopes
(pps/%MVT when the reference is %MVT) and the reflex angle of the
recruitment-brace-peak polyline.

Optional uncertainty estimates
------------------------------
``compute_brace_height(..., ci=95)`` or ``pics_brace(..., ci=95)`` adds
intervals to the returned dataclass.  The default CI engine is an experimental
posterior-predictive HDI:

1. infer plausible discharge times from the smoothed pps trace;
2. jitter those times using a discharge-time uncertainty model;
3. recompute instantaneous discharge rate;
4. refit the discharge-rate smoother via
   ``hdsemg_shared.motor_unit.discharge_rate.smooth_discharge_rate_svr``;
5. recompute the brace metrics for every draw;
6. summarize the draw distribution using either HDI or ETI intervals.

This interval is a model-based sensitivity/credible interval, not an
author-validated clinical confidence interval.  It is intended to expose how
brace metrics respond to spike timing, smoothing, and endpoint choices.

References
----------
- Beauchamp JA, Pearcey GEP, Khurram OU, Chardon M, Wang YC, Powers RK,
  Dewald JPA, Heckman CJ. A geometric approach to quantifying the
  neuromodulatory effects of persistent inward currents on individual motor
  unit discharge patterns. J Neural Eng. 2023;20(1):016034.
  doi:10.1088/1741-2552/acb1d7.
- The local ``discharge_rate.py`` module supplies spike-time conversion,
  instantaneous discharge rate, and SVR smoothing utilities.

Example
-------
>>> result = compute_brace_height(smooth_rate_pps, torque_percent_mvt, fsamp=2048)
>>> result.brace_height_norm
>>> result.acceleration_slope, result.attenuation_slope, result.angle
>>> ax = plot_brace(result)

Backward-compatible aliases:
``pics_brace`` and ``compute_pic_brace`` call ``compute_brace_height``;
``BraceHeightResult`` aliases ``BracePICResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import concurrent.futures as _futures
import math
import os

import numpy as np


MAX_BRACE_HEIGHT_NORM = 200.0
DEFAULT_MIN_DISCHARGES = 10


MetricName = str
IntervalKind = str
CIMethod = str


@dataclass
class MetricInterval:
    """One uncertainty interval for one scalar brace metric.

    Attributes
    ----------
    point : float
        Deterministic estimate from the input trace.
    mean, sd : float
        Mean and sample standard deviation across CI draws.
    lower, upper : float
        Interval bounds.  Their interpretation depends on ``interval``:
        highest-density interval (HDI) or equal-tailed interval (ETI).
    level : float
        Requested interval mass in percent, e.g. 95.
    interval : {"hdi", "eti"}
        Interval construction method.
    n : int
        Number of finite draws used.
    """

    point: float
    mean: float
    sd: float
    lower: float
    upper: float
    level: float
    interval: IntervalKind
    n: int


@dataclass
class BracePICCI:
    """Uncertainty summary attached to :class:`BracePICResult`.

    ``draws`` stores one array per metric.  If ``store_trace_summary`` is true in
    :class:`CIOptions`, ``trace_mean`` and ``trace_sd`` contain the mean and
    standard deviation of the smoothed discharge-rate draws on the analysed time
    base for use by :func:`plot_brace`.
    """

    level: float
    method: CIMethod
    interval: IntervalKind
    intervals: Dict[MetricName, MetricInterval]
    draws: Dict[MetricName, np.ndarray] = field(default_factory=dict, repr=False)
    trace_reference: Optional[np.ndarray] = field(default=None, repr=False)
    trace_mean: Optional[np.ndarray] = field(default=None, repr=False)
    trace_sd: Optional[np.ndarray] = field(default=None, repr=False)
    n_requested: int = 0
    n_successful: int = 0
    n_failed: int = 0
    options: Dict[str, Any] = field(default_factory=dict, repr=False)

    def metric(self, name: str) -> MetricInterval:
        """Return the interval summary for a metric name."""
        return self.intervals[name]


@dataclass
class CIOptions:
    """Configuration for optional brace-metric uncertainty estimation.

    Parameters
    ----------
    level : float
        Interval mass in percent.  ``ci=95`` sets this to 95.
    method : {"jitter_svr", "trace_noise", "sensitivity"}
        ``"jitter_svr"`` is the recommended experimental default.  It infers
        latent discharge times from the smoothed pps trace, jitters them, refits
        SVR discharge-rate smoothing, and recomputes brace metrics.
        ``"trace_noise"`` perturbs the smoothed trace directly and is faster but
        less physiologically motivated.
        ``"sensitivity"`` evaluates deterministic endpoint/smoothing choices and
        reports their envelope as an interval-like summary.
    interval : {"hdi", "eti"}
        HDI gives the shortest interval containing the requested draw mass; ETI
        gives equal-tailed quantiles.
    n_draws : int
        Number of posterior-predictive or perturbation draws.
    n_jobs : int
        Parallel workers.  ``1`` disables parallelism; ``-1`` uses all CPUs.
    parallel_backend : {"thread", "process"}
        Threading has lower overhead and avoids pickling issues with package
        imports.  Process mode can help for very large draw counts.
    random_state : int, optional
        Seed for reproducible CI draws.
    jitter_sd_s : float, optional
        Absolute discharge-time jitter SD in seconds.  If omitted, SD is
        ``jitter_fraction_isi / local_rate`` for each inferred spike.
    jitter_fraction_isi : float
        Relative jitter as a fraction of local ISI when ``jitter_sd_s`` is not
        supplied.
    min_isi_s : float
        Lower bound for reconstructed ISIs after jittering.
    trace_noise_sd : float, optional
        Direct trace-noise SD for ``method="trace_noise"``.  If omitted, a
        robust second-difference estimate is used.
    svr_kwargs : dict
        Keyword arguments passed to ``smooth_discharge_rate_svr`` during
        ``method="jitter_svr"``.
    recruitment_windows, peak_windows, brace_windows : tuple[int, ...]
        Deterministic averaging-window choices used by
        ``method="sensitivity"``.  Values are in samples on the analysed trace.
    store_draws : bool
        Keep scalar draw arrays in ``result.ci.draws``.
    store_trace_summary : bool
        Keep mean and SD of smoothed discharge-rate draws for plotting.
    """

    level: float = 95.0
    method: CIMethod = "jitter_svr"
    interval: IntervalKind = "hdi"
    n_draws: int = 500
    n_jobs: int = 1
    parallel_backend: str = "thread"
    random_state: Optional[int] = None
    jitter_sd_s: Optional[float] = None
    jitter_fraction_isi: float = 0.10
    min_isi_s: float = 0.020
    trace_noise_sd: Optional[float] = None
    min_discharges: int = DEFAULT_MIN_DISCHARGES
    svr_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "C": 10.0,
        "epsilon": 0.1,
        "gamma": "scale",
        "kernel": "rbf",
    })
    recruitment_windows: Tuple[int, ...] = (1, 5, 11)
    peak_windows: Tuple[int, ...] = (1, 5, 11)
    brace_windows: Tuple[int, ...] = (1, 5, 11)
    store_draws: bool = True
    store_trace_summary: bool = True


@dataclass
class BracePICResult:
    """Structured output of brace-method PIC quantification.

    Attributes
    ----------
    brace_height_norm : float
        Normalized brace height, in percent right-triangle height (% rTri).
    brace_height : float
        Equivalent vertical discharge-rate deviation, in pps.  The normalized
        value is the primary paper-style metric; this pps value is retained for
        interpretability and backward compatibility.
    brace_distance : float
        Orthogonal distance from brace point to hypotenuse in raw plotting units
        (where x is reference and y is pps).  Use cautiously because the axes
        have different physical units.
    right_triangle_height : float
        Raw altitude of the recruitment-peak right triangle.  By construction,
        ``100 * brace_distance / right_triangle_height == brace_height_norm``.
    acceleration_slope, attenuation_slope : float
        Phase slopes in pps per reference unit, e.g. pps/%MVT.
    angle : float
        Reflex angle at the brace point in degrees.  A linear trace is 180 deg;
        larger values indicate stronger bowing.
    recruitment_idx, brace_idx, peak_idx : int
        Indices into the original input arrays.
    peak_reference_idx : int
        Index of peak force/torque used for the paper exclusion check.
    valid : bool
        False if a paper exclusion/inspection criterion was triggered.
    exclusion_reasons : list[str]
        Human-readable reasons for invalid status.
    x, y : np.ndarray
        Analysed recruitment-to-peak segment in reference units and pps.
    time : np.ndarray or None
        Analysed time segment, if a time base or ``fsamp`` was supplied.
    ci : BracePICCI or None
        Optional uncertainty summary when ``ci`` was requested.
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

    brace_distance: float = np.nan
    right_triangle_height: float = np.nan
    peak_reference_idx: Optional[int] = None

    recruitment_reference: float = np.nan
    recruitment_rate: float = np.nan
    brace_reference: float = np.nan
    brace_rate: float = np.nan
    peak_reference: float = np.nan
    peak_rate: float = np.nan
    projection_reference: float = np.nan
    projection_rate: float = np.nan

    x: np.ndarray = field(default=None, repr=False)
    y: np.ndarray = field(default=None, repr=False)
    time: Optional[np.ndarray] = field(default=None, repr=False)
    original_indices: Optional[np.ndarray] = field(default=None, repr=False)

    reference_unit: str = "%MVT"
    discharge_unit: str = "pps"
    phase_fit: str = "endpoints"
    distance_mode: str = "positive"
    endpoint_windows: Dict[str, int] = field(default_factory=dict, repr=False)
    checks: Dict[str, bool] = field(default_factory=dict)
    ci: Optional[BracePICCI] = field(default=None, repr=False)

    @property
    def brace_height_percent_rtri(self) -> float:
        """Alias for ``brace_height_norm``."""
        return self.brace_height_norm

    @property
    def acceleration_slope_pps_per_percent_mvt(self) -> float:
        """Alias used by earlier prototypes."""
        return self.acceleration_slope

    @property
    def attenuation_slope_pps_per_percent_mvt(self) -> float:
        """Alias used by earlier prototypes."""
        return self.attenuation_slope

    @property
    def angle_deg(self) -> float:
        """Alias for ``angle``."""
        return self.angle

    def as_dict(self, *, include_ci: bool = True) -> Dict[str, Any]:
        """Return a flat dictionary suitable for a pandas row."""
        out = {
            "brace_height_norm": self.brace_height_norm,
            "brace_height": self.brace_height,
            "brace_distance": self.brace_distance,
            "right_triangle_height": self.right_triangle_height,
            "acceleration_slope": self.acceleration_slope,
            "attenuation_slope": self.attenuation_slope,
            "angle": self.angle,
            "recruitment_idx": self.recruitment_idx,
            "brace_idx": self.brace_idx,
            "peak_idx": self.peak_idx,
            "peak_reference_idx": self.peak_reference_idx,
            "valid": self.valid,
            "exclusion_reasons": "; ".join(self.exclusion_reasons),
            "recruitment_reference": self.recruitment_reference,
            "recruitment_rate": self.recruitment_rate,
            "brace_reference": self.brace_reference,
            "brace_rate": self.brace_rate,
            "peak_reference": self.peak_reference,
            "peak_rate": self.peak_rate,
            "projection_reference": self.projection_reference,
            "projection_rate": self.projection_rate,
            "reference_unit": self.reference_unit,
            "discharge_unit": self.discharge_unit,
            "phase_fit": self.phase_fit,
            "distance_mode": self.distance_mode,
        }
        if include_ci and self.ci is not None:
            for name, interval in self.ci.intervals.items():
                out[f"{name}_{self.ci.interval}{self.ci.level:g}_lower"] = interval.lower
                out[f"{name}_{self.ci.interval}{self.ci.level:g}_upper"] = interval.upper
                out[f"{name}_draw_mean"] = interval.mean
                out[f"{name}_draw_sd"] = interval.sd
                out[f"{name}_draw_n"] = interval.n
            out["ci_method"] = self.ci.method
            out["ci_interval"] = self.ci.interval
            out["ci_level"] = self.ci.level
            out["ci_n_successful"] = self.ci.n_successful
            out["ci_n_failed"] = self.ci.n_failed
        return out


# Backward-compatible class name.
BraceHeightResult = BracePICResult


def compute_brace_height(
    discharge_rate: np.ndarray,
    reference: np.ndarray,
    *,
    recruitment_idx: Optional[int] = None,
    peak_idx: Optional[int] = None,
    peak_reference_idx: Optional[int] = None,
    fsamp: Optional[float] = None,
    time: Optional[np.ndarray] = None,
    reference_unit: str = "%MVT",
    discharge_unit: str = "pps",
    distance_mode: str = "positive",
    phase_fit: str = "endpoints",
    recruitment_window: int = 1,
    peak_window: int = 1,
    brace_window: int = 1,
    peak_torque_tolerance_s: float = 0.0,
    ci: Union[bool, float] = False,
    ci_options: Optional[Union[CIOptions, Mapping[str, Any]]] = None,
    ci_method: Optional[CIMethod] = None,
    ci_interval: Optional[IntervalKind] = None,
    ci_interval_method: Optional[IntervalKind] = None,
    ci_n_draws: Optional[int] = None,
    ci_n_jobs: Optional[int] = None,
    ci_random_state: Optional[int] = None,
    random_state: Optional[int] = None,
) -> BracePICResult:
    """Compute brace-method PIC metrics for one smoothed MU discharge trace.

    Parameters
    ----------
    discharge_rate : np.ndarray
        Smoothed continuous discharge-rate trace in pps.  NaNs outside the MU's
        active period are allowed and are ignored when no explicit indices are
        supplied.
    reference : np.ndarray
        Reference force/torque trace sampled on the same time base, usually
        percent MVT/MVC.
    recruitment_idx, peak_idx, peak_reference_idx : int, optional
        Indices into the original arrays.  Defaults are: first finite active
        sample, peak discharge after recruitment, and peak reference.
    fsamp : float, optional
        Sampling frequency in Hz.  Used for time axes and tolerance conversion.
    time : np.ndarray, optional
        Explicit time axis in seconds.  If omitted and ``fsamp`` is supplied,
        ``np.arange(n) / fsamp`` is used.
    distance_mode : {"positive", "absolute"}
        ``"positive"`` selects the largest above-line deviation, which is the
        PIC-amplification interpretation.  ``"absolute"`` selects the largest
        magnitude deviation and can flag below-line curvature as a brace.
    phase_fit : {"endpoints", "ols"}
        ``"endpoints"`` uses chords recruitment->brace and brace->peak.
        ``"ols"`` fits least-squares lines to the two phases, including the
        brace sample in both phases.
    recruitment_window, peak_window, brace_window : int
        Optional local averaging windows in samples for endpoint/brace points.
        Defaults to 1 sample, matching direct geometric quantification.
    peak_torque_tolerance_s : float
        Optional tolerance for the "peak discharge after peak torque" check.
        The paper criterion is strict; default is 0.
    ci : bool or float
        ``False`` disables uncertainty estimation.  ``True`` requests the
        default 95% HDI.  A number, e.g. ``95``, requests that interval level.
    ci_options : CIOptions or dict, optional
        Detailed uncertainty options.
    ci_method, ci_interval, ci_interval_method, ci_n_draws, ci_n_jobs,
    ci_random_state, random_state : optional
        Convenience overrides for fields in ``ci_options``.

    Returns
    -------
    BracePICResult
        Structured result with scalar metrics, geometry points, checks, and
        optional uncertainty intervals.
    """
    result = _compute_brace_height_core(
        discharge_rate=discharge_rate,
        reference=reference,
        recruitment_idx=recruitment_idx,
        peak_idx=peak_idx,
        peak_reference_idx=peak_reference_idx,
        fsamp=fsamp,
        time=time,
        reference_unit=reference_unit,
        discharge_unit=discharge_unit,
        distance_mode=distance_mode,
        phase_fit=phase_fit,
        recruitment_window=recruitment_window,
        peak_window=peak_window,
        brace_window=brace_window,
        peak_torque_tolerance_s=peak_torque_tolerance_s,
    )

    opts = _make_ci_options(
        ci,
        ci_options=ci_options,
        method=ci_method,
        interval=ci_interval if ci_interval is not None else ci_interval_method,
        n_draws=ci_n_draws,
        n_jobs=ci_n_jobs,
        random_state=ci_random_state if ci_random_state is not None else random_state,
    )
    if opts is not None:
        result.ci = _compute_ci(
            result,
            full_reference=np.asarray(reference, dtype=np.float64),
            full_discharge=np.asarray(discharge_rate, dtype=np.float64),
            fsamp=fsamp,
            opts=opts,
            core_kwargs={
                "reference_unit": reference_unit,
                "discharge_unit": discharge_unit,
                "distance_mode": distance_mode,
                "phase_fit": phase_fit,
                "recruitment_window": recruitment_window,
                "peak_window": peak_window,
                "brace_window": brace_window,
                "peak_torque_tolerance_s": peak_torque_tolerance_s,
            },
        )

    return result


def pics_brace(*args: Any, **kwargs: Any) -> BracePICResult:
    """Alias for :func:`compute_brace_height`."""
    return compute_brace_height(*args, **kwargs)


def compute_brace_pic(*args: Any, **kwargs: Any) -> BracePICResult:
    """Alias for :func:`compute_brace_height`."""
    return compute_brace_height(*args, **kwargs)


def compute_pic_brace(*args: Any, **kwargs: Any) -> BracePICResult:
    """Alias for :func:`compute_brace_height`."""
    return compute_brace_height(*args, **kwargs)


def brace_height_from_spike_train(
    spikes: np.ndarray,
    reference: np.ndarray,
    fsamp: float,
    *,
    kind: str = "auto",
    smoother: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None,
    n_eval: Optional[int] = None,
    t_eval: Optional[np.ndarray] = None,
    ci: Union[bool, float] = False,
    ci_options: Optional[Union[CIOptions, Mapping[str, Any]]] = None,
    **brace_kwargs: Any,
) -> BracePICResult:
    """Compute brace metrics directly from a MU spike train.

    The function relies on the existing ``discharge_rate.py`` utilities in
    ``hdsemg_shared.motor_unit``; those functions are intentionally not
    duplicated here.

    Parameters
    ----------
    spikes : np.ndarray
        Binary spike train or discharge sample indices.
    reference : np.ndarray
        Reference force/torque trace sampled at ``fsamp``.
    fsamp : float
        Sampling frequency in Hz.
    kind : {"auto", "binary", "indices"}
        Interpretation of ``spikes``.
    smoother : callable, optional
        Custom smoother ``smoother(times, rate, t_eval) -> (t_eval, smooth)``.
        Defaults to ``smooth_discharge_rate_svr``.
    n_eval : int, optional
        Number of evaluation samples between recruitment and derecruitment.
        Defaults to 2048-Hz sampling over the active interval.
    t_eval : np.ndarray, optional
        Explicit evaluation times.
    ci, ci_options
        Forwarded to :func:`compute_brace_height`.
    **brace_kwargs
        Additional arguments forwarded to :func:`compute_brace_height`.
    """
    from .discharge_rate import (
        firing_times_from_binary,
        firing_times_from_indices,
        instantaneous_discharge_rate,
        smooth_discharge_rate_svr,
    )

    spikes = np.asarray(spikes)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 1:
        raise ValueError("reference must be a 1D array.")
    if fsamp <= 0:
        raise ValueError("fsamp must be positive.")

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
        smoother = smooth_discharge_rate_svr

    if t_eval is None:
        if n_eval is None:
            n_eval = max(3, int(round((rate_times[-1] - rate_times[0]) * float(fsamp))) + 1)
        t_eval = np.linspace(rate_times[0], rate_times[-1], int(n_eval))
    else:
        t_eval = np.asarray(t_eval, dtype=np.float64)

    t_eval, smooth_rate = smoother(rate_times, rate, t_eval)
    ref_time = np.arange(reference.size, dtype=np.float64) / float(fsamp)
    ref_on_rate = np.interp(t_eval, ref_time, reference)

    return compute_brace_height(
        smooth_rate,
        ref_on_rate,
        fsamp=fsamp,
        time=t_eval,
        ci=ci,
        ci_options=ci_options,
        **brace_kwargs,
    )


def brace_pic_from_spike_train(*args: Any, **kwargs: Any) -> BracePICResult:
    """Alias for :func:`brace_height_from_spike_train`."""
    return brace_height_from_spike_train(*args, **kwargs)


def pics_brace_openhdemg_all(
    emgfile: Mapping[str, Any],
    *,
    smoothfits: Optional[Any] = None,
    ci: Union[bool, float] = False,
    ci_options: Optional[Union[CIOptions, Mapping[str, Any]]] = None,
    **brace_kwargs: Any,
) -> Tuple[Any, List[BracePICResult]]:
    """Run brace metrics for all MUs in an ``openhdemg`` file-like object.

    If ``smoothfits`` is supplied, it is used directly and no SVR smoothing is
    performed.  This supports validation against externally smoothed traces such
    as manually digitized Fig. 1 discharge-rate curves.

    Parameters
    ----------
    emgfile : mapping
        ``openhdemg`` file object containing at least ``REF_SIGNAL``, ``FSAMP``,
        and either ``NUMBER_OF_MUS`` or ``MUPULSES``.
    smoothfits : pandas.DataFrame or array-like, optional
        Smoothed discharge rates with shape ``(samples, MUs)``.  NaNs outside MU
        activity are allowed.  If omitted, ``openhdemg.library.compute_svr`` is
        called.
    ci, ci_options, **brace_kwargs
        Forwarded to :func:`compute_brace_height`.

    Returns
    -------
    summary_df : pandas.DataFrame
        One row per MU.
    results : list[BracePICResult]
        Structured per-MU results.
    """
    import pandas as pd

    fsamp = float(emgfile["FSAMP"])
    ref = _reference_array_from_openhdemg(emgfile)

    if smoothfits is None:
        import openhdemg.library as emg
        svrfits = emg.compute_svr(emgfile)
        smoothfits = pd.DataFrame(svrfits["gensvr"]).transpose()

    smooth_arr = np.asarray(smoothfits, dtype=np.float64)
    if smooth_arr.ndim == 1:
        smooth_arr = smooth_arr[:, None]
    if smooth_arr.shape[0] != ref.size and smooth_arr.shape[1] == ref.size:
        smooth_arr = smooth_arr.T
    if smooth_arr.shape[0] != ref.size:
        raise ValueError(
            f"smoothfits must have {ref.size} rows to match REF_SIGNAL; "
            f"got shape {smooth_arr.shape}."
        )

    results: List[BracePICResult] = []
    rows: List[Dict[str, Any]] = []
    for mu in range(smooth_arr.shape[1]):
        try:
            res = compute_brace_height(
                smooth_arr[:, mu],
                ref,
                fsamp=fsamp,
                ci=ci,
                ci_options=ci_options,
                **brace_kwargs,
            )
            row = res.as_dict()
            row["mu"] = mu
        except Exception as exc:
            res = None
            row = {"mu": mu, "valid": False, "error": str(exc)}
        results.append(res)
        rows.append(row)

    return pd.DataFrame(rows), results


def plot_brace(
    result: BracePICResult,
    *,
    ax: Optional[Any] = None,
    show_ci: bool = True,
    ci_shadow: str = "sd",
    ci_metric_label: bool = True,
    show_points: bool = True,
    show_scale_bars: bool = True,
    scale_reference: float = 10.0,
    scale_discharge: float = 10.0,
    equal_scale: bool = True,
    title: Optional[str] = None,
    trace_kwargs: Optional[Mapping[str, Any]] = None,
    ci_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Plot brace geometry with optional CI mean and SD/interval shadow.

    Parameters
    ----------
    result : BracePICResult
        Output from :func:`compute_brace_height`.
    ax : matplotlib Axes, optional
        Existing axes.  If omitted, a new figure and axes are created.
    show_ci : bool
        If true and ``result.ci`` has a trace summary, plot the CI draw mean and
        uncertainty shadow.
    ci_shadow : {"sd", "interval"}
        ``"sd"`` plots mean ± one SD.  ``"interval"`` uses the pointwise draw
        interval when draw traces are available; if unavailable, it falls back to
        SD.
    show_scale_bars : bool
        Add 10 %MVT and 10 pps scale bars by default.
    scale_reference, scale_discharge : float
        Scale-bar lengths.  With defaults and ``equal_scale=True``, the plot
        matches the Fig. 1 panel-c convention that 10 %MVT and 10 pps occupy the
        same display length.
    equal_scale : bool
        Set ``ax.set_aspect(scale_reference / scale_discharge)``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    trace_style = {"linewidth": 2.0, "label": "smoothed discharge"}
    if trace_kwargs:
        trace_style.update(trace_kwargs)

    x = np.asarray(result.x, dtype=float)
    y = np.asarray(result.y, dtype=float)

    if show_ci and result.ci is not None and result.ci.trace_reference is not None:
        ci_style = {"alpha": 0.18, "linewidth": 0.0, "label": "CI draw SD"}
        if ci_kwargs:
            ci_style.update(ci_kwargs)
        xr = result.ci.trace_reference
        ym = result.ci.trace_mean
        ys = result.ci.trace_sd
        if ym is not None and ys is not None:
            ax.plot(xr, ym, linewidth=1.2, alpha=0.8, label="CI draw mean")
            ax.fill_between(xr, ym - ys, ym + ys, **ci_style)

    ax.plot(x, y, **trace_style)

    # The brace-height segment is drawn to the raw perpendicular projection on
    # the recruitment-to-peak line.  This makes the plotted segment visibly
    # perpendicular when x and y are displayed with the same data scaling.
    ax.plot(
        [result.recruitment_reference, result.peak_reference],
        [result.recruitment_rate, result.peak_rate],
        linestyle="--",
        linewidth=1.0,
        label="recruitment-peak line",
    )
    ax.plot(
        [result.recruitment_reference, result.brace_reference],
        [result.recruitment_rate, result.brace_rate],
        linewidth=2.0,
        label="acceleration",
    )
    ax.plot(
        [result.brace_reference, result.peak_reference],
        [result.brace_rate, result.peak_rate],
        linewidth=2.0,
        label="attenuation",
    )
    ax.plot(
        [result.brace_reference, result.projection_reference],
        [result.brace_rate, result.projection_rate],
        linewidth=2.0,
        label="brace height",
    )

    if show_points:
        ax.plot(result.recruitment_reference, result.recruitment_rate, "o", markersize=5)
        ax.plot(result.brace_reference, result.brace_rate, "o", markersize=5)
        ax.plot(result.peak_reference, result.peak_rate, "o", markersize=5)

    if ci_metric_label and result.ci is not None:
        label = _format_ci_label(result)
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox={"boxstyle": "round", "alpha": 0.12})

    ax.set_xlabel(f"Reference ({result.reference_unit})")
    ax.set_ylabel(f"Discharge rate ({result.discharge_unit})")
    if title is not None:
        ax.set_title(title)

    if equal_scale:
        ax.set_aspect(float(scale_reference) / float(scale_discharge), adjustable="datalim")

    if show_scale_bars:
        _add_scale_bars(ax, scale_reference, scale_discharge, result.reference_unit, result.discharge_unit)

    return ax


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------


def _compute_brace_height_core(
    *,
    discharge_rate: np.ndarray,
    reference: np.ndarray,
    recruitment_idx: Optional[int],
    peak_idx: Optional[int],
    peak_reference_idx: Optional[int],
    fsamp: Optional[float],
    time: Optional[np.ndarray],
    reference_unit: str,
    discharge_unit: str,
    distance_mode: str,
    phase_fit: str,
    recruitment_window: int,
    peak_window: int,
    brace_window: int,
    peak_torque_tolerance_s: float,
) -> BracePICResult:
    y0_all = np.asarray(discharge_rate, dtype=np.float64).reshape(-1)
    x0_all = np.asarray(reference, dtype=np.float64).reshape(-1)
    if x0_all.shape != y0_all.shape:
        raise ValueError("discharge_rate and reference must be 1D arrays of equal length.")
    if x0_all.ndim != 1:
        raise ValueError("discharge_rate and reference must be 1D arrays.")

    if time is not None:
        t0_all = np.asarray(time, dtype=np.float64).reshape(-1)
        if t0_all.shape != x0_all.shape:
            raise ValueError("time must have the same shape as reference.")
    elif fsamp is not None:
        if fsamp <= 0:
            raise ValueError("fsamp must be positive.")
        t0_all = np.arange(x0_all.size, dtype=np.float64) / float(fsamp)
    else:
        t0_all = None

    valid_idx = np.flatnonzero(np.isfinite(x0_all) & np.isfinite(y0_all))
    if valid_idx.size < 3:
        raise ValueError("Need at least three finite reference/discharge samples.")

    idx_map = valid_idx
    x_all = x0_all[idx_map]
    y_all = y0_all[idx_map]
    t_all = t0_all[idx_map] if t0_all is not None else None

    rec_local = 0 if recruitment_idx is None else _original_to_local_index(idx_map, recruitment_idx)
    if peak_reference_idx is None:
        peak_ref_local = int(np.argmax(x_all))
    else:
        peak_ref_local = _original_to_local_index(idx_map, peak_reference_idx)

    if peak_idx is None:
        peak_local = rec_local + int(np.argmax(y_all[rec_local:]))
    else:
        peak_local = _original_to_local_index(idx_map, peak_idx)

    if not (0 <= rec_local < peak_local < x_all.size):
        raise ValueError(
            "Require recruitment_idx < peak_idx within finite active samples; "
            f"got recruitment local={rec_local}, peak local={peak_local}, n={x_all.size}."
        )

    x = x_all[rec_local: peak_local + 1]
    y = y_all[rec_local: peak_local + 1]
    t = t_all[rec_local: peak_local + 1] if t_all is not None else None
    idx_segment = idx_map[rec_local: peak_local + 1]

    if x.size < 3:
        raise ValueError("Recruitment-to-peak segment must contain at least three samples.")

    recruitment_window = _positive_int(recruitment_window, "recruitment_window")
    peak_window = _positive_int(peak_window, "peak_window")
    brace_window = _positive_int(brace_window, "brace_window")

    x_rec = _window_mean_at_segment_start(x, recruitment_window)
    y_rec = _window_mean_at_segment_start(y, recruitment_window)
    x_peak = _window_mean_at_segment_end(x, peak_window)
    y_peak = _window_mean_at_segment_end(y, peak_window)

    dx = x_peak - x_rec
    dy = y_peak - y_rec
    if dx <= 0:
        raise ValueError("Reference at peak discharge must exceed reference at recruitment.")
    if dy <= 0:
        raise ValueError("Peak discharge rate must exceed recruitment discharge rate.")

    u = (x - x_rec) / dx
    v = (y - y_rec) / dy
    signed_deviation = v - u

    if distance_mode == "positive":
        local_brace = int(np.argmax(signed_deviation))
        max_dev = float(signed_deviation[local_brace])
    elif distance_mode == "absolute":
        local_brace = int(np.argmax(np.abs(signed_deviation)))
        max_dev = float(abs(signed_deviation[local_brace]))
    else:
        raise ValueError("distance_mode must be 'positive' or 'absolute'.")

    brace_idx_original = int(idx_segment[local_brace])
    x_brace = _window_mean_centered(x, local_brace, brace_window)
    y_brace = _window_mean_centered(y, local_brace, brace_window)

    # If a brace averaging window is requested, the averaged brace point is the
    # point used for both phase slopes and the final height.  With the default
    # one-sample window this is exactly the maximum-deviation sample.
    dev_at_brace_point = (y_brace - y_rec) / dy - (x_brace - x_rec) / dx
    max_dev = float(abs(dev_at_brace_point) if distance_mode == "absolute" else dev_at_brace_point)

    brace_height_norm = 100.0 * max_dev
    brace_height_pps = dy * max_dev

    # Raw perpendicular projection for plotting.  The normalized metric is still
    # equivalent to raw orthogonal distance divided by raw right-triangle height.
    projection_reference, projection_rate = _raw_orthogonal_projection(
        (x_rec, y_rec), (x_peak, y_peak), (x_brace, y_brace)
    )
    hypotenuse = math.hypot(dx, dy)
    right_triangle_height = (dx * dy) / hypotenuse if hypotenuse > 0 else np.nan
    brace_distance = abs(_signed_raw_line_distance(
        (x_rec, y_rec), (x_peak, y_peak), (x_brace, y_brace)
    ))

    if phase_fit == "endpoints":
        acceleration_slope = _safe_slope(x_rec, y_rec, x_brace, y_brace)
        attenuation_slope = _safe_slope(x_brace, y_brace, x_peak, y_peak)
    elif phase_fit == "ols":
        acceleration_slope = _ols_slope(x[:local_brace + 1], y[:local_brace + 1])
        attenuation_slope = _ols_slope(x[local_brace:], y[local_brace:])
    else:
        raise ValueError("phase_fit must be 'endpoints' or 'ols'.")

    angle = _reflex_vertex_angle((x_rec, y_rec), (x_brace, y_brace), (x_peak, y_peak))

    if fsamp is not None:
        tol_samples = int(round(float(fsamp) * float(peak_torque_tolerance_s)))
    elif t_all is not None:
        # Convert tolerance from seconds to a conservative index tolerance using
        # the median finite sampling interval.
        dt = np.median(np.diff(t_all)) if t_all.size > 1 else np.inf
        tol_samples = int(round(float(peak_torque_tolerance_s) / dt)) if dt > 0 else 0
    else:
        tol_samples = 0

    peak_idx_original = int(idx_map[peak_local])
    peak_reference_idx_original = int(idx_map[peak_ref_local])

    checks = {
        "negative_acceleration_slope": bool(np.isfinite(acceleration_slope) and acceleration_slope < 0),
        "brace_height_above_200_percent": bool(brace_height_norm > MAX_BRACE_HEIGHT_NORM),
        "peak_discharge_after_peak_torque": bool(peak_idx_original > peak_reference_idx_original + tol_samples),
        "brace_at_edge": bool(local_brace == 0 or local_brace == x.size - 1),
        "no_positive_above_line_brace": bool(distance_mode == "positive" and max_dev <= 0),
        "attenuation_slope_exceeds_acceleration_slope": bool(
            np.isfinite(acceleration_slope) and np.isfinite(attenuation_slope)
            and attenuation_slope > acceleration_slope
        ),
    }

    reasons: List[str] = []
    if checks["negative_acceleration_slope"]:
        reasons.append("negative acceleration slope")
    if checks["brace_height_above_200_percent"]:
        reasons.append(f"normalized brace height > {MAX_BRACE_HEIGHT_NORM:.0f}% rTri")
    if checks["peak_discharge_after_peak_torque"]:
        reasons.append("peak discharge after peak force/torque")

    return BracePICResult(
        brace_height=brace_height_pps,
        brace_height_norm=brace_height_norm,
        brace_distance=brace_distance,
        right_triangle_height=right_triangle_height,
        acceleration_slope=acceleration_slope,
        attenuation_slope=attenuation_slope,
        angle=angle,
        recruitment_idx=int(idx_map[rec_local]),
        brace_idx=brace_idx_original,
        peak_idx=peak_idx_original,
        peak_reference_idx=peak_reference_idx_original,
        valid=len(reasons) == 0,
        exclusion_reasons=reasons,
        recruitment_reference=float(x_rec),
        recruitment_rate=float(y_rec),
        brace_reference=float(x_brace),
        brace_rate=float(y_brace),
        peak_reference=float(x_peak),
        peak_rate=float(y_peak),
        projection_reference=float(projection_reference),
        projection_rate=float(projection_rate),
        x=x,
        y=y,
        time=t,
        original_indices=idx_segment,
        reference_unit=reference_unit,
        discharge_unit=discharge_unit,
        phase_fit=phase_fit,
        distance_mode=distance_mode,
        endpoint_windows={
            "recruitment_window": recruitment_window,
            "peak_window": peak_window,
            "brace_window": brace_window,
        },
        checks=checks,
    )


# ---------------------------------------------------------------------------
# CI/HDI/ETI implementation
# ---------------------------------------------------------------------------


def _compute_ci(
    result: BracePICResult,
    *,
    full_reference: np.ndarray,
    full_discharge: np.ndarray,
    fsamp: Optional[float],
    opts: CIOptions,
    core_kwargs: Mapping[str, Any],
) -> BracePICCI:
    level = float(opts.level)
    if not (0 < level < 100):
        raise ValueError("CI level must be in (0, 100).")
    if opts.interval not in {"hdi", "eti"}:
        raise ValueError("ci interval must be 'hdi' or 'eti'.")
    if opts.method not in {"jitter_svr", "trace_noise", "sensitivity"}:
        raise ValueError("ci method must be 'jitter_svr', 'trace_noise', or 'sensitivity'.")

    if opts.method == "sensitivity":
        draw_records, trace_draws = _sensitivity_draws(result, full_reference, full_discharge, fsamp, opts, core_kwargs)
    else:
        n = int(opts.n_draws)
        if n <= 0:
            raise ValueError("n_draws must be positive.")
        rng = np.random.default_rng(opts.random_state)
        seeds = rng.integers(0, np.iinfo(np.uint32).max, size=n, dtype=np.uint32)
        worker_args = [
            (
                int(seed),
                opts.method,
                result.x,
                result.y,
                result.time,
                fsamp,
                opts,
                dict(core_kwargs),
            )
            for seed in seeds
        ]
        outputs = _parallel_map(_ci_draw_worker, worker_args, opts.n_jobs, opts.parallel_backend)
        draw_records = [out[0] for out in outputs if out is not None and out[0] is not None]
        trace_draws = [out[1] for out in outputs if out is not None and out[1] is not None]

    metric_points = {
        "brace_height_norm": result.brace_height_norm,
        "brace_height": result.brace_height,
        "acceleration_slope": result.acceleration_slope,
        "attenuation_slope": result.attenuation_slope,
        "angle": result.angle,
    }

    draws: Dict[str, np.ndarray] = {}
    intervals: Dict[str, MetricInterval] = {}
    for name, point in metric_points.items():
        arr = np.asarray([rec.get(name, np.nan) for rec in draw_records], dtype=float)
        arr = arr[np.isfinite(arr)]
        draws[name] = arr if opts.store_draws else np.asarray([], dtype=float)
        intervals[name] = _metric_interval(arr, point, level, opts.interval)

    trace_mean = trace_sd = trace_reference = None
    if opts.store_trace_summary and len(trace_draws) > 1:
        min_len = min(len(z) for z in trace_draws)
        trace_matrix = np.vstack([np.asarray(z[:min_len], dtype=float) for z in trace_draws])
        trace_mean = np.nanmean(trace_matrix, axis=0)
        trace_sd = np.nanstd(trace_matrix, axis=0, ddof=1)
        trace_reference = result.x[:min_len]

    n_requested = int(opts.n_draws) if opts.method != "sensitivity" else len(draw_records)

    return BracePICCI(
        level=level,
        method=opts.method,
        interval=opts.interval,
        intervals=intervals,
        draws=draws,
        trace_reference=trace_reference,
        trace_mean=trace_mean,
        trace_sd=trace_sd,
        n_requested=n_requested,
        n_successful=len(draw_records),
        n_failed=max(0, n_requested - len(draw_records)),
        options=_ci_options_as_dict(opts),
    )


def _ci_draw_worker(args: Tuple[Any, ...]) -> Tuple[Optional[Dict[str, float]], Optional[np.ndarray]]:
    seed, method, ref, dr, time, fsamp, opts, core_kwargs = args
    rng = np.random.default_rng(seed)
    ref = np.asarray(ref, dtype=float)
    dr = np.asarray(dr, dtype=float)
    time_arr = _time_for_ci(time, ref.size, fsamp)

    try:
        if method == "jitter_svr":
            draw_y = _jitter_svr_trace(ref, dr, time_arr, fsamp, opts, rng)
        elif method == "trace_noise":
            sd = opts.trace_noise_sd if opts.trace_noise_sd is not None else _robust_trace_noise_sd(dr)
            draw_y = dr + rng.normal(0.0, sd, size=dr.size)
        else:
            return None, None

        draw_result = _compute_brace_height_core(
            discharge_rate=draw_y,
            reference=ref,
            recruitment_idx=0,
            peak_idx=None,
            peak_reference_idx=None,
            fsamp=fsamp,
            time=time_arr,
            **core_kwargs,
        )
        return _result_metric_record(draw_result), draw_y
    except Exception:
        return None, None


def _jitter_svr_trace(
    ref: np.ndarray,
    dr: np.ndarray,
    time_arr: np.ndarray,
    fsamp: Optional[float],
    opts: CIOptions,
    rng: np.random.Generator,
) -> np.ndarray:
    from .discharge_rate import instantaneous_discharge_rate, smooth_discharge_rate_svr

    spike_times = _infer_spike_times_from_smoothed_rate(time_arr, dr)
    if spike_times.size < opts.min_discharges:
        raise ValueError("Too few inferred spikes for jitter-SVR CI.")

    local_rate = np.interp(spike_times, time_arr, np.maximum(dr, 1e-6))
    if opts.jitter_sd_s is None:
        jitter_sd = opts.jitter_fraction_isi / np.maximum(local_rate, 1e-6)
    else:
        jitter_sd = np.full_like(spike_times, float(opts.jitter_sd_s))

    jittered = spike_times + rng.normal(0.0, jitter_sd)
    jittered = _regularize_spike_times(jittered, time_arr[0], time_arr[-1], opts.min_isi_s)
    rate_times, rate = instantaneous_discharge_rate(jittered, min_discharges=opts.min_discharges)
    _, smooth = smooth_discharge_rate_svr(rate_times, rate, time_arr, **opts.svr_kwargs)
    return np.asarray(smooth, dtype=float)


def _sensitivity_draws(
    result: BracePICResult,
    full_reference: np.ndarray,
    full_discharge: np.ndarray,
    fsamp: Optional[float],
    opts: CIOptions,
    core_kwargs: Mapping[str, Any],
) -> Tuple[List[Dict[str, float]], List[np.ndarray]]:
    records: List[Dict[str, float]] = []
    traces: List[np.ndarray] = []
    for rw in opts.recruitment_windows:
        for pw in opts.peak_windows:
            for bw in opts.brace_windows:
                kwargs = dict(core_kwargs)
                kwargs["recruitment_window"] = int(rw)
                kwargs["peak_window"] = int(pw)
                kwargs["brace_window"] = int(bw)
                try:
                    res = _compute_brace_height_core(
                        discharge_rate=result.y,
                        reference=result.x,
                        recruitment_idx=0,
                        peak_idx=None,
                        peak_reference_idx=None,
                        fsamp=fsamp,
                        time=result.time,
                        **kwargs,
                    )
                    records.append(_result_metric_record(res))
                    traces.append(result.y.copy())
                except Exception:
                    continue
    return records, traces


def _make_ci_options(
    ci: Union[bool, float],
    *,
    ci_options: Optional[Union[CIOptions, Mapping[str, Any]]],
    method: Optional[CIMethod],
    interval: Optional[IntervalKind],
    n_draws: Optional[int],
    n_jobs: Optional[int],
    random_state: Optional[int],
) -> Optional[CIOptions]:
    if ci is False or ci is None:
        return None

    if isinstance(ci_options, CIOptions):
        opts = replace(ci_options)
    elif isinstance(ci_options, Mapping):
        opts = CIOptions(**_coerce_ci_options_mapping(ci_options))
    else:
        opts = CIOptions()

    if ci is True:
        opts.level = 95.0
    else:
        opts.level = float(ci)
    if 0.0 < opts.level <= 1.0:
        opts.level *= 100.0

    if method is not None:
        opts.method = _normalize_ci_method(method)
    if interval is not None:
        opts.interval = _normalize_ci_interval(interval)
    if n_draws is not None:
        opts.n_draws = int(n_draws)
    if n_jobs is not None:
        opts.n_jobs = int(n_jobs)
    if random_state is not None:
        opts.random_state = int(random_state)
    opts.method = _normalize_ci_method(opts.method)
    opts.interval = _normalize_ci_interval(opts.interval)

    return opts


def _coerce_ci_options_mapping(ci_options: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(ci_options)
    aliases = {
        "ci_interval_method": "interval",
        "interval_kind": "interval",
        "jitter_cv": "jitter_fraction_isi",
        "refractory_s": "min_isi_s",
        "start_avg_samples": "recruitment_windows",
        "peak_avg_samples": "peak_windows",
        "brace_avg_samples": "brace_windows",
    }
    for old, new in aliases.items():
        if old in data and new not in data:
            data[new] = data[old]

    svr_kwargs = dict(data.get("svr_kwargs", {}))
    if "svr_C" in data and "C" not in svr_kwargs:
        svr_kwargs["C"] = _first_option(data["svr_C"])
    if "svr_epsilon" in data and "epsilon" not in svr_kwargs:
        svr_kwargs["epsilon"] = _first_option(data["svr_epsilon"])
    if "svr_gamma" in data and "gamma" not in svr_kwargs:
        svr_kwargs["gamma"] = _first_option(data["svr_gamma"])
    if "svr_kernel" in data and "kernel" not in svr_kwargs:
        svr_kwargs["kernel"] = _first_option(data["svr_kernel"])
    if svr_kwargs:
        data["svr_kwargs"] = svr_kwargs

    allowed = set(CIOptions.__dataclass_fields__)
    out = {key: value for key, value in data.items() if key in allowed}
    if "method" in out:
        out["method"] = _normalize_ci_method(out["method"])
    if "interval" in out:
        out["interval"] = _normalize_ci_interval(out["interval"])
    return out


def _first_option(value: Any) -> Any:
    if isinstance(value, (str, bytes)):
        return value
    try:
        return next(iter(value))
    except TypeError:
        return value
    except StopIteration:
        return value


def _normalize_ci_method(method: str) -> str:
    method = str(method)
    if method == "bayesian_jitter_svr":
        return "jitter_svr"
    return method


def _normalize_ci_interval(interval: str) -> str:
    return str(interval).lower()


def _metric_interval(values: np.ndarray, point: float, level: float, interval: str) -> MetricInterval:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return MetricInterval(point, np.nan, np.nan, np.nan, np.nan, level, interval, 0)

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    if values.size == 1:
        lower = upper = float(values[0])
    elif interval == "eti":
        alpha = (100.0 - level) / 2.0
        lower, upper = np.percentile(values, [alpha, 100.0 - alpha])
    else:
        lower, upper = _hdi(values, level / 100.0)

    return MetricInterval(
        point=float(point),
        mean=mean,
        sd=sd,
        lower=float(lower),
        upper=float(upper),
        level=float(level),
        interval=interval,
        n=int(values.size),
    )


def _hdi(values: np.ndarray, mass: float) -> Tuple[float, float]:
    values = np.sort(np.asarray(values, dtype=float))
    n = values.size
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(values[0]), float(values[0])
    k = int(np.floor(mass * n))
    k = min(max(k, 1), n - 1)
    widths = values[k:] - values[: n - k]
    j = int(np.argmin(widths))
    return float(values[j]), float(values[j + k])


def _parallel_map(func: Callable[[Any], Any], args: Sequence[Any], n_jobs: int, backend: str) -> List[Any]:
    if n_jobs is None or int(n_jobs) == 1 or len(args) <= 1:
        return [func(arg) for arg in args]
    workers = os.cpu_count() if int(n_jobs) == -1 else int(n_jobs)
    workers = max(1, min(int(workers), len(args)))
    if backend == "process":
        executor_cls = _futures.ProcessPoolExecutor
    elif backend == "thread":
        executor_cls = _futures.ThreadPoolExecutor
    else:
        raise ValueError("parallel_backend must be 'thread' or 'process'.")
    with executor_cls(max_workers=workers) as ex:
        return list(ex.map(func, args))


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _original_to_local_index(idx_map: np.ndarray, original_idx: int) -> int:
    matches = np.flatnonzero(idx_map == int(original_idx))
    if matches.size == 0:
        raise ValueError(f"Index {original_idx} is not finite/available in the analysed trace.")
    return int(matches[0])


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
    return value


def _window_mean_at_segment_start(a: np.ndarray, window: int) -> float:
    return float(np.mean(a[: min(window, a.size)]))


def _window_mean_at_segment_end(a: np.ndarray, window: int) -> float:
    return float(np.mean(a[max(0, a.size - window):]))


def _window_mean_centered(a: np.ndarray, center: int, window: int) -> float:
    half = window // 2
    start = max(0, int(center) - half)
    stop = min(a.size, int(center) + (window - half))
    return float(np.mean(a[start:stop]))


def _safe_slope(x0: float, y0: float, x1: float, y1: float) -> float:
    dx = x1 - x0
    return float((y1 - y0) / dx) if dx != 0 else np.nan


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.ptp(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, deg=1)[0])


def _reflex_vertex_angle(rec: Tuple[float, float], brace: Tuple[float, float], peak: Tuple[float, float]) -> float:
    ba = np.array([rec[0] - brace[0], rec[1] - brace[1]], dtype=float)
    bc = np.array([peak[0] - brace[0], peak[1] - brace[1]], dtype=float)
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return np.nan
    cos_theta = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    interior = math.degrees(math.acos(float(cos_theta)))
    return float(360.0 - interior)


def _raw_orthogonal_projection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
) -> Tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return np.nan, np.nan
    t = ((x3 - x1) * dx + (y3 - y1) * dy) / denom
    return float(x1 + t * dx), float(y1 + t * dy)


def _signed_raw_line_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
) -> float:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy)
    if denom == 0:
        return np.nan
    # Positive when p3 lies above the p1->p2 line for an increasing ramp.
    return float((dx * (y3 - y1) - dy * (x3 - x1)) / denom)


def _time_for_ci(time: Optional[np.ndarray], n: int, fsamp: Optional[float]) -> np.ndarray:
    if time is not None:
        out = np.asarray(time, dtype=float)
        if out.size == n:
            return out
    if fsamp is None:
        return np.arange(n, dtype=float)
    return np.arange(n, dtype=float) / float(fsamp)


def _infer_spike_times_from_smoothed_rate(time_arr: np.ndarray, rate: np.ndarray) -> np.ndarray:
    time_arr = np.asarray(time_arr, dtype=float)
    rate = np.maximum(np.asarray(rate, dtype=float), 1e-6)
    if time_arr.size != rate.size or time_arr.size < 2:
        return np.asarray([], dtype=float)
    dt = np.diff(time_arr)
    # Trapezoidal integral of pps gives expected spike count.
    increments = 0.5 * (rate[:-1] + rate[1:]) * dt
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    n_events = int(np.floor(cumulative[-1]))
    if n_events < 1:
        return np.asarray([], dtype=float)
    levels = np.arange(1, n_events + 1, dtype=float)
    return np.interp(levels, cumulative, time_arr)


def _regularize_spike_times(times: np.ndarray, t_min: float, t_max: float, min_isi_s: float) -> np.ndarray:
    times = np.sort(np.clip(np.asarray(times, dtype=float), t_min, t_max))
    if times.size == 0:
        return times
    kept = [float(times[0])]
    for t in times[1:]:
        if t - kept[-1] >= min_isi_s:
            kept.append(float(t))
    return np.asarray(kept, dtype=float)


def _robust_trace_noise_sd(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return max(1e-6, float(np.nanstd(y)) * 0.05)
    d2 = np.diff(y, n=2)
    med = np.nanmedian(d2)
    mad = np.nanmedian(np.abs(d2 - med))
    # Var(second difference of iid noise) = 6 sigma^2.
    sd = (mad / 0.6744897501960817) / math.sqrt(6.0) if mad > 0 else np.nan
    if not np.isfinite(sd) or sd <= 0:
        sd = max(1e-6, float(np.nanstd(y)) * 0.02)
    return float(sd)


def _result_metric_record(res: BracePICResult) -> Dict[str, float]:
    return {
        "brace_height_norm": float(res.brace_height_norm),
        "brace_height": float(res.brace_height),
        "acceleration_slope": float(res.acceleration_slope),
        "attenuation_slope": float(res.attenuation_slope),
        "angle": float(res.angle),
    }


def _ci_options_as_dict(opts: CIOptions) -> Dict[str, Any]:
    out = dict(opts.__dict__)
    out["svr_kwargs"] = dict(opts.svr_kwargs)
    return out


def _format_ci_label(result: BracePICResult) -> str:
    lines = [
        f"BH={result.brace_height_norm:.1f} %rTri",
        f"ACC={result.acceleration_slope:.2f}",
        f"ATT={result.attenuation_slope:.2f}",
        f"ANG={result.angle:.0f} deg",
    ]
    if result.ci is not None and "brace_height_norm" in result.ci.intervals:
        iv = result.ci.intervals["brace_height_norm"]
        lines.append(f"BH {result.ci.level:g}% {result.ci.interval}: [{iv.lower:.1f}, {iv.upper:.1f}]")
    return "\n".join(lines)


def _add_scale_bars(ax: Any, scale_x: float, scale_y: float, x_unit: str, y_unit: str) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + 0.08 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.10 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + scale_x], [y0, y0], color="black", linewidth=2)
    ax.plot([x0, x0], [y0, y0 + scale_y], color="black", linewidth=2)
    ax.text(x0 + scale_x / 2, y0 - 0.04 * (ylim[1] - ylim[0]), f"{scale_x:g} {x_unit}",
            ha="center", va="top", fontsize=9)
    ax.text(x0 - 0.03 * (xlim[1] - xlim[0]), y0 + scale_y / 2, f"{scale_y:g} {y_unit}",
            ha="right", va="center", rotation=90, fontsize=9)


def _reference_array_from_openhdemg(emgfile: Mapping[str, Any]) -> np.ndarray:
    ref = emgfile["REF_SIGNAL"]
    try:
        # pandas DataFrame or Series
        if hasattr(ref, "iloc"):
            if getattr(ref, "ndim", 1) == 2:
                return ref.iloc[:, 0].to_numpy(dtype=float)
            return ref.to_numpy(dtype=float)
    except Exception:
        pass
    arr = np.asarray(ref, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr.reshape(-1)


__all__ = [
    "MAX_BRACE_HEIGHT_NORM",
    "MetricInterval",
    "BracePICCI",
    "CIOptions",
    "BracePICResult",
    "BraceHeightResult",
    "compute_brace_height",
    "compute_brace_pic",
    "pics_brace",
    "compute_pic_brace",
    "brace_height_from_spike_train",
    "brace_pic_from_spike_train",
    "pics_brace_openhdemg_all",
    "plot_brace",
]
