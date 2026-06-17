"""
Discharge-rate utilities for decomposed motor-unit (MU) spike trains.

A decomposed MUAP spike train can be expressed either as a binary array (one
sample per recording instant, ``1`` where the MU discharged) or as the list of
discharge instants (sample indices or times). This module converts those
representations into the *instantaneous discharge rate* (the reciprocal of the
inter-spike interval, ISI) and provides an optional Support Vector Regression
(SVR) smoother to obtain a continuous discharge-rate trace, as used by
Beauchamp et al. (2023) prior to brace-height quantification.

References:
- Beauchamp et al. (2023), *J. Neural Eng.* 20 016034 — §2.3.1 (pre-processing).
- Beauchamp et al. (2022) — SVR smoothing of MU discharge rate.

Usage:
>>> times = firing_times_from_binary(spike_train, fsamp=2048)
>>> rate_times, rate = instantaneous_discharge_rate(times)
>>> smooth = smooth_discharge_rate_svr(rate_times, rate, t_eval)  # optional, needs sklearn
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

#: Minimum number of consecutive discharges required for a usable MU
#: (Beauchamp et al. 2023 excluded MUs with fewer than ten discharges).
MIN_DISCHARGES = 10


def firing_times_from_binary(spike_train: np.ndarray, fsamp: float) -> np.ndarray:
    """
    Convert a binary spike train into discharge times in seconds.

    Parameters
    ----------
    spike_train : np.ndarray
        One-dimensional binary array (non-zero where the MU discharged).
    fsamp : float
        Sampling frequency of the spike train in Hz.

    Returns
    -------
    np.ndarray
        Sorted discharge times in seconds.

    Raises
    ------
    ValueError
        If ``spike_train`` is not one-dimensional or ``fsamp`` is not positive.
    """
    spike_train = np.asarray(spike_train)
    if spike_train.ndim != 1:
        raise ValueError("spike_train must be a 1D array.")
    if fsamp <= 0:
        raise ValueError("fsamp must be a positive number.")
    indices = np.flatnonzero(spike_train)
    return indices.astype(np.float64) / float(fsamp)


def firing_times_from_indices(indices: np.ndarray, fsamp: float) -> np.ndarray:
    """
    Convert discharge sample indices into discharge times in seconds.

    Parameters
    ----------
    indices : np.ndarray
        One-dimensional array of discharge sample indices.
    fsamp : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Sorted discharge times in seconds.

    Raises
    ------
    ValueError
        If ``indices`` is not one-dimensional or ``fsamp`` is not positive.
    """
    indices = np.asarray(indices)
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D array.")
    if fsamp <= 0:
        raise ValueError("fsamp must be a positive number.")
    return np.sort(indices.astype(np.float64)) / float(fsamp)


def instantaneous_discharge_rate(
    firing_times: np.ndarray,
    min_discharges: int = MIN_DISCHARGES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the instantaneous discharge rate from discharge times.

    The instantaneous discharge rate is the reciprocal of the inter-spike
    interval (ISI) and is, by convention, assigned to the time of the *later*
    spike of each pair (Beauchamp et al. 2023).

    Parameters
    ----------
    firing_times : np.ndarray
        One-dimensional array of discharge times in seconds (need not be sorted).
    min_discharges : int, optional
        Minimum number of discharges required. MUs with fewer discharges are
        rejected (default: ``MIN_DISCHARGES`` = 10).

    Returns
    -------
    times : np.ndarray
        Times (s) at which each discharge-rate sample is defined (later spike of
        each ISI pair); length ``len(firing_times) - 1``.
    rate : np.ndarray
        Instantaneous discharge rate in pulses per second (pps).

    Raises
    ------
    ValueError
        If ``firing_times`` is not one-dimensional, has fewer than
        ``min_discharges`` entries, or contains non-increasing times (ISI <= 0).
    """
    firing_times = np.asarray(firing_times, dtype=np.float64)
    if firing_times.ndim != 1:
        raise ValueError("firing_times must be a 1D array.")
    if firing_times.size < min_discharges:
        raise ValueError(
            f"MU has {firing_times.size} discharges, fewer than the required "
            f"minimum of {min_discharges}; excluded from analysis."
        )
    times = np.sort(firing_times)
    isi = np.diff(times)
    if np.any(isi <= 0):
        raise ValueError("Discharge times must be strictly increasing (ISI > 0).")
    rate = 1.0 / isi
    return times[1:], rate


def smooth_discharge_rate_svr(
    times: np.ndarray,
    rate: np.ndarray,
    t_eval: Optional[np.ndarray] = None,
    *,
    C: float = 10.0,
    epsilon: float = 0.1,
    gamma="scale",
    kernel: str = "rbf",
    **svr_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth an instantaneous discharge-rate trace with Support Vector Regression.

    This reproduces the smoothing step of Beauchamp et al. (2023), who trained an
    SVR model (MATLAB ``fitrsvm``) to predict discharge rate as a function of
    time. It requires :mod:`scikit-learn`, which is *not* a hard dependency of
    ``hdsemg_shared``; it is imported lazily so that the rest of the package
    works without it.

    Parameters
    ----------
    times : np.ndarray
        Times (s) of the instantaneous discharge-rate samples.
    rate : np.ndarray
        Instantaneous discharge rate (pps) at ``times``.
    t_eval : np.ndarray, optional
        Times (s) at which to evaluate the smoothed trace. Defaults to a dense
        linear grid spanning ``[times.min(), times.max()]`` (200 points).
    C, epsilon, gamma, kernel : SVR hyperparameters
        Passed to :class:`sklearn.svm.SVR`. The defaults are reasonable starting
        points; tune them for your data as recommended by Beauchamp et al.
    **svr_kwargs
        Additional keyword arguments forwarded to :class:`sklearn.svm.SVR`.

    Returns
    -------
    t_eval : np.ndarray
        The evaluation times.
    smooth_rate : np.ndarray
        Smoothed discharge rate (pps) at ``t_eval``.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    ValueError
        If ``times`` and ``rate`` have mismatched shapes.
    """
    try:
        from sklearn.svm import SVR
    except ImportError as exc:  # pragma: no cover - exercised only without sklearn
        raise ImportError(
            "smooth_discharge_rate_svr requires scikit-learn. "
            "Install it with `pip install scikit-learn`, or supply your own "
            "pre-smoothed discharge-rate trace to the brace-height functions."
        ) from exc

    times = np.asarray(times, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    if times.shape != rate.shape or times.ndim != 1:
        raise ValueError("times and rate must be 1D arrays of equal length.")

    if t_eval is None:
        t_eval = np.linspace(times.min(), times.max(), 200)
    t_eval = np.asarray(t_eval, dtype=np.float64)

    model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel, **svr_kwargs)
    model.fit(times.reshape(-1, 1), rate)
    smooth_rate = model.predict(t_eval.reshape(-1, 1))
    return t_eval, smooth_rate
