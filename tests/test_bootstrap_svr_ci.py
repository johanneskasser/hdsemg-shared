import numpy as np
import pytest

from hdsemg_shared.motor_unit.brace_pic import CIOptions, compute_brace_pic
from hdsemg_shared.motor_unit.discharge_rate import smooth_discharge_rate_svr


def _synthetic_bowed_trace(n_idr=40, noise_sd=0.3, rng_seed=42):
    """IDR with known upward bow (simulates PIC amplification) plus noise."""
    rng = np.random.default_rng(rng_seed)
    rate_times = np.linspace(0.5, 4.5, n_idr)
    t_norm = (rate_times - rate_times[0]) / (rate_times[-1] - rate_times[0])
    idr_clean = 8.0 + 12.0 * t_norm + 5.0 * np.sin(np.pi * t_norm)
    idr = idr_clean + rng.normal(0.0, noise_sd, size=n_idr)
    return rate_times, idr


def _build_trace(rate_times, idr, n_eval=512, svr_kwargs=None):
    svr_kwargs = svr_kwargs or {"C": 10.0, "gamma": "scale"}
    t_eval = np.linspace(rate_times[0], rate_times[-1], n_eval)
    _, smooth = smooth_discharge_rate_svr(rate_times, idr, t_eval, **svr_kwargs)
    ref = np.linspace(5.0, 30.0, n_eval)
    peak_ref_idx = int(np.argmax(ref))
    return t_eval, np.asarray(smooth, dtype=float), ref, peak_ref_idx


def test_bootstrap_svr_produces_finite_intervals():
    rate_times, idr = _synthetic_bowed_trace()
    t_eval, smooth, ref, peak_ref_idx = _build_trace(rate_times, idr)

    opts = CIOptions(
        method="bootstrap_svr",
        n_draws=200,
        n_jobs=1,
        random_state=99,
        bootstrap_rate_times=rate_times,
        bootstrap_rate_values=idr,
    )
    result = compute_brace_pic(
        smooth, ref, time=t_eval, peak_reference_idx=peak_ref_idx, ci=95, ci_options=opts,
    )
    assert result.ci is not None
    assert result.ci.n_successful > 0
    for name in ("brace_height_norm", "brace_height", "acceleration_slope", "attenuation_slope", "angle"):
        iv = result.ci.intervals[name]
        assert np.isfinite(iv.lower), f"{name} lower not finite"
        assert np.isfinite(iv.upper), f"{name} upper not finite"
        assert iv.lower <= iv.point <= iv.upper, f"{name}: {iv.lower} <= {iv.point} <= {iv.upper}"


def test_bootstrap_svr_draw_mean_near_deterministic():
    """Draw mean should be approximately unbiased w.r.t. deterministic BH."""
    rate_times, idr = _synthetic_bowed_trace()
    svr_kwargs = {"C": 10.0, "gamma": "scale"}
    t_eval, smooth, ref, peak_ref_idx = _build_trace(rate_times, idr, svr_kwargs=svr_kwargs)

    det = compute_brace_pic(smooth, ref, time=t_eval, peak_reference_idx=peak_ref_idx)

    opts = CIOptions(
        method="bootstrap_svr",
        n_draws=500,
        n_jobs=1,
        random_state=42,
        bootstrap_rate_times=rate_times,
        bootstrap_rate_values=idr,
        svr_kwargs=svr_kwargs,
    )
    bs = compute_brace_pic(
        smooth, ref, time=t_eval, peak_reference_idx=peak_ref_idx, ci=95, ci_options=opts,
    )
    # ponytail: 30% tolerance — bootstrap mean should be much closer than
    # jitter-SVR's systematic −20..−30% bias
    assert abs(bs.brace_height_norm - det.brace_height_norm) / max(abs(det.brace_height_norm), 1e-6) < 0.30


def test_bootstrap_svr_missing_arrays_raises():
    rate_times, idr = _synthetic_bowed_trace()
    t_eval, smooth, ref, peak_ref_idx = _build_trace(rate_times, idr)

    opts = CIOptions(method="bootstrap_svr", n_draws=10, n_jobs=1)
    with pytest.raises(ValueError, match="bootstrap_svr requires"):
        compute_brace_pic(
            smooth, ref, time=t_eval, peak_reference_idx=peak_ref_idx, ci=95, ci_options=opts,
        )
