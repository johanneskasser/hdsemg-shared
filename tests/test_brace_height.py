import numpy as np
import pytest

from hdsemg_shared.motor_unit import (
    BraceHeightResult,
    brace_height_from_spike_train,
    compute_brace_height,
    firing_times_from_binary,
    firing_times_from_indices,
    instantaneous_discharge_rate,
)


# ---------------------------------------------------------------------------
# Brace-height geometry
# ---------------------------------------------------------------------------

def test_linear_trace_has_zero_brace_height():
    x = np.linspace(0.0, 30.0, 100)
    y = np.linspace(5.0, 25.0, 100)
    res = compute_brace_height(y, x)
    assert res.brace_height_norm == pytest.approx(0.0, abs=1e-9)
    assert res.brace_height == pytest.approx(0.0, abs=1e-9)
    assert res.valid


def test_right_triangle_corner_is_about_100_percent():
    # Discharge jumps to peak almost immediately -> trace hugs the right-triangle
    # corner -> normalized brace height approaches 100 % rTri.
    n = 1000
    x = np.linspace(0.0, 30.0, n)
    y = np.full(n, 25.0)
    y[0] = 5.0
    res = compute_brace_height(y, x, peak_idx=n - 1)
    assert res.brace_height_norm == pytest.approx(100.0, abs=0.5)


def test_known_two_segment_trace_matches_hand_computation():
    # rec=(0,0), brace=(1,3), peak=(2,4); dx=2, dy=4.
    # normalized deviations: [0, 0.25, 0] -> bh_norm = 25 %, brace at index 1.
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 3.0, 4.0])
    res = compute_brace_height(y, x)

    assert res.brace_idx == 1
    assert res.brace_height_norm == pytest.approx(25.0)
    assert res.brace_height == pytest.approx(1.0)            # dy * 0.25
    assert res.acceleration_slope == pytest.approx(3.0)      # (3-0)/(1-0)
    assert res.attenuation_slope == pytest.approx(1.0)       # (4-3)/(2-1)
    assert res.angle == pytest.approx(206.565, abs=1e-2)     # reflex vertex angle
    assert res.valid


def test_concave_down_trace_brace_point_is_interior():
    x = np.linspace(0.0, 10.0, 200)
    # sqrt-shaped: steep rise then attenuation, bowing above the chord.
    y = 20.0 * np.sqrt(x / 10.0) + 5.0
    res = compute_brace_height(y, x)
    assert 0 < res.brace_idx < res.peak_idx
    assert res.brace_height_norm > 0
    assert res.acceleration_slope > res.attenuation_slope > 0
    assert res.angle > 180.0
    assert res.valid


# ---------------------------------------------------------------------------
# Exclusion criteria
# ---------------------------------------------------------------------------

def test_exclusion_over_200_percent_and_negative_acceleration():
    # Reference dips well below its recruitment value -> deviation > 200 % rTri
    # and the recruitment->brace chord has a negative slope.
    x = np.array([5.0, -6.0, 2.0, 10.0])
    y = np.array([0.0, 8.0, 9.0, 10.0])
    res = compute_brace_height(y, x)
    assert not res.valid
    assert any("200" in r for r in res.exclusion_reasons)
    assert "negative acceleration slope" in res.exclusion_reasons


def test_exclusion_peak_discharge_after_peak_torque():
    x = np.array([0.0, 1.0, 2.0, 1.5, 1.0])   # torque peaks at index 2
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])   # discharge peaks at index 4
    res = compute_brace_height(y, x)
    assert res.peak_idx == 4
    assert not res.valid
    assert "peak discharge after peak force/torque" in res.exclusion_reasons


def test_degenerate_segment_raises():
    with pytest.raises(ValueError):
        compute_brace_height(np.array([1.0, 2.0]), np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# Discharge-rate utilities
# ---------------------------------------------------------------------------

def test_firing_times_from_binary():
    train = np.zeros(10)
    train[[2, 5, 9]] = 1
    times = firing_times_from_binary(train, fsamp=100.0)
    np.testing.assert_allclose(times, [0.02, 0.05, 0.09])


def test_instantaneous_discharge_rate_values_and_alignment():
    times = np.arange(12) * 0.1  # constant ISI of 0.1 s -> 10 pps
    rate_times, rate = instantaneous_discharge_rate(times)
    assert rate_times.size == 11
    np.testing.assert_allclose(rate, 10.0)
    # rate is assigned to the later spike of each pair
    np.testing.assert_allclose(rate_times, times[1:])


def test_too_few_discharges_raises():
    with pytest.raises(ValueError):
        instantaneous_discharge_rate(np.arange(5) * 0.1)


def test_non_increasing_times_raises():
    with pytest.raises(ValueError):
        instantaneous_discharge_rate(np.array([0.0, 0.1, 0.1, 0.2] + [0.3] * 8))


# ---------------------------------------------------------------------------
# End-to-end wrapper, both input forms
# ---------------------------------------------------------------------------

def _linear_smoother(times, rate, t_eval):
    return t_eval, np.interp(t_eval, times, rate)


def _build_spikes():
    # Decreasing ISIs -> increasing discharge rate (PIC-like onset).
    isi = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 2, 2])
    indices = np.concatenate(([0], np.cumsum(isi)))  # 13 discharges
    return indices


def test_binary_and_indices_give_identical_results():
    fsamp = 100.0
    indices = _build_spikes()
    length = int(indices[-1]) + 1
    reference = np.linspace(0.0, 30.0, length)

    train = np.zeros(length)
    train[indices] = 1

    res_bin = brace_height_from_spike_train(
        train, reference, fsamp, kind="binary", smoother=_linear_smoother
    )
    res_idx = brace_height_from_spike_train(
        indices, reference, fsamp, kind="indices", smoother=_linear_smoother
    )

    assert isinstance(res_bin, BraceHeightResult)
    assert res_bin.brace_height_norm == pytest.approx(res_idx.brace_height_norm)
    assert res_bin.acceleration_slope == pytest.approx(res_idx.acceleration_slope)
    assert res_bin.attenuation_slope == pytest.approx(res_idx.attenuation_slope)


def test_auto_kind_detection_matches_explicit():
    fsamp = 100.0
    indices = _build_spikes()
    length = int(indices[-1]) + 1
    reference = np.linspace(0.0, 30.0, length)

    res_auto = brace_height_from_spike_train(
        indices, reference, fsamp, smoother=_linear_smoother
    )
    res_idx = brace_height_from_spike_train(
        indices, reference, fsamp, kind="indices", smoother=_linear_smoother
    )
    assert res_auto.brace_height_norm == pytest.approx(res_idx.brace_height_norm)


# ---------------------------------------------------------------------------
# Optional SVR smoother (only if scikit-learn is installed)
# ---------------------------------------------------------------------------

def test_svr_smoother_optional():
    pytest.importorskip("sklearn")
    from hdsemg_shared.motor_unit import smooth_discharge_rate_svr

    rng = np.random.default_rng(0)
    times = np.linspace(0.0, 5.0, 50)
    rate = 10.0 + 2.0 * times + rng.normal(0, 0.3, times.size)
    t_eval, smooth = smooth_discharge_rate_svr(times, rate, np.linspace(0, 5, 100))
    assert smooth.shape == (100,)
    # smoother output should be far less noisy than the raw signal
    assert np.std(np.diff(smooth)) < np.std(np.diff(rate))
