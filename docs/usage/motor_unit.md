# Motor Unit Analysis

The `motor_unit` subpackage quantifies the discharge behaviour of decomposed
motor units (MUs). It implements the **brace-height PIC** method of
Beauchamp et al. (2023) — a single-unit, geometric estimate of persistent
inward current (PIC) amplification — together with the discharge-rate
utilities it builds upon.

---

## Background

During a linear (triangular) ramp contraction, a MU that behaves as a passive
integrator of synaptic drive would discharge linearly with the produced
force/torque. Intrinsic activation from PICs makes the discharge rate rise
steeply just after recruitment and then attenuate, bowing the
discharge-vs-force trace away from a straight line. The maximum deviation from
that line — the **brace height** — is used as a proxy for PIC amplification
(and neuromodulatory drive).

The instant at which the maximum deviation occurs (the **brace point**) splits
the ascending phase into an **acceleration phase** (secondary range) and an
**attenuation phase** (tertiary range), yielding supplemental metrics:
acceleration slope, attenuation slope, and the angle between them.

---

## Geometry

For the ascending segment from recruitment to peak discharge rate, with
reference force/torque `x` and smoothed discharge rate `y`:

- The "theoretical linear discharge" is the straight hypotenuse from
  `(x_rec, y_rec)` to `(x_peak, y_peak)`.
- **Brace height** is the maximum orthogonal distance from that hypotenuse to
  the discharge trace.
- It is normalized to the altitude of the right triangle whose hypotenuse is
  the same line, giving units of **percent of the right-triangle height (% rTri)**.

Because discharge rate (pps) and force/torque are on different scales, the
geometry is evaluated in axes normalized to the recruitment→peak range. In that
frame the orthogonal distance and the right-triangle altitude carry the same
scale factor, which cancels, so the normalized brace height reduces to the
scale-invariant expression:

$$
\text{BH}_{\%rTri} = 100 \cdot \max\!\left(
\frac{y - y_\text{rec}}{y_\text{peak} - y_\text{rec}}
- \frac{x - x_\text{rec}}{x_\text{peak} - x_\text{rec}}
\right)
$$

The raw brace height is the equivalent vertical deviation in pps,
$(y_\text{peak} - y_\text{rec}) \cdot \max(\cdot)$.

**Exclusion criteria** (flagged via `valid` / `exclusion_reasons`, not
removed): negative acceleration slope, normalized brace height above
200 % rTri, or peak discharge occurring after peak force/torque.

---

## Pipeline

1. Convert a MU spike train (binary array or discharge sample indices) to
   firing times, then to the instantaneous discharge rate (reciprocal ISI).
2. Smooth the discharge rate into a continuous trace using Support Vector
   Regression (`smooth_discharge_rate_svr`; lazily imports scikit-learn).
3. Compute brace-height PIC metrics against the reference force/torque trace
   with `compute_brace_pic`.

---

## Example Usage

### From a pre-smoothed discharge-rate trace

```python
import numpy as np
from hdsemg_shared.motor_unit import compute_brace_pic

# smooth_rate in pps, torque in %MVT — same time base
result = compute_brace_pic(smooth_rate, torque)

print(result.brace_height_norm)    # % rTri  (primary PIC metric)
print(result.brace_height)         # pps vertical deviation
print(result.acceleration_slope)   # pps per %MVT
print(result.attenuation_slope)    # pps per %MVT
print(result.angle)                # degrees (180° = linear)
print(result.valid, result.exclusion_reasons)
```

### Directly from a spike train (recommended)

`brace_pic_from_spike_train` handles the full pipeline — firing times,
instantaneous rate, SVR smoothing, and brace metrics — in a single call:

```python
from hdsemg_shared.motor_unit import brace_pic_from_spike_train

# spikes: binary spike train or array of discharge sample indices
result = brace_pic_from_spike_train(spikes, torque, fsamp=2048)
```

Pass a custom smoother if you do not want the default SVR:

```python
result = brace_pic_from_spike_train(
    spikes, torque, fsamp=2048,
    smoother=my_smoother,  # callable: (times, rate, t_eval) -> (t_eval, smooth)
)
```

### All MUs in an openhdemg file

`compute_brace_pic_openhdemg_all` iterates over every MU, runs SVR
smoothing via `openhdemg.library.compute_svr`, and returns a summary
DataFrame alongside the structured per-MU results:

```python
import openhdemg.library as emg
from hdsemg_shared.motor_unit import compute_brace_pic_openhdemg_all

emgfile = emg.emg_from_samplefile()
summary_df, results = compute_brace_pic_openhdemg_all(emgfile)
print(summary_df[["mu", "brace_height_norm", "valid"]])
```

---

## Key `compute_brace_pic` Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `discharge_rate` | — | Smoothed discharge-rate trace in pps |
| `reference` | — | Force/torque trace on same time base |
| `recruitment_idx` | auto | First finite active sample (or explicit override) |
| `peak_idx` | auto | Peak discharge index (or explicit override) |
| `peak_reference_idx` | auto | Index of peak force/torque |
| `fsamp` | `None` | Sampling frequency in Hz |
| `time` | `None` | Explicit time axis in seconds |
| `distance_mode` | `"positive"` | `"positive"` (above-line deviation) or `"absolute"` |
| `phase_fit` | `"endpoints"` | `"endpoints"` or `"ols"` for phase slope estimation |
| `recruitment_window` | `1` | Samples averaged at recruitment for endpoint |
| `peak_window` | `1` | Samples averaged at peak for endpoint |
| `brace_window` | `1` | Samples averaged at brace point |
| `peak_torque_tolerance_s` | `0.0` | Grace window (s) for peak-discharge-after-peak-force check |
| `ci` | `False` | Request uncertainty intervals: `False`, `True`, or coverage level (e.g. `95`) |
| `ci_options` | `None` | `CIOptions` instance or dict with CI configuration |

---

## Uncertainty Estimation (CI)

`compute_brace_pic` accepts an optional `ci` argument that adds
model-based sensitivity/credible intervals to the result. The default
method (`jitter_svr`) infers latent discharge times from the smoothed
trace, adds discharge-time jitter, refits the SVR smoother for each draw,
and summarizes the draw distribution as an HDI or ETI interval.

```python
result = compute_brace_pic(smooth_rate, torque, ci=95)

ci = result.ci
print(ci.intervals["brace_height_norm"].lower,
      ci.intervals["brace_height_norm"].upper)  # 95 % HDI bounds
```

When CI is enabled, the scalar metric attributes on `BracePICResult`
(e.g. `brace_height_norm`) are the **means of the successful draws**.
Without CI they are the deterministic estimates from the input trace.

Key `CIOptions` parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `method` | `"jitter_svr"` | Draw generation strategy (`"jitter_svr"`, `"bootstrap_svr"`, `"trace_noise"`, `"sensitivity"`) |
| `interval` | `"hdi"` | `"hdi"` (shortest) or `"eti"` (equal-tailed) |
| `n_draws` | `500` | Number of draws |
| `n_jobs` | `1` | Parallel workers (`-1` = all CPUs) |
| `random_state` | `None` | Seed for reproducibility |
| `jitter_fraction_isi` | `0.10` | Jitter SD as fraction of local ISI |
| `svr_kwargs` | `{"C": 10, ...}` | SVR settings used for jitter-SVR and bootstrap-SVR draws |
| `bootstrap_rate_times` | `None` | Raw IDR times (s); required for `"bootstrap_svr"` |
| `bootstrap_rate_values` | `None` | Raw IDR values (pps); required for `"bootstrap_svr"` |
| `store_draws` | `True` | Keep per-draw scalar arrays in `result.ci.draws` |
| `store_trace_summary` | `True` | Keep mean/SD of smoothed-rate draws for plotting |

```python
from hdsemg_shared.motor_unit import CIOptions, compute_brace_pic

opts = CIOptions(n_draws=200, n_jobs=-1, random_state=42)
result = compute_brace_pic(smooth_rate, torque, ci=95, ci_options=opts)
```

### Bootstrap-SVR method

The `"jitter_svr"` method can systematically underestimate brace height
because jittering spike times causes the SVR to over-smooth, flattening the
curve. The `"bootstrap_svr"` method avoids this by resampling the residuals
of the original IDR→SVR fit instead:

1. Fit SVR to the raw IDR at the original spike times.
2. Compute and mean-center the residuals.
3. Per draw: resample residuals with replacement, add to the fitted values,
   refit SVR with the same hyperparameters, recompute brace metrics.

The draw mean is approximately unbiased w.r.t. the deterministic estimate.
This method requires the raw instantaneous discharge rate (times and values)
to be passed via `CIOptions`:

```python
from hdsemg_shared.motor_unit import CIOptions, compute_brace_pic

opts = CIOptions(
    method="bootstrap_svr",
    n_draws=500,
    n_jobs=-1,
    random_state=42,
    bootstrap_rate_times=rate_times,   # from instantaneous_discharge_rate()
    bootstrap_rate_values=idr,         # from instantaneous_discharge_rate()
)
result = compute_brace_pic(smooth_rate, torque, ci=95, ci_options=opts)
```

> **Note:** These intervals reflect sensitivity to spike timing and smoothing
> choices. They are not author-validated clinical confidence intervals.

---

## Plotting

`plot_brace` renders the brace geometry with the acceleration and
attenuation phases, the linear-discharge hypotenuse, and the brace
deviation. When CI data are present an uncertainty shadow is drawn.

```python
from hdsemg_shared.motor_unit import plot_brace

ax = plot_brace(result, title=f"MU0 — {result.brace_height_norm:.1f} % rTri")
```

Key parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `show_ci` | `True` | Draw CI shadow when available |
| `ci_shadow` | `"sd"` | `"sd"` or `"interval"` shadow style |
| `show_scale_bars` | `False` | Annotate scale bars on the axes |
| `title` | `None` | Axes title |

---

## Result Fields (`BracePICResult`)

| Field | Unit | Description |
| --- | --- | --- |
| `brace_height_norm` | % rTri | Normalized brace height (primary metric) |
| `brace_height` | pps | Equivalent vertical deviation |
| `brace_distance` | raw | Orthogonal distance in plot units |
| `right_triangle_height` | raw | Altitude of recruitment→peak right triangle |
| `acceleration_slope` | pps/ref | Slope of the acceleration phase |
| `attenuation_slope` | pps/ref | Slope of the attenuation phase |
| `angle` | degrees | Reflex angle at brace point (180° = linear) |
| `recruitment_idx`, `brace_idx`, `peak_idx` | samples | Key geometry indices |
| `peak_reference_idx` | samples | Index of peak force/torque |
| `recruitment_reference`, `recruitment_rate` | ref, pps | Values at recruitment point |
| `brace_reference`, `brace_rate` | ref, pps | Values at brace point |
| `peak_reference`, `peak_rate` | ref, pps | Values at peak |
| `valid` | bool | `False` if an exclusion criterion is triggered |
| `exclusion_reasons` | list[str] | Human-readable exclusion messages |
| `x`, `y` | — | Analysed recruitment→peak segment arrays |
| `time` | s | Time axis (if `fsamp` or `time` was supplied) |
| `ci` | `BracePICCI` | Uncertainty summary (when requested) |

`BracePICResult.as_dict(include_ci=True)` returns a flat dict suitable for
building a pandas DataFrame row.

---

## Metric Interpretation

> The full interpretation context — including comparisons with ΔF, simulation
> evidence, and physiological specificity — is in the analysis notebook
> `notebooks/brace_pic_openhdemg_sample.ipynb`, section
> *"Interpretation of PIC Metrics following Beauchamp et al. (2023)"*.

Brief summary from that notebook:

| Metric | Primarily indexes | Key limitation |
| --- | --- | --- |
| **BH (% rTri)** | PIC amplification / neuromodulatory drive | Does not capture recruitment–derecruitment hysteresis |
| **Acceleration slope** | Early high-gain "secondary range" turn-on | Most variable; sensitive to endpoint placement |
| **Attenuation slope** | Inhibitory command pattern | Not a pure PIC-amplitude metric |
| **Angle** | Overall curvature of ascending discharge | Physiological specificity weaker than BH |

180° angle = linear discharge; values above 180° indicate bowing consistent
with PIC amplification. The paper's conclusion: **brace height and attenuation
slope together can help separate neuromodulatory drive from
excitation–inhibition coupling** — a separation ΔF alone cannot provide.

---

## Sensitivity to Processing Choices

> See `notebooks/brace_pic_openhdemg_sample.ipynb` for the full sensitivity
> analysis with tables and plots. The key findings are summarised below.

The notebook sweeps endpoint placement, averaging windows, phase-fit method,
peak-force timing tolerance, and local SVR hyperparameters across 8 192
combinations and compares the resulting metric spread to jitter-SVR draw
uncertainty (HDI draw SD):

- **Endpoint placement** (recruitment/peak index shift) is the dominant
  sensitivity source for BH and angle (factor effect ≈ 0.8–0.9 × HDI draw SD).
- **Peak shift** and **phase fitting** dominate attenuation slope sensitivity.
- **Recruitment shift** dominates acceleration slope sensitivity.
- **Window averaging** (recruitment/peak/brace window) has smaller effects on
  BH and angle, negligible effects on slopes.
- **Peak-force timing tolerance** has moderate effects on BH, angle, and
  attenuation slope.
- **Local SVR sensitivity** (C, gamma, epsilon varied ±50 % around the
  paper setting) is not dominant; C has the largest effect at ≈ 0.6 × HDI
  draw SD on BH.

Practical implication: always report your endpoint-placement strategy
(`recruitment_idx`, `peak_idx`, `peak_reference_idx`, and the window
parameters) alongside the metrics. The `BRACE_KWARGS` dict in the notebook
serves as a reproducible parameter record.

---

## References

- Beauchamp et al. (2023), *A geometric approach to quantifying the
  neuromodulatory effects of persistent inward currents on individual motor
  unit discharge patterns*, **J. Neural Eng.** 20 016034.
  doi:[10.1088/1741-2552/acb1d7](https://doi.org/10.1088/1741-2552/acb1d7)
- Ugliara et al. (2025), *Isometric handgrip contraction increases tibialis
  anterior intrinsic motoneuron excitability* (bioRxiv).

---

## Source Code

> The implementation lives in `src/hdsemg_shared/motor_unit/`:
> `discharge_rate.py` (firing-time and rate utilities) and `brace_pic.py`
> (PIC geometry, CI machinery, and plotting).

---

### API Documentation

::: hdsemg_shared.motor_unit
    handler: python
    options:
      heading_level: 3
      show_root_heading: false
