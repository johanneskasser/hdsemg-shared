# Motor Unit Analysis

The `motor_unit` package quantifies the discharge behaviour of decomposed motor
units (MUs). It currently implements the **brace-height** method of
Beauchamp et al. (2023), a single-unit, geometric estimate of persistent inward
current (PIC) amplification, together with the discharge-rate utilities it
builds upon.

---

## Background

During a linear (triangular) ramp contraction, a MU that behaves as a passive
integrator of synaptic drive would discharge linearly with the produced
force/torque. Intrinsic activation from PICs makes the discharge rate rise
steeply just after recruitment and then attenuate, bowing the discharge-vs-force
trace away from a straight line. The maximum deviation from that line — the
**brace height** — is therefore used as a proxy for PIC amplification (and, by
extension, neuromodulatory drive).

The instant at which the maximum deviation occurs (the **brace point**) splits
the ascending phase into an **acceleration phase** (secondary range) and an
**attenuation phase** (tertiary range), yielding supplemental metrics: the
acceleration slope, attenuation slope, and the angle between them.

---

## Geometry

For the ascending segment from recruitment to peak discharge rate, with
reference force/torque `x` and smoothed discharge rate `y`:

- The "theoretical linear discharge" is the straight hypotenuse from
  `(x_rec, y_rec)` to `(x_peak, y_peak)`.
- **Brace height** is the maximum orthogonal distance from that hypotenuse to
  the discharge trace.
- It is normalized to the altitude of the right triangle whose hypotenuse is the
  same line, giving units of **percent of the right-triangle height (% rTri)**.

Because discharge rate (pps) and force/torque are on different scales, the
geometry is evaluated in axes normalized to the recruitment→peak range. In that
frame the orthogonal distance and the right-triangle altitude carry the same
scale factor, which cancels, so the normalized brace height reduces to the
scale-invariant expression

$$
\text{BH}_{\%rTri} = 100 \cdot \max\!\left(
\frac{y - y_\text{rec}}{y_\text{peak} - y_\text{rec}}
- \frac{x - x_\text{rec}}{x_\text{peak} - x_\text{rec}}
\right)
$$

The raw brace height is the equivalent vertical deviation in pps,
$(y_\text{peak} - y_\text{rec}) \cdot \max(\cdot)$.

**Exclusion criteria** (flagged via `valid` / `exclusion_reasons`, not removed):
a negative acceleration slope, a normalized brace height above 200 % rTri, or
peak discharge occurring after peak force/torque.

---

## Pipeline

1. Convert a MUAP spike train (binary array or discharge indices) to discharge
   times, then to the instantaneous discharge rate (reciprocal of the ISI).
2. Smooth the discharge rate into a continuous trace (the paper uses Support
   Vector Regression; `smooth_discharge_rate_svr` provides this and lazily
   imports scikit-learn, which is an optional dependency).
3. Compute brace height and its supplemental metrics against the reference
   force/torque trace.

---

## Example Usage

### From a pre-smoothed discharge-rate trace

```python
import numpy as np
from hdsemg_shared.motor_unit import compute_brace_height

# smoothed discharge rate (pps) and reference torque (% MVT), same time base
result = compute_brace_height(smooth_rate, torque)

print(result.brace_height_norm)    # brace height in % rTri
print(result.acceleration_slope)   # pps per % MVT
print(result.attenuation_slope)    # pps per % MVT
print(result.angle)                # degrees
print(result.valid, result.exclusion_reasons)
```

### Directly from a spike train

```python
from hdsemg_shared.motor_unit import brace_height_from_spike_train

# `spikes` may be a binary spike train or an array of discharge sample indices
result = brace_height_from_spike_train(spikes, torque, fsamp=2048)
```

To smooth without scikit-learn, pass your own
`smoother(times, rate, t_eval) -> (t_eval, smooth_rate)` callable.

---

## References

- Beauchamp et al. (2023), *A geometric approach to quantifying the
  neuromodulatory effects of persistent inward currents on individual motor unit
  discharge patterns*, **J. Neural Eng.** 20 016034.
- Ugliara et al. (2025), *Isometric handgrip contraction increases tibialis
  anterior intrinsic motoneuron excitability* (bioRxiv).

## Source Code

> The implementation can be found in `src/hdsemg_shared/motor_unit/`
> (`discharge_rate.py`, `brace_height.py`), with type hints and detailed
> docstrings.

---

### API Documentation

::: hdsemg_shared.motor_unit
    handler: python
    options:
      heading_level: 3
      show_root_heading: false
