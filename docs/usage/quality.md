# Quality

Measures that say whether a recording is fit to analyse: one number per
**channel**, and one answer per **grid**.

Nothing in `hdsemg_shared.quality` returns a verdict. Every function returns a
measured number, and where the threshold sits is the caller's decision — it
depends on the muscle, the electrode, the task and the population, not on the
measurement. That separation is what lets a quality gate store its evidence
next to the human answer it produced, and re-derive either one without the
other.

```python
from hdsemg_shared.quality import channel_amplitude, propagation
```

---

## Channel measures

All of them take a channel matrix, **channels-by-samples** — `EMGFile.data` is
samples-by-channels, so pass `emg.data.T` — and return one entry per row. A
channel that is entirely NaN, an excluded or unwired position, yields NaN
rather than raising.

| function | question it answers |
|---|---|
| `flat_channels` | is the electrode disconnected or saturated? |
| `channel_amplitude` | how strong is it, in-band? |
| `channel_spectrum` | where does its power sit — MNF and MDF? |
| `line_noise_ratio` | does it pick up the mains? |
| `clipping_fraction` | does it sit at the amplifier's rail? |
| `neighbor_correlation` | does it share anything with the electrodes next to it? |
| `robust_z` | how far from its own grid is any of the above? |

### Finding the channels that carry no signal

```python
from hdsemg_shared.quality import flat_channels

dead = flat_channels(emg.data.T)
```

A flat channel is not a small signal, it is no signal. Left in, it contributes
zero energy to the mean square over the grid area and drags
`global_amplitude` **down** in proportion to how many there are. Measured
across five subjects, flat channels are always exactly zero and nothing else
comes within 20 % of the grid median, so the default threshold sits in a wide
empty gap.

### Scoring a channel against its own grid

Absolute amplitudes vary between subjects, sessions and muscles, so the useful
question is not "is this channel loud" but "is this channel unlike the rest of
*this* grid". `robust_z` answers that with a median/MAD score:

```python
from hdsemg_shared.quality import channel_amplitude, robust_z

amp = channel_amplitude(emg.data.T, fs)
outliers = np.flatnonzero(np.abs(robust_z(amp.rms)) > 3.5)
```

Median and MAD rather than mean and standard deviation, because the outliers
being looked for are *in* the sample: with two bad channels, a classical
z-score is inflated by exactly those two and hides the second one.

### Line noise

```python
from hdsemg_shared.quality import line_noise_ratio

noise = line_noise_ratio(emg.data.T, fs, freqs=(50.0, 100.0, 150.0))
noise.ratio            # per channel, the worst of the three
noise.per_frequency    # 3 x nChannels, one row per harmonic
```

The mean power within ±2 Hz of the line frequency, over the median power of
the ring 2–20 Hz around it. A clean channel reads about **1.4**; mains pickup
reads one to two orders of magnitude above.

Two details differ from a naive peak test, and both matter:

- The background is a **local ring**, not the whole spectrum. sEMG power falls
  by orders of magnitude above the signal band, so a whole-spectrum background
  is dominated by near-empty high-frequency bins. Measured on a synthetic,
  mains-free, band-limited signal, that mistake reported a ratio of **898×**.
- The peak window is reduced by its **mean**, not its maximum. The maximum of
  *K* noisy bins grows like log *K*, so a max-based ratio drifts with record
  length and has no fixed clean value.

### Neighbour correlation — pass a high-activity window

```python
from hdsemg_shared.quality import neighbor_correlation

r = neighbor_correlation(emg.data.T, emg_map, fs, window=peak_window)
```

A real channel shares its motor unit action potentials with the electrodes
next to it; an isolated noisy one does not.

!!! warning "Over a whole trial this measures nothing"
    A force-tracking trial is mostly rest, and at rest neighbouring monopolar
    channels share little but their own uncorrelated noise — so `r` collapses
    toward zero for **good** channels too. Restrict it to a window where the
    muscle is actually active. Calibrate the threshold on data known to be
    good before rejecting anything with it; there is no universal value.

---

## Grid propagation: direction, velocity, innervation zone

```python
from hdsemg_shared.quality import propagation

prop = propagation(emg.data.T, emg_map, ied_mm=10.0, fs=fs, window=peak_window)

prop.fiber_angle_deg          # 0 deg = along the grid COLUMN axis
prop.propagation_score        # 0..1, does this grid see propagation at all
prop.cv_reported_ms           # THE velocity — read this one
prop.cv_status                # and read this before it
prop.iz_detected, prop.iz_position_m
```

!!! warning "Read `cv_reported_ms`, not `conduction_velocity_ms`"
    A conduction velocity is only defined where the potentials travel in a
    **single direction**. Across an innervation zone they travel both ways, so
    the whole-grid figure there is an average of two opposite propagations and
    is not a velocity of anything.

    `cv_reported_ms` switches to the one-sided estimate (`cv_side_ms`)
    whenever `cv_status == "iz_split"`, and equals `conduction_velocity_ms`
    otherwise. Reading it instead of choosing between the two by hand is what
    keeps an end-plate-averaged number out of a result.

Electrodes are projected onto a candidate direction, binned at half the
inter-electrode distance, averaged within a bin, and consecutive bins are
cross-correlated. Along the true fibre direction those adjacent-bin delays all
have the **same magnitude** — one bin spacing divided by the conduction
velocity.

### Read `cv_status` before the velocity

| status | meaning |
|---|---|
| `"ok"` | one propagation direction across the whole grid |
| `"iz_split"` | an innervation zone splits the grid; `cv_reported_ms` is the **one-sided** estimate, which is the only one defined here, or `NaN` when neither side carries four bin pairs |
| `"too_few_pairs"` | fewer than four bin pairs existed at all; the velocity is NaN |
| `"out_of_range"` | bin pairs existed but none implied a physiological velocity; the velocity is NaN |

A status that says the velocity is not estimable is handed `NaN`, never a
number — so a caller who reads the number first cannot mistake it for a
measurement.

!!! warning "`iz_split` does not guarantee a velocity"
    When an innervation zone splits the grid and **neither side** carries
    four bin pairs, there is no one-sided estimate, and `cv_reported_ms` is
    `NaN`. It is deliberately *not* backfilled with the whole-grid median:
    that number averages pairs on both sides of the end plate, where the
    potentials travel in opposite directions, so it is not the velocity of
    anything.

    This matters because such a number looks entirely respectable. Over 1296
    measured grids from one study, 101 were in this state and 53 of them
    produced a whole-grid figure inside the physiological 3.0–5.5 m/s band —
    nothing about the value itself would have told a reader it was
    meaningless. Read `cv_reported_ms`, and treat `NaN` as "not measured"
    rather than as "grid is bad".

!!! note "An innervation zone is a note, not a failure"
    A grid straddling an innervation zone is perfectly usable for amplitude,
    and its velocity survives here too — measured on **one side**, via
    `cv_reported_ms`. `cv_status == "iz_split"` is information to report,
    **not** a reason to discard the grid.

### Why the search does not use a straight-line fit

The obvious criterion — regress delay against distance from a common anchor
and take the direction with the best R² — fails exactly where it matters. Under
an innervation zone the potentials travel *both* ways from the end-plate
region, so delay against distance is V-shaped and its linear R² collapses.

On a synthetic grid with a planted innervation zone, the true direction scored
R² = 0.474 while a direction *across* the fibres, where every delay is
identically zero, scored 0.480 and won. The reported velocity was 2.64 m/s of
nothing at all.

The adjacent-bin delays at that same true direction were

```
[2.93, 2.44, 2.44, 2.44, 2.44, -2.44, -2.44, -2.44, -2.44, -2.93, -2.44] ms
```

— one constant magnitude, one sign reversal: the correct velocity *and* the
innervation zone, both plainly there. So the search scores

```
propagation_score = consistency x coherence x coverage
```

- **consistency** `(mean|d|)² / mean(d²)` — 1 when every delay has the same
  magnitude, and **blind to sign**, so a reversal does not lower it;
- **coherence** — the mean normalised cross-correlation at those delays.
  Consistency alone is not enough: an oblique direction bins one or two
  electrodes per bin, and sample quantisation then makes several delays land
  on the same value by accident. A direction 87° off scored consistency 1.000
  and coherence 0.652, against 0.998 at the true direction;
- **coverage** — the share of bin pairs that implied a physiological velocity,
  so a direction with two lucky pairs cannot beat one with twelve.

`r_squared` is still reported, because a **low `r_squared` beside a high
`propagation_score`** is itself the signature of an innervation zone.

### Checking a grid's assumed orientation

`propagation` measures which way the fibres actually run, which is what an
SD/DD derivation axis should be checked against — far more informative than
whether a reviewer ticked a box:

```python
prop = propagation(emg.data.T, emg_map, ied_mm, fs, window=peak_window)

# 0 deg is the axis global_amplitude differences along with diff_direction='cols'
axis_disagreement_deg = min(abs(prop.fiber_angle_deg),
                            180 - abs(prop.fiber_angle_deg))
trustworthy = prop.propagation_score > 0.5
```

!!! warning "Pool the trials of a session before believing a disagreement"
    An electrode grid is usually applied once and not moved again, so whether
    it is rotated is a property of the **session**, not of a trial. A single
    trial disagreeing is measurement noise by construction.

    Measured over 96 grids of one participant, the disagreement was bimodal
    with an empty valley from 5° to 30° — but every grid had a *median* of
    exactly 0° with a third of its trials in the far tail, and those trials
    carried lower `propagation_score`. Pool a session's trials and take the
    majority; a grid whose trials *mostly* measure 90° really is mounted
    across the axis, and the remedy is to rotate the map, never to discard
    the grid.

---

## Amplitude maps: the grid laid out in space

`propagation` reads a grid through **delays** — when each position fires.
`amplitude_map` reads the same grid through **amplitudes** — how strong each
position is over one short epoch, laid out the way the electrodes sit on the
muscle. It is what a heat map is drawn from, and it finds the innervation zone
a second, independent way.

```python
from hdsemg_shared.quality import (amplitude_map, innervation_zone_line,
                                   upsample_map, barycenter)

amap = amplitude_map(emg.T, emg_map, ied_mm=10.0, fs=fs, window=peak_window)
iz = innervation_zone_line(upsample_map(amap))

iz.center_xy_mm      # (15.0, 25.98) mm
iz.full_width        # False means a PARTIAL detection - see below
barycenter(amap)     # where the signal sits under the grid
```

Differences are taken **along each grid column**, between electrodes adjacent
in the map's inner index — the same axis `propagation` calls 0°.

### Two innervation zone estimates, on purpose

Under an end plate the potentials leave in both directions, so the
differentials either side partly cancel and the amplitude has a local
**minimum**. That is what `innervation_zone_line` finds. `propagation` instead
finds where the adjacent-bin delays **reverse sign**. The two fail differently:

| | needs | fooled by |
|---|---|---|
| amplitude dip | no delay estimate, so it survives short or noisy epochs | any other cause of a dip — a lifted electrode, a fat pocket, a bad connector row |
| delay reversal | a usable delay at every bin | little else, but it simply fails when the epoch is too short |

Where they agree, the innervation zone is real. Where they do not, **the
disagreement is the finding** and neither should be quoted alone.

### `full_width` is the confidence

The dip is found per column, then kept only where it moves by no more than
half an inter-electrode distance from one column to the next — an end plate is
a band across the muscle, not a scatter of unrelated dips. A run narrower than
the grid is a **partial** detection, and `center_xy_mm` is then the centre of
that part, not of the grid. Check `full_width` before quoting the position.

!!! note "Upsampling carries the electrode spacing with it"
    After `upsample_map`, the spacing of `x_mm` is 0.1 mm, not the electrode
    spacing. The continuity tolerance needs the real one, so `AmplitudeMap`
    carries `ied_mm` through the interpolation rather than letting anything
    read it off the axis.

---

## Relationship to `hdsemg-select`

`hdsemg_shared.quality.propagation` generalises `hdsemg-select`'s
`FiberTrajectoryAnalyzer`: it takes an `emg_map` instead of that application's
`grid` object and `display_grid` array, so it needs no electrode code table
and nothing from the GUI. `line_noise_ratio` generalises that application's
`auto_flagger` frequency test into a ratio rather than a label.

`auto_flagger._detect_artifact` is deliberately **not** ported — it flags
variance *above* `1e-9`, which fires on every live channel. Use
`flat_channels` for dead channels and `clipping_fraction` for artefacts.
