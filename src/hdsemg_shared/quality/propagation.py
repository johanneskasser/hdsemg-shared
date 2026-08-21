"""
PROPAGATION: fibre direction, conduction velocity and innervation zone of a
grid

  Answers three questions about ONE HDsEMG grid at once, because they are
  the same measurement read three ways:

    1  along which direction do the action potentials travel   [deg]
    2  how fast                                                [m/s]
    3  is there an innervation zone under the grid, and where  [m]

  Electrodes are projected onto a candidate direction, binned at half the
  inter-electrode distance, averaged within a bin, and consecutive bins are
  cross-correlated. Along the true fibre direction those adjacent-bin delays
  all have the SAME MAGNITUDE, one bin spacing divided by the conduction
  velocity. Along any other direction they scatter, and across the fibres
  they collapse to zero. That is the whole measurement.

  WHY THE SEARCH SCORES DELAY CONSISTENCY, NOT A STRAIGHT-LINE FIT
    The obvious criterion - regress delay against distance from a common
    anchor and take the direction with the best R^2 - fails exactly where it
    matters most. Under an innervation zone the potentials travel BOTH ways
    from the end-plate region, so delay against distance is V-shaped and its
    linear R^2 collapses. Measured on a synthetic grid with a planted
    innervation zone, the true direction scored R^2 = 0.474 while a
    direction ACROSS the fibres, where every delay is identically zero,
    scored 0.480 and won. The reported velocity was then 2.64 m/s of
    nothing at all.

    The adjacent-bin delays at that same true direction were
    [2.93, 2.44, 2.44, 2.44, 2.44, -2.44, -2.44, -2.44, -2.44, -2.93, -2.44]
    ms: one constant magnitude, one sign reversal, i.e. the correct velocity
    AND the innervation zone, both plainly there. So the search scores how
    consistent those magnitudes are, which a sign reversal does not disturb,
    and the velocity is their median rather than a regression slope. The
    across-fibre direction is rejected by the same criterion, because a zero
    delay implies an infinite velocity and is not physiological.

  WHY THE DELAYS ARE MEASURED ON THE SINGLE DIFFERENTIAL
    A monopolar HDsEMG channel is dominated by a common component every
    electrode of the grid sees at once - far-field activity, mains, the
    reference. Cross-correlating two monopolar signals therefore peaks at
    very nearly zero lag whatever the fibres are doing, and the small
    residual lag that is left is noise.

    Measured on ID343's 10 mm 4x8 grid, along the axis the SD/DD derivation
    actually differences along, the monopolar adjacent-bin delays were
      [1.0, 0.5, 0.0, -0.5, -0.5, -0.5, -0.5] ms
    i.e. at or below one sample at 2 kHz, giving velocities pinned to the
    10 m/s ceiling and a propagation score of 0. The SINGLE DIFFERENTIAL
    between the same bins gave
      [1.5, 1.0, -1.5, -1.0, -2.0, -2.0] ms
    - four to five times larger, physiological (2.5-6.7 m/s over 10 mm), and
    carrying a clean sign reversal, i.e. an innervation zone, between the
    second and third pair. Same electrodes, same window, same filter.

    Differencing adjacent bins subtracts the common component and leaves the
    travelling one, which is why Farina and Merletti estimate conduction
    velocity from differential channels rather than monopolar ones.
    hdsemg-select correlates the monopolar signal; that is the single
    biggest reason its velocities on this data are not usable, and it is why
    derivation defaults to 'SD' here.

  *THE FREE ANGLE SEARCH HAS A LATTICE ARTEFACT - READ THIS BEFORE TRUSTING
   fiber_angle_deg*
    Electrodes sit on a square lattice, and the half-IED binning packs them
    into few well-populated bins only at directions COMMENSURATE with that
    lattice - 0, +/-45 and +/-90 degrees. At any other direction the bins hold
    one or two electrodes each and the delays between them are noisy. The
    score therefore prefers the commensurate directions whatever the fibres
    are doing.

    Measured over 1296 real grids: 444 of them returned EXACTLY 0 deg and 296
    returned EXACTLY +/-45 deg, with nothing at all between -10 and 0. The
    propagation score at +/-45 (0.602) was indistinguishable from the score at
    0 or 90 (0.626), so the score does not separate them either. A real
    anatomical fibre angle would scatter around a value; this piles up on the
    lattice.

    So: fiber_angle_deg is trustworthy when the propagation is strong, and
    falls back to a lattice direction when it is not. IF THE QUESTION IS
    BINARY - do the potentials run along this axis or across it - DO NOT USE
    THE FREE SEARCH. Pass the two axes explicitly and compare their scores:

        two = propagation(emg.T, emg_map, ied_mm, fs, angles = [0.0, 90.0])
        along, across = two.search_score

    On the same data that comparison separated cleanly where the free search
    did not: a participant whose grids all read "along" scored 0.59-0.63
    against 0.00-0.36, and one whose grids all read "across" scored 0.45-0.91
    against 0.00-0.35 - including the grids the free search had placed at
    +/-45 deg.

  WHY AN INNERVATION ZONE IS A NOTE, NOT A FAILURE
    A grid straddling an innervation zone is perfectly usable for amplitude,
    and its velocity survives too - but only ON ONE SIDE. A conduction
    velocity is defined where the potentials travel in a single direction,
    and across an end-plate region they travel both ways, so a whole-grid
    number there is an average of two opposite propagations and is not a
    velocity of anything. cv_reported_ms therefore switches to the one-sided
    estimate whenever cv_status is "iz_split", and that is the field to read.
    cv_status is NOT a reason to discard a grid.

  REFERENCE
    Farina D, Merletti R. Estimation of average muscle fiber conduction
      velocity from two-dimensional surface EMG recordings.
      J Neurosci Methods 134:199-208, 2004.

  PORT of hdsemg-select's select_logic/fiber_trajectory.py
  (FiberTrajectoryAnalyzer), generalised for hdsemg-shared by Claude Opus 5,
  2026-08-19. The projection, the half-IED binning, the cross-correlation
  delay and the sign-reversal innervation zone test are ported unchanged.

  DIFFERENCES TO THE hdsemg-select ORIGINAL
    x it takes an emg_map (preprocessing.grid_map's convention) instead of
      select's own grid object plus display_grid, so it needs nothing from
      that application and no electrode code table
    x the direction is chosen by adjacent-bin delay consistency, not by the
      anchor-first regression R^2, for the reason set out above. That
      regression is still computed and reported as r_squared, because a low
      value next to a high propagation_score is itself the signature of an
      innervation zone
    x it band-passes before cross-correlating by default; select correlates
      the raw monopolar signal. Pass bpf = False for the original behaviour
    x it correlates the SINGLE DIFFERENTIAL between adjacent bins rather
      than the monopolar bin signals, which is what makes the delays
      measurable at all on real data. Pass derivation = 'MP' for select's
      behaviour
    x the CV is reported UNCLIPPED, with cv_status, instead of being clipped
      into [2, 10] m/s with no record that it was
    x a direction that carries nothing reports NaN and a cv_status rather
      than select's silent (0.0, 0.0)

  ADDITIONS OF THIS IMPLEMENTATION (no equivalent in select)
    propagation_score, iz_detected, cv_side_ms, cv_reported_ms, cv_status,
    n_valid_pairs, n_electrodes and the derivation option.
"""

import warnings
from typing import NamedTuple

import numpy as np
from scipy.signal import correlate
from scipy.stats import linregress

from hdsemg_shared.preprocessing.grid_map import _as_map, _check_indices
from hdsemg_shared.quality.channel_metrics import (
    DEFAULT_PAD_S,
    _apply_window,
    _bandpass,
    _merged_bpf,
)

#: Physiological bounds on muscle fibre conduction velocity. A bin pair whose
#: implied velocity falls outside them carries no propagation and is dropped.
#: Typical human range is 3-5 m/s; these are deliberately wider.
MIN_CV_MS = 2.0
MAX_CV_MS = 10.0

#: Fewer surviving bin pairs than this and neither the direction nor the
#: velocity means anything.
MIN_VALID_PAIRS = 4

#: Angles searched by default, one degree apart over the half circle. A
#: direction and its opposite give the same delays with the sign flipped, and
#: the measure is blind to that sign, so the half circle is the whole search.
DEFAULT_ANGLES = np.arange(-90.0, 91.0)


class PropagationResult(NamedTuple):
    """
    fiber_angle_deg       ... direction whose adjacent-bin delays are most
                               consistent. 0 deg points along the grid COLUMN
                               axis, the axis global_amplitude differences
                               along with diff_direction = 'cols' [deg]
    propagation_score     ... 0 to 1, how consistent those delay magnitudes
                               are, scaled by the share of bin pairs that
                               carried a physiological velocity at all. THIS
                               is the number that says whether the grid sees
                               propagation; 0 means it does not []
    conduction_velocity_ms... median of the per-bin-pair velocities over the
                               WHOLE grid at that direction, NaN when not
                               estimable. *NOT the number to quote when an
                               innervation zone is present* - see
                               cv_reported_ms [m/s]
    cv_side_ms            ... the same median over the longer side of a
                               detected innervation zone ALONE. This is the
                               physiologically correct estimate when there
                               is one: a velocity is only defined where the
                               potentials travel in a single direction, and
                               across an end-plate region they do not. None
                               when there is no innervation zone, or when
                               that side cannot carry MIN_VALID_PAIRS [m/s]
    cv_reported_ms        ... THE velocity to use: cv_side_ms when an
                               innervation zone splits the grid, otherwise
                               conduction_velocity_ms. Reading this instead
                               of picking between the two by hand is what
                               keeps a whole-grid number, averaged across an
                               end-plate region, out of a result [m/s]
    cv_status             ... char, one of:
                               x "ok"            - one propagation direction
                                  across the whole grid
                               x "iz_split"      - an innervation zone
                                  splits the grid; the velocity still holds,
                                  see cv_side_ms
                               x "too_few_pairs" - fewer than
                                  MIN_VALID_PAIRS bin pairs existed at all
                               x "out_of_range"  - bin pairs existed but not
                                  one of them implied a physiological
                                  velocity
    r_squared             ... R^2 of the anchor-first delay-against-distance
                               regression at fiber_angle_deg, the criterion
                               hdsemg-select selects on. LOW here together
                               with a HIGH propagation_score is the
                               signature of an innervation zone []
    iz_detected           ... whether a propagation reversal was found []
    iz_position_m         ... projected position of that reversal, or None [m]
    search_angles         ... every direction that was tried [deg]
    search_score          ... propagation_score at each of them []
    search_cv_ms          ... velocity at each of them, NaN where none [m/s]
    pairwise_delays_ms    ... signed delays between consecutive projection
                               bins at fiber_angle_deg [ms]
    pairwise_distances_m  ... midpoint position of each of those pairs [m]
    n_valid_pairs         ... bin pairs that carried a physiological
                               velocity at fiber_angle_deg []
    n_electrodes          ... live electrodes the map placed []
    """

    fiber_angle_deg: float
    propagation_score: float
    conduction_velocity_ms: float
    cv_side_ms: float
    cv_reported_ms: float
    cv_status: str
    r_squared: float
    iz_detected: bool
    iz_position_m: float
    search_angles: np.ndarray
    search_score: np.ndarray
    search_cv_ms: np.ndarray
    pairwise_delays_ms: np.ndarray
    pairwise_distances_m: np.ndarray
    n_valid_pairs: int
    n_electrodes: int


def propagation(emg_channels, emg_map, ied_mm, fs, angles = None, bpf = None,
                window = None, pad_s = DEFAULT_PAD_S, derivation = 'SD'):
    """
    PROPAGATION estimates fibre direction, conduction velocity and
    innervation zone position of one HDsEMG grid

      prop = propagation(emg.T, emg_map, ied_mm = 10.0, fs = 2048)
      prop = propagation(emg.T, emg_map, 10.0, fs, window = peak_window)

      INPUT
        emg_channels ... matrix, each row contains one channel's RAW
                          monopolar signal over time (rows = channels,
                          columns = samples). Filtering happens here, so
                          pass it unfiltered. May hold more channels than
                          the map refers to [xV]
        emg_map      ... list of lists or 2-D array, outer index = grid
                          COLUMN, inner index = ROW, base-0 channel numbers,
                          NaN where no electrode sits or a channel is
                          excluded []
        ied_mm       ... inter-electrode distance of the grid [mm]
        fs           ... double or int, sampling frequency [Hz]

      OPTIONAL INPUT
        angles       ... vector of directions to search, default
                          DEFAULT_ANGLES (-90 to 90 in 1 deg steps) [deg]
        bpf          ... band-pass options as in
                          channel_metrics.channel_amplitude; None (default)
                          means channel_metrics.DEFAULT_BPF, False means do
                          not filter at all (hdsemg-select's behaviour) []
        window       ... slice, boolean mask or (start, stop) tuple over
                          samples. A window where the muscle is ACTIVE gives
                          far better delay estimates than a whole trial that
                          is mostly rest []
        pad_s        ... reflection padding before filtering, default
                          0.25 [s]
        derivation   ... char, 'SD' (default) cross-correlates the SINGLE
                          DIFFERENTIAL between adjacent bins, 'MP' the
                          monopolar bin signals themselves as hdsemg-select
                          does. *KEEP THE DEFAULT* unless reproducing that
                          behaviour - see WHY THE DELAYS ARE MEASURED ON THE
                          SINGLE DIFFERENTIAL in the module docstring []

      OUTPUT
        prop         ... PropagationResult, see its own docstring. Read
                          propagation_score and cv_status BEFORE
                          conduction_velocity_ms

      *INFO* ... an excluded channel belongs in the map as NaN, exactly like
                  an unwired position; both are simply not placed
      *INFO* ... a channel whose signal is all zero or all NaN is dropped
                  even if the map places it, so a grid with dead electrodes
                  needs no separate masking step
      *INFO* ... the delay resolution is one sample, so at 2 kHz and 10 mm
                  spacing a single bin pair resolves the velocity only to
                  about 20 %. The median over many pairs is what makes the
                  estimate usable; a grid with few rows will be coarse
    """
    x = np.asarray(emg_channels, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"emg_channels must be 2-D (channels x samples), got shape {x.shape}.")
    if ied_mm <= 0:
        raise ValueError(f"ied_mm must be positive, got {ied_mm}.")

    grid = _as_map(emg_map)
    _check_indices(grid, x.shape[0])

    prepared = x if bpf is False else _bandpass(x, _merged_bpf(bpf), fs, pad_s)
    prepared = _apply_window(prepared, window)

    ied_m = float(ied_mm) * 1e-3
    mono = _placed_signals(prepared, grid)
    n_electrodes = len(mono)

    search_angles = np.asarray(DEFAULT_ANGLES if angles is None else angles,
                               dtype=np.float64).ravel()
    if search_angles.size == 0:
        raise ValueError("angles must name at least one direction.")

    if n_electrodes < 2:
        return _not_estimable(search_angles, n_electrodes)

    search_score = np.zeros(search_angles.size)
    search_cv = np.full(search_angles.size, np.nan)
    for i, theta in enumerate(search_angles):
        delays, _, spacings, peaks = _binned_adjacent_delays(
            theta, mono, ied_m, fs, derivation)
        score, cv, _ = _score_direction(delays, spacings, peaks)
        search_score[i] = score
        search_cv[i] = cv

    best = _best_angle_index(search_score)
    best_angle = float(search_angles[best])

    delays_ms, positions_m, spacings_m, peaks = _binned_adjacent_delays(
        best_angle, mono, ied_m, fs, derivation)
    score, cv, n_valid_pairs = _score_direction(delays_ms, spacings_m, peaks)

    iz_position = _detect_iz(delays_ms, positions_m)
    iz_detected = iz_position is not None

    cv_side = None
    if delays_ms.size < MIN_VALID_PAIRS:
        cv_status = "too_few_pairs"
    elif n_valid_pairs == 0:
        # Delays were there, but none of them implied a velocity that could
        # be a muscle fibre; that is a different fault from having no data
        cv_status = "out_of_range"
    elif n_valid_pairs < MIN_VALID_PAIRS:
        cv_status = "too_few_pairs"
    elif iz_detected:
        cv_status = "iz_split"
        cv_side = _cv_on_one_side(delays_ms, positions_m, spacings_m, iz_position)
    else:
        cv_status = "ok"

    # A status that says the velocity is not estimable must not be handed a
    # velocity anyway: one bin pair on a two-electrode grid does produce a
    # number, and a caller reading the number before the status would take
    # it for a measurement. The status is the guard, so it clears the field
    if cv_status in ("too_few_pairs", "out_of_range"):
        cv = np.nan

    # Reported for continuity with hdsemg-select, and because a low value
    # beside a high propagation_score is itself the innervation zone signature
    r_squared = _anchor_regression_r2(best_angle, mono, ied_m, fs)

    # A velocity is only defined where the potentials travel one way, so an
    # innervation zone makes the ONE-SIDED estimate the reported one
    cv_reported = cv_side if (cv_status == "iz_split" and cv_side is not None) else cv

    return PropagationResult(
        fiber_angle_deg=best_angle,
        propagation_score=float(score),
        conduction_velocity_ms=float(cv),
        cv_side_ms=cv_side,
        cv_reported_ms=cv_reported,
        cv_status=cv_status,
        r_squared=r_squared,
        iz_detected=iz_detected,
        iz_position_m=iz_position,
        search_angles=search_angles,
        search_score=search_score,
        search_cv_ms=search_cv,
        pairwise_delays_ms=delays_ms,
        pairwise_distances_m=positions_m,
        n_valid_pairs=int(n_valid_pairs),
        n_electrodes=n_electrodes,
    )


# ---------------------------------------------------------------------------
# signal preparation
# ---------------------------------------------------------------------------

def differential_map(emg_channels, emg_map, ied_mm, fs, angle_deg, bpf = None,
                     window = None, pad_s = DEFAULT_PAD_S, derivation = 'SD'):
    """
    DIFFERENTIAL_MAP returns the signals a propagation figure is drawn from

      positions, signals = differential_map(emg.T, emg_map, 10.0, fs,
                                            prop.fiber_angle_deg)

      The same binning propagation() measures on, handed back so a caller can
      DRAW it: electrodes projected onto one direction, binned at half the
      inter-electrode distance, averaged within a bin, and differenced between
      adjacent bins.

      INPUT
        emg_channels ... matrix, each row one channel's RAW signal over time [xV]
        emg_map      ... the grid's channel map, see propagation []
        ied_mm       ... inter-electrode distance [mm]
        fs           ... sampling frequency [Hz]
        angle_deg    ... the direction to project along, normally
                          propagation()'s fiber_angle_deg [deg]

      OPTIONAL INPUT
        bpf / window / pad_s ... as propagation []
        derivation   ... 'SD' (default) or 'MP' []

      OUTPUT
        positions    ... where each row sits along the direction [m]
        signals      ... nPositions-by-nSamples, row i at positions[i] [xV]

      *INFO* ... an image of `signals` shows the potentials travelling as
                  slanted stripes, and an innervation zone as the point where
                  their slant REVERSES. propagation()'s iz_position_m lands on
                  the same axis as `positions`, so it can be drawn straight
                  onto that image as a horizontal line
    """
    x = np.asarray(emg_channels, dtype=np.float64)
    grid = _as_map(emg_map)
    _check_indices(grid, x.shape[0])

    prepared = x if bpf is False else _bandpass(x, _merged_bpf(bpf), fs, pad_s)
    prepared = _apply_window(prepared, window)

    mono = _placed_signals(prepared, grid)
    if not mono:
        return np.zeros(0), np.zeros((0, prepared.shape[1]))

    binned = _binned_signals(float(angle_deg), mono, float(ied_mm) * 1e-3, derivation)
    if not binned:
        return np.zeros(0), np.zeros((0, prepared.shape[1]))

    positions = np.asarray([p for p, _ in binned], dtype=np.float64)
    signals = np.vstack([s for _, s in binned])
    return positions, signals


def _placed_signals(prepared, grid):
    """
    The signals the map actually places, keyed by (column, row).

    A position holding NaN, and a channel that is dead or partly NaN, is
    simply not placed - there is nothing to correlate either way.
    """
    n_cols, n_rows = grid.shape
    mono = {}
    for col in range(n_cols):
        for row in range(n_rows):
            here = grid[col, row]
            if np.isnan(here):
                continue
            signal = prepared[int(here)]
            if not np.all(np.isfinite(signal)) or np.linalg.norm(signal) < 1e-12:
                continue  # dead or NaN-masked channel, excluded from the fit
            mono[(col, row)] = signal
    return mono


def _projection(col, row, theta_deg, ied_m):
    """
    Where an electrode sits along the candidate direction.

    0 deg projects onto the ROW index, i.e. along a grid column, which is
    the axis global_amplitude differences along with diff_direction='cols'.
    """
    theta = np.radians(theta_deg)
    return (row * np.cos(theta) + col * np.sin(theta)) * ied_m


# ---------------------------------------------------------------------------
# the measurement: adjacent-bin delays
# ---------------------------------------------------------------------------

def _binned_signals(theta_deg, mono, ied_m, derivation):
    """
    One signal per projection bin along the direction, as (position, signal).

    Electrodes are binned at half the inter-electrode distance along the
    direction and averaged within a bin - the binning is ported from
    hdsemg-select. With derivation='SD' each bin signal is then replaced by
    its difference with the next one, which is what removes the common mode,
    see the module docstring.
    """
    bins = {}
    for (col, row), signal in mono.items():
        key = int(round(_projection(col, row, theta_deg, ied_m) / ied_m * 2))
        bins.setdefault(key, []).append(signal)

    averaged = [(key / 2.0 * ied_m, np.mean(signals, axis=0))
                for key, signals in sorted(bins.items())]

    if derivation == 'MP':
        return averaged
    if derivation != 'SD':
        raise ValueError(f"derivation must be 'SD' or 'MP', got {derivation!r}.")

    return [((p_i + p_j) / 2.0, sig_j - sig_i)
            for (p_i, sig_i), (p_j, sig_j) in zip(averaged, averaged[1:])]


def _binned_adjacent_delays(theta_deg, mono, ied_m, fs, derivation = 'SD'):
    """
    Signed delays between consecutive bin signals, and how far apart they sit.

      OUTPUT
        delays_ms   ... signed delay of each consecutive pair [ms]
        midpoints_m ... where each pair sits, for locating a reversal [m]
        spacings_m  ... how far apart the two signals of each pair are [m]
        peaks       ... normalised correlation at each of those delays []
    """
    signals = _binned_signals(theta_deg, mono, ied_m, derivation)

    delays_ms, midpoints_m, spacings_m, peaks = [], [], [], []
    for (p_i, sig_i), (p_j, sig_j) in zip(signals, signals[1:]):
        tau, peak = _xcorr_delay(sig_i, sig_j, fs)
        if tau is None:
            continue
        delays_ms.append(tau * 1000.0)
        midpoints_m.append((p_i + p_j) / 2.0)
        spacings_m.append(p_j - p_i)
        peaks.append(peak)

    return (np.asarray(delays_ms), np.asarray(midpoints_m),
            np.asarray(spacings_m), np.asarray(peaks))


def _xcorr_delay(s_i, s_j, fs):
    """
    (delay of s_j against s_i, normalised correlation at that delay), or
    (None, 0.0) when either signal is dead.

    The peak HEIGHT matters as much as its position: it says how much the
    two bins actually have in common, which is what separates the true
    fibre direction from an oblique one that merely happens to produce
    equal delays, see _score_direction.
    """
    norm_i = np.linalg.norm(s_i)
    norm_j = np.linalg.norm(s_j)
    if norm_i < 1e-12 or norm_j < 1e-12:
        return None, 0.0
    xc = correlate(s_i, s_j, mode="full") / (norm_i * norm_j)
    peak = int(np.argmax(xc))
    return (peak - (len(s_j) - 1)) / fs, float(xc[peak])


def _physiological(delays_ms, spacings_m):
    """
    Which bin pairs imply a velocity a muscle fibre could actually have,
    and what those velocities are.

    A zero delay implies an infinite velocity and is rejected here, which is
    how a direction ACROSS the fibres - where every delay is zero - is kept
    from winning the search.
    """
    if delays_ms.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        velocities = np.abs(spacings_m) / (np.abs(delays_ms) * 1e-3)
    valid = np.isfinite(velocities) & (velocities >= MIN_CV_MS) & (velocities <= MAX_CV_MS)
    return valid, velocities


def _score_direction(delays_ms, spacings_m, peaks):
    """
    (score, CV, nValidPairs) of one candidate direction.

      score ... the product of three factors over the physiological pairs,
                 each 0 to 1:
                  x CONSISTENCY, (mean|d|)^2 / mean(d^2). This is 1 exactly
                     when every delay has the same magnitude, which is what
                     a single conduction velocity along the fibre produces,
                     and it is BLIND TO SIGN, so an innervation zone does
                     not lower it
                  x COHERENCE, the mean normalised cross-correlation at
                     those delays. Consistency alone is NOT enough: an
                     oblique direction bins one or two electrodes per bin,
                     and the sample quantisation of the delay then makes
                     several of them land on the same value by accident.
                     Measured on the synthetic innervation-zone grid, a
                     direction 87 deg off scored consistency 1.000 - and
                     coherence 0.652, against 0.998 at the true direction,
                     which is what separates them
                  x COVERAGE, the share of pairs that were physiological at
                     all, so a direction with two lucky pairs cannot beat
                     one with twelve
      CV    ... median of those pairs' velocities, NaN if there are none
    """
    valid, velocities = _physiological(delays_ms, spacings_m)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return 0.0, np.nan, 0

    magnitudes = np.abs(delays_ms[valid])
    consistency = float(np.mean(magnitudes) ** 2 / np.mean(magnitudes ** 2))
    coherence = float(np.clip(np.mean(peaks[valid]), 0.0, 1.0))
    coverage = n_valid / delays_ms.size
    score = consistency * coherence * coverage
    return score, float(np.median(velocities[valid])), n_valid


def _cv_on_one_side(delays_ms, midpoints_m, spacings_m, iz_position_m):
    """
    The velocity over the side of the innervation zone holding more bin
    pairs, as a cross-check on the whole-grid median.

    Only pairs that do not straddle the reversal are used. Returns None when
    neither side can offer MIN_VALID_PAIRS physiological pairs.
    """
    before = midpoints_m < iz_position_m
    after = ~before
    for side in sorted((before, after), key=np.count_nonzero, reverse=True):
        valid, velocities = _physiological(delays_ms[side], spacings_m[side])
        if np.count_nonzero(valid) >= MIN_VALID_PAIRS:
            return float(np.median(velocities[valid]))
    return None


def _detect_iz(delays_ms, midpoints_m):
    """
    The first propagation reversal, interpolated between the two pairs.

    Near-zero delays are skipped so noise around a bin boundary cannot look
    like a reversal. Ported unchanged from hdsemg-select.
    """
    if len(delays_ms) < 2:
        return None
    threshold = max(float(np.max(np.abs(delays_ms))) * 0.1, 0.2)
    for k in range(len(delays_ms) - 1):
        here, there = delays_ms[k], delays_ms[k + 1]
        if abs(here) < threshold or abs(there) < threshold:
            continue
        if here * there < 0:
            fraction = abs(here) / (abs(here) + abs(there))
            return float(midpoints_m[k] + fraction * (midpoints_m[k + 1] - midpoints_m[k]))
    return None


# ---------------------------------------------------------------------------
# the hdsemg-select regression, kept for continuity and as an IZ signature
# ---------------------------------------------------------------------------

def _anchor_pairs(theta_deg, mono, ied_m):
    """
    Pair every electrode with the one furthest upstream along the direction.

    Pairing against a common anchor rather than between neighbours gives
    n-1 pairs whose distances increase monotonically, so the regression
    always has varying x whatever the angle. Ported unchanged from
    hdsemg-select.
    """
    projected = sorted(
        ((_projection(col, row, theta_deg, ied_m), (col, row)) for col, row in mono),
        key=lambda entry: entry[0],
    )
    if len(projected) < 2:
        return []

    anchor_p, anchor_pos = projected[0]
    pairs = []
    for p, pos in projected[1:]:
        distance = p - anchor_p
        if distance > 1e-9:
            pairs.append((anchor_pos, pos, distance))
    return pairs


def _anchor_regression_r2(theta_deg, mono, ied_m, fs):
    """
    R^2 of delay against distance from a common anchor, hdsemg-select's own
    selection criterion.

    Reported, never selected on: it is the quantity an innervation zone
    destroys, which is exactly what makes it worth showing next to
    propagation_score.
    """
    distances, delays = [], []
    for pos_i, pos_j, distance in _anchor_pairs(theta_deg, mono, ied_m):
        tau, _ = _xcorr_delay(mono[pos_i], mono[pos_j], fs)
        if tau is None or abs(tau) < 1e-9:
            continue
        implied_cv = distance / abs(tau)
        if implied_cv < MIN_CV_MS or implied_cv > MAX_CV_MS:
            continue
        distances.append(distance)
        delays.append(tau)

    if len(delays) < MIN_VALID_PAIRS:
        return 0.0

    d = np.asarray(distances)
    t = np.asarray(delays)
    if np.ptp(d) < 1e-9:
        mean_t = float(np.mean(t))
        if abs(mean_t) < 1e-9:
            return 0.0
        return max(0.0, 1.0 - float(np.var(t)) / (mean_t ** 2 + 1e-30))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = linregress(d, t)
    return float(fit.rvalue ** 2) if np.isfinite(fit.rvalue) else 0.0


def _best_angle_index(search_score):
    """
    The MIDDLE of the best-scoring plateau, not its first index.

    The half-IED binning quantises the projection, so a whole run of
    neighbouring directions bins the electrodes identically and scores
    within numerical noise of each other. Taking argmax then reports
    whichever edge of that plateau came first - on a 4-column grid whose
    true direction is 0 deg it reported -4 deg. The middle of the plateau
    is the honest answer, and it is what the run below returns.
    """
    if search_score.size == 0:
        return 0
    top = int(np.argmax(search_score))
    best = search_score[top]
    if not np.isfinite(best) or best <= 0:
        return top

    near = search_score >= best * (1.0 - 1e-6)
    lower = top
    while lower - 1 >= 0 and near[lower - 1]:
        lower -= 1
    upper = top
    while upper + 1 < near.size and near[upper + 1]:
        upper += 1
    return (lower + upper) // 2


def _not_estimable(search_angles, n_electrodes):
    """The result for a grid that places too few live electrodes to fit anything."""
    return PropagationResult(
        fiber_angle_deg=np.nan,
        propagation_score=0.0,
        conduction_velocity_ms=np.nan,
        cv_side_ms=None,
        cv_reported_ms=np.nan,
        cv_status="too_few_pairs",
        r_squared=0.0,
        iz_detected=False,
        iz_position_m=None,
        search_angles=search_angles,
        search_score=np.zeros(search_angles.size),
        search_cv_ms=np.full(search_angles.size, np.nan),
        pairwise_delays_ms=np.asarray([]),
        pairwise_distances_m=np.asarray([]),
        n_valid_pairs=0,
        n_electrodes=n_electrodes,
    )
