"""
AMPLITUDE_MAP: where under a grid the signal is strong, and where the end
plate is

  A SPATIAL picture of one grid over one short epoch: the differential
  amplitude at every electrode position, laid out the way the electrodes
  physically sit on the muscle. It answers a different question from
  propagation():

    propagation()   ... WHEN each position fires, so a direction and a
                         velocity, read from delays
    amplitude_map() ... HOW STRONG each position is, so an innervation zone
                         and a barycentre, read from amplitudes

  WHY THE INNERVATION ZONE IS FOUND TWICE, TWO DIFFERENT WAYS
    They are independent estimators of the same thing and they fail
    differently, which is the point of having both.

    Under an end-plate region the potentials leave in both directions, so
    the two single differentials either side of it subtract two waves of
    OPPOSITE travel and partly cancel: the differential amplitude has a
    local MINIMUM there. That is what this module looks for, and it needs no
    delay estimate at all, so it still works on an epoch too short or too
    noisy for cross-correlation.

    propagation() instead finds the position where the adjacent-bin delays
    REVERSE SIGN. That needs a usable delay at every bin, but it is immune
    to an amplitude dip caused by anything other than an end plate - a
    lifted electrode, a subcutaneous fat pocket, a bad connector row.

    Where they agree, the innervation zone is real. Where they do not, the
    disagreement is the finding, and neither should be quoted alone.

  THE DIP RULE IS NOT SIMPLY THE MINIMUM
    Taking min() down each column finds the lowest value, which at the edge
    of a grid is usually just the last electrode rather than an end plate.
    The rule here, ported from the reference MATLAB implementation, takes
    the most prominent LOCAL minimum, so a monotonic run down a column
    yields nothing rather than yielding its endpoint. A line is then only
    accepted where it stays continuous within half an inter-electrode
    distance from one column to the next, because an end plate is a band
    across the muscle, not a scatter of unrelated dips.

  VALIDATED AGAINST THE REFERENCE IMPLEMENTATION
    Checked against a MATLAB run over 116 consecutive 125 ms epochs of a
    32-channel 4x8 recording, matching its 30-450 Hz bandpass: the
    innervation zone position agrees to a median of -0.04 mm, within 1 mm on
    79 of 116 epochs and within 2 mm on 87. The barycentre agrees to 0.5 mm.

    The interpolation kernel had to match to get there. MATLAB's interp2
    'cubic' is a bicubic CONVOLUTION (Keys, a = -0.5), and substituting an
    interpolating spline moved the agreement to a median of 0.34 mm and
    within 1 mm on only 56 epochs. The reason is not smoothing: where a
    column holds two nearly equal dips, the two surfaces disagree about
    which is DEEPER and the detection jumps to the other dip. On the worst
    epoch, where 210 of 301 interpolated columns held two competing minima,
    the spline landed 33 mm from MATLAB's answer and the Keys kernel lands
    2.8 mm from it.

  NOTHING HERE RETURNS A VERDICT, as everywhere else in this package.

  >>> amap = amplitude_map(emg.T, emg_map, ied_mm = 10.0, fs = fs,
  ...                      window = peak_window)
  >>> iz = innervation_zone_line(upsample_map(amap))
  >>> iz.center_xy_mm
  (15.0, 25.94)

  (c) H Penasso. Written for hdsemg-shared by Claude Opus 5, 2026-08-21.
"""

from typing import NamedTuple

import numpy as np
from scipy.signal import find_peaks

from hdsemg_shared.preprocessing.grid_map import _as_map, _check_indices
from hdsemg_shared.quality.channel_metrics import (
    DEFAULT_PAD_S,
    _apply_window,
    _bandpass,
    _merged_bpf,
)

#: Upsampled step used by the reference implementation, and a sensible
#: default: fine enough that the dip position is limited by the data rather
#: than by the grid it is read off.
DEFAULT_UPSAMPLE_MM = 0.1

#: How far the dip may move from one electrode column to the next before the
#: line is judged to be unrelated dips rather than one end-plate band. Half
#: an inter-electrode distance, as in the reference implementation.
IZ_CONTINUITY_FRACTION = 0.5


class AmplitudeMap(NamedTuple):
    """
    One grid's differential amplitude laid out in space.

      values      ... matrix nY-by-nX; amplitude at each differential
                       position, NaN where the map has no electrode [xV]
      x_mm        ... vector nX; electrode COLUMN position, the axis the
                       fibres run ACROSS [mm]
      y_mm        ... vector nY; position ALONG a grid column, the axis the
                       potentials travel on. Offset by half an
                       inter-electrode distance for SD, by a whole one for
                       DD, because a differential sits BETWEEN electrodes
                       [mm]
      derivation  ... char; 'MP', 'SD' or 'DD' []
      measure     ... char; 'RMS' or 'ARV' []
      n_samples   ... how many samples went into each amplitude [-]
      ied_mm      ... the ELECTRODE spacing this map was built from, carried
                       along so that it survives upsampling. After
                       upsample_map the spacing of x_mm is no longer the
                       electrode spacing, and anything that needs the real
                       one - the innervation zone continuity test - would
                       otherwise read it off the axis and get it wrong by
                       two orders of magnitude [mm]

    *INFO* ... x_mm and y_mm are PHYSICAL grid positions, with column 0 at
                x = 0. A figure that mirrors the x axis to match how the
                grid faces the viewer is free to do so, but must say it has,
                because a mirrored map swaps medial and lateral.
    """
    values: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    derivation: str
    measure: str
    n_samples: int
    ied_mm: float


class InnervationZone(NamedTuple):
    """
    Where the amplitude dips, read column by column.

      y_mm        ... vector nX; the dip position in each column, NaN where
                       the column held no local minimum or where continuity
                       to its neighbour broke [mm]
      x_mm        ... vector nX; the matching column positions [mm]
      center_xy_mm... tuple; mean x and mean y over the longest continuous
                       run, (nan, nan) when no run survived [mm]
      n_columns   ... how many columns the map has [-]
      columns_covered ... how many of them the accepted run spans. A run
                       narrower than the grid is a PARTIAL detection and the
                       centre it produces is the centre of that part, not of
                       the grid [-]
      full_width  ... logic; whether the run spans every column []
      angle_deg   ... the fibre direction's tilt away from the grid COLUMN
                       axis, from a straight-line fit to the accepted run.
                       An end plate is a band ACROSS the fibres, so the
                       band's tilt away from the row direction IS the
                       fibres' tilt away from the column direction. NaN when
                       no run survived [deg]
      fit_y_mm    ... vector nX; that straight line, NaN outside the
                       accepted run, for drawing [mm]

    *INFO* ... angle_deg is what corrects a velocity measured DOWN a grid
                column, see angle_corrected_velocity. It is an estimate of
                the fibre direction independent of propagation()'s angle
                search, and it VARIES LESS between trials of a grid that
                never moved - over 116 consecutive epochs of one recording,
                SD 11.7 deg against the search's 28.1 deg.

                Read that steadiness carefully. The band's tilt is measured
                ACROSS the columns, so a grid few columns wide under-reads
                it and collapses toward zero, which also looks steady. On
                twelve columns a planted 30 deg is recovered to within 6;
                on four it is not, while the delay search still finds it.
                Below about eight columns, treat this angle as a weak prior
                and not as a correction to lean on.
    """
    y_mm: np.ndarray
    x_mm: np.ndarray
    center_xy_mm: tuple
    n_columns: int
    columns_covered: int
    full_width: bool
    angle_deg: float
    fit_y_mm: np.ndarray


def amplitude_map(emg_channels, emg_map, ied_mm, fs, window = None,
                  derivation = 'SD', measure = 'RMS', bpf = None,
                  pad_s = DEFAULT_PAD_S):
    """
    AMPLITUDE_MAP returns the differential amplitude at every grid position

      amap = amplitude_map(emg.T, emg_map, 10.0, fs, window = peak_window)

      Differences are taken ALONG each grid column, between electrodes
      adjacent in the map's inner index, which is the direction the
      potentials travel when the grid is aligned with the fibres. This is
      the same convention propagation() calls 0 degrees.

      INPUT
        emg_channels ... matrix, each row one channel's RAW signal over
                          time [xV]
        emg_map      ... the grid's channel map, see propagation(): outer
                          index the grid COLUMN, inner index the row, base-0
                          channel numbers, NaN where the grid has no
                          electrode []
        ied_mm       ... inter-electrode distance [mm]
        fs           ... sampling frequency [Hz]

      OPTIONAL INPUT
        window       ... slice; the epoch to measure over, normally a short
                          one centred on the burst. None uses all of it []
        derivation   ... char; 'SD' (default), 'DD' or 'MP' []
        measure      ... char; 'RMS' (default) or 'ARV' []
        bpf          ... bandpass options, see channel_metrics. False skips
                          filtering []
        pad_s        ... filter padding [s]

      OUTPUT
        amap         ... AmplitudeMap; see that type

      *INFO* ... the amplitude is measured AFTER windowing, so a window
                  short enough to sit inside one burst gives the picture at
                  that instant, and a window over the whole contraction
                  gives the average one. The reference implementation steps
                  a 125 ms window across the trial to make a film of it.
    """
    x = np.asarray(emg_channels, dtype=np.float64)
    grid = _as_map(emg_map)
    _check_indices(grid, x.shape[0])

    ied = float(ied_mm)
    if not ied > 0:
        raise ValueError("ied_mm must be positive")
    if derivation not in ('MP', 'SD', 'DD'):
        raise ValueError("derivation must be 'MP', 'SD' or 'DD'")
    if measure not in ('RMS', 'ARV'):
        raise ValueError("measure must be 'RMS' or 'ARV'")

    prepared = x if bpf is False else _bandpass(x, _merged_bpf(bpf), fs, pad_s)
    prepared = _apply_window(prepared, window)

    n_cols, n_rows = grid.shape
    step = {'MP': 0, 'SD': 1, 'DD': 2}[derivation]
    n_out = n_rows - step
    if n_out < 1:
        raise ValueError(
            f"a {n_rows}-row grid cannot carry a {derivation} derivation")

    values = np.full((n_out, n_cols), np.nan)
    for c in range(n_cols):
        for r in range(n_out):
            sig = _derivative(prepared, grid, c, r, step)
            if sig is not None:
                values[r, c] = _amplitude(sig, measure)

    return AmplitudeMap(
        values=values,
        x_mm=np.arange(n_cols, dtype=np.float64) * ied,
        y_mm=(np.arange(n_out, dtype=np.float64) + step / 2.0) * ied,
        derivation=derivation,
        measure=measure,
        n_samples=int(prepared.shape[1]),
        ied_mm=ied,
    )


def upsample_map(amap, step_mm = DEFAULT_UPSAMPLE_MM):
    """
    UPSAMPLE_MAP interpolates a map onto a fine grid, for drawing and for
    locating the dip below electrode spacing

      fine = upsample_map(amap)

      INPUT
        amap    ... AmplitudeMap from amplitude_map() []
        step_mm ... spacing of the interpolated grid [mm]

      OUTPUT
        amap    ... AmplitudeMap on the finer axes []

      *INFO* ... Keys cubic CONVOLUTION in both directions, the kernel
                  MATLAB's interp2(..., 'cubic') uses, not an interpolating
                  spline - see _resample_axis for why that distinction
                  changes answers here. The interpolated surface can
                  overshoot the measured range slightly. That is visible in
                  the reference implementation too, where the colour bar
                  runs below zero on a strictly positive quantity. Missing
                  positions are filled from their neighbours BEFORE
                  interpolating, because a NaN would otherwise spread over
                  the whole surface; a map too small to interpolate is
                  returned unchanged.
    """
    z = np.array(amap.values, dtype=np.float64)
    if z.shape[0] < 2 or z.shape[1] < 2 or np.count_nonzero(np.isfinite(z)) < 4:
        return amap

    z = _fill_missing(z)
    y = _fine_axis(amap.y_mm, step_mm)
    x = _fine_axis(amap.x_mm, step_mm)
    z = _resample_axis(z, y.size, axis=0)
    z = _resample_axis(z, x.size, axis=1)
    return amap._replace(values=z, x_mm=x, y_mm=y)


def innervation_zone_line(amap, ied_mm = None):
    """
    INNERVATION_ZONE_LINE finds the amplitude dip that marks the end plate

      iz = innervation_zone_line(upsample_map(amap), ied_mm = 10.0)

      Per column the most prominent LOCAL minimum, then the longest run of
      columns over which that dip moves by no more than half an
      inter-electrode distance at a time.

      INPUT
        amap   ... AmplitudeMap, normally upsampled first so the dip can be
                    placed between electrodes []

      OPTIONAL INPUT
        ied_mm ... inter-electrode distance, used only for the continuity
                    tolerance. None takes the map's own ied_mm, which
                    survives upsampling, and is what you want [mm]

      OUTPUT
        iz     ... InnervationZone; see that type

      *INFO* ... this is only meaningful on a DIFFERENTIAL map. A monopolar
                  amplitude has no end-plate minimum to find, so 'MP' is
                  rejected rather than answered.
    """
    if amap.derivation == 'MP':
        raise ValueError(
            "an innervation zone shows as a dip in a DIFFERENTIAL amplitude; "
            "a monopolar map has no such dip, so pass an 'SD' or 'DD' map")

    z = np.asarray(amap.values, dtype=np.float64)
    n_cols = z.shape[1]
    tol = _continuity_tolerance(amap, ied_mm)

    y_iz = np.full(n_cols, np.nan)
    for c in range(n_cols):
        col = z[:, c]
        if not np.all(np.isfinite(col)):
            continue
        depth = col.max() - col
        peaks, _ = find_peaks(depth)
        if peaks.size:
            y_iz[c] = amap.y_mm[peaks[int(np.argmax(depth[peaks]))]]

    broken = np.nonzero(np.abs(np.diff(y_iz)) > tol)[0]
    y_iz[broken] = np.nan

    start, stop = _longest_run(y_iz)
    covered = stop - start
    if covered:
        center = (float(np.mean(amap.x_mm[start:stop])),
                  float(np.mean(y_iz[start:stop])))
    else:
        center = (float('nan'), float('nan'))
    y_iz[:start] = np.nan
    y_iz[stop:] = np.nan

    x_mm = np.asarray(amap.x_mm, dtype=np.float64)
    angle_deg, fit_y = _iz_line_fit(x_mm, y_iz, start, stop)

    return InnervationZone(
        y_mm=y_iz,
        x_mm=x_mm,
        center_xy_mm=center,
        n_columns=n_cols,
        columns_covered=int(covered),
        full_width=bool(covered == n_cols),
        angle_deg=angle_deg,
        fit_y_mm=fit_y,
    )


def angle_corrected_velocity(cv_ms, angle_deg):
    """
    ANGLE_CORRECTED_VELOCITY turns a velocity measured DOWN A GRID COLUMN
    into one along the muscle fibres

      cv = angle_corrected_velocity(cv_along_column, iz.angle_deg)

      Two electrodes one inter-electrode distance apart down a column are
      only ied*cos(angle) apart ALONG the fibre when the fibres run at
      `angle` to that column. The delay between them measures the shorter
      distance, so dividing by the full ied - which is what an uncorrected
      estimate does - reports a velocity too FAST by 1/cos(angle).

      INPUT
        cv_ms     ... velocity measured along the grid column axis [m/s]
        angle_deg ... the fibres' tilt from that axis, normally
                       InnervationZone.angle_deg [deg]

      OUTPUT
        cv_ms     ... velocity along the fibres, NaN where the angle is
                       unknown or so near 90 deg that nothing is measurable
                       down a column [m/s]

      *INFO* ... this correction is SMALL until the grid is badly rotated:
                  10 deg costs 1.5 %, 20 deg costs 6 %, and only past 30 deg
                  does it exceed 13 %. It is therefore not a way to reconcile
                  two velocity estimates that disagree by tens of per cent -
                  a disagreement that size is a difference of METHOD, not of
                  geometry. Applying it to a velocity that was already
                  measured along the fibre direction, as propagation() does
                  at its own fiber_angle_deg, corrects twice and is wrong.
    """
    cv = float(cv_ms)
    angle = float(angle_deg)
    if not np.isfinite(cv) or not np.isfinite(angle):
        return float('nan')
    cosine = np.cos(np.radians(angle))
    if abs(cosine) < 1e-3:
        return float('nan')
    return cv * float(cosine)


def barycenter(amap):
    """
    BARYCENTER returns the amplitude-weighted centre of a map

      x_mm, y_mm = barycenter(amap)

      INPUT
        amap  ... AmplitudeMap []

      OUTPUT
        x_mm  ... amplitude-weighted mean column position [mm]
        y_mm  ... amplitude-weighted mean position along the column [mm]

      *INFO* ... where the signal sits under the grid. It drifts when a grid
                  slips, and it is NOT an innervation zone: the barycentre
                  follows the strongest signal, the end plate the weakest.
    """
    z = np.asarray(amap.values, dtype=np.float64)
    live = np.isfinite(z) & (np.nan_to_num(z) > 0)
    if not live.any():
        return float('nan'), float('nan')
    w = np.where(live, z, 0.0)
    total = w.sum()
    return (float((w.sum(0) * amap.x_mm).sum() / total),
            float((w.sum(1) * amap.y_mm).sum() / total))


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _derivative(prepared, grid, col, row, step):
    """The MP, SD or DD signal at one map position, or None if unwired."""
    idx = []
    for offset in range(step + 1):
        ch = grid[col, row + offset]
        if not np.isfinite(ch):
            return None
        idx.append(int(ch))
    if step == 0:
        return prepared[idx[0]]
    if step == 1:
        return prepared[idx[1]] - prepared[idx[0]]
    return prepared[idx[2]] - 2.0 * prepared[idx[1]] + prepared[idx[0]]


def _amplitude(sig, measure):
    if sig.size == 0:
        return np.nan
    if measure == 'RMS':
        return float(np.sqrt(np.mean(sig ** 2)))
    return float(np.mean(np.abs(sig)))


def _fill_missing(z):
    """
    Replace NaN positions by interpolating along each axis and averaging the
    two, so a single unwired electrode does not blank the whole surface.
    """
    if np.all(np.isfinite(z)):
        return z
    filled = []
    for axis in (0, 1):
        work = np.array(np.moveaxis(z, axis, 0), dtype=np.float64)
        for k in range(work.shape[1]):
            line = work[:, k]
            live = np.isfinite(line)
            if live.sum() >= 2:
                line[~live] = np.interp(
                    np.flatnonzero(~live), np.flatnonzero(live), line[live])
        filled.append(np.moveaxis(work, 0, axis))
    out = np.nanmean(np.stack(filled), axis=0)
    if np.any(~np.isfinite(out)):
        out[~np.isfinite(out)] = float(np.nanmean(z))
    return out


def _iz_line_fit(x_mm, y_iz, start, stop):
    """
    A straight line through the accepted innervation zone run, and the angle
    it implies for the fibres.

    Least squares over the run only. Fewer than two DISTINCT x positions
    cannot define a direction, and a run that is one interpolated column
    wide would otherwise return a confident-looking angle from noise.
    """
    nan_line = np.full(x_mm.size, np.nan)
    if stop - start < 2:
        return float('nan'), nan_line

    xs, ys = x_mm[start:stop], y_iz[start:stop]
    if np.ptp(xs) <= 0 or not np.all(np.isfinite(ys)):
        return float('nan'), nan_line

    slope, intercept = np.polyfit(xs, ys, 1)
    fit = nan_line.copy()
    fit[start:stop] = slope * xs + intercept

    # The SIGN, which is easy to get backwards and invisible at small tilts.
    # The fitted band runs along (1, slope), i.e. at +arctan(slope) from the
    # row axis. The fibres run PERPENDICULAR to it, along (-sin, cos) of that
    # same angle - and propagation() names a direction by the theta for which
    # it projects onto (sin theta, cos theta). Matching the two gives
    # theta = -arctan(slope), so the returned angle is directly comparable
    # with fiber_angle_deg rather than its mirror image.
    return float(-np.degrees(np.arctan(slope))), fit


def _keys_weights(s):
    """
    Keys bicubic CONVOLUTION weights, a = -0.5, for the four taps around a
    fractional position s in [0, 1).
    """
    def kernel(x):
        x = np.abs(x)
        out = np.zeros_like(x)
        inner = x <= 1.0
        out[inner] = 1.5 * x[inner] ** 3 - 2.5 * x[inner] ** 2 + 1.0
        outer = (x > 1.0) & (x < 2.0)
        out[outer] = -0.5 * x[outer] ** 3 + 2.5 * x[outer] ** 2 - 4.0 * x[outer] + 2.0
        return out

    s = np.asarray(s, dtype=np.float64)
    return np.stack([kernel(s + 1.0), kernel(s), kernel(s - 1.0),
                     kernel(s - 2.0)], axis=-1)


def _resample_axis(values, n_out, axis):
    """
    Resample one axis by Keys cubic convolution, the kernel MATLAB's
    interp2(..., 'cubic') uses.

    NOT an interpolating spline. The difference is not cosmetic on this
    measurement: where a grid column holds two nearly equal amplitude dips,
    the two surfaces can disagree about which is DEEPER, and the innervation
    zone then jumps to the other dip rather than shifting slightly. Measured
    against a MATLAB reference run of 116 epochs, the spline agreed to a
    median of 0.34 mm and within 1 mm on 56 epochs; this kernel agrees to
    -0.04 mm and within 1 mm on 79. On the worst epoch - one where 210 of 301
    interpolated columns held two competing minima - the spline picked a dip
    33 mm away from MATLAB's, and this picks the same one, to 2.8 mm.

    Ends are extrapolated as 3*a0 - 3*a1 + a2, again as interp2 does. An axis
    too short for that falls back to linear.
    """
    work = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    n_in = work.shape[0]
    if n_in < 2 or n_out < 1:
        return values

    t = np.linspace(0.0, n_in - 1, n_out)
    k = np.clip(np.floor(t).astype(int), 0, n_in - 2)

    if n_in < 3:
        s = (t - k).reshape((-1,) + (1,) * (work.ndim - 1))
        return np.moveaxis(work[k] * (1.0 - s) + work[k + 1] * s, 0, axis)

    padded = np.concatenate([
        (3.0 * work[0] - 3.0 * work[1] + work[2])[None],
        work,
        (3.0 * work[-1] - 3.0 * work[-2] + work[-3])[None]])
    weights = _keys_weights(t - k)
    taps = padded[k[:, None] + np.arange(4)[None, :]]
    return np.moveaxis(np.einsum("ij,ij...->i...", weights, taps), 0, axis)


def _fine_axis(axis, step_mm):
    """
    The interpolated axis, ending EXACTLY on the last electrode.

    arange with a float step accumulates rounding and lands a fraction short
    of the endpoint, which then becomes the extent a figure draws the map
    over: the surface would be shown very slightly narrower than the span it
    was measured across. linspace pins both ends instead.
    """
    start, end = float(axis[0]), float(axis[-1])
    n = int(round((end - start) / float(step_mm))) + 1
    return np.linspace(start, end, max(n, 2))


def _continuity_tolerance(amap, ied_mm):
    spacing = float(ied_mm) if ied_mm is not None else float(amap.ied_mm)
    if not spacing > 0:
        return np.inf
    return spacing * IZ_CONTINUITY_FRACTION


def _longest_run(values):
    """Half-open bounds of the longest finite run, (0, 0) when there is none."""
    best = (0, 0)
    i = 0
    n = values.size
    while i < n:
        if not np.isfinite(values[i]):
            i += 1
            continue
        j = i
        while j < n and np.isfinite(values[j]):
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    return best
