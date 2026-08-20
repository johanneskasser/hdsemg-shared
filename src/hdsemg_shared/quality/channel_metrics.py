"""
CHANNEL_METRICS: per-channel quality numbers of an HDsEMG grid

  Every function here answers ONE question about ONE channel and returns a
  NUMBER, never a verdict. Where the threshold sits is a property of the
  study, the electrode and the muscle, not of the measurement, so it is the
  caller who decides what counts as bad. Keeping the two apart is what lets
  a quality gate record its evidence next to its decision and re-derive one
  without the other.

  The channel matrix is channels-by-samples throughout, and an emg_map
  follows preprocessing.grid_map's convention: outer index = grid COLUMN,
  inner index = ROW, base-0 channel numbers, NaN where no electrode sits.

  A channel that is entirely NaN (an excluded or unwired position) yields
  NaN for every measure rather than raising, so a map with gaps can be
  passed straight through.

  WHY THESE SIX
    x flat            ... a disconnected or saturated electrode carries no
                           signal at all and drags a grid-wide mean square
                           DOWN in proportion to how many there are
    x amplitude       ... contact quality, and the input to a within-grid
                           outlier score
    x spectrum        ... MNF/MDF far from the grid's own median means
                           movement artefact (low) or noise (high)
    x line noise      ... mains pickup is narrow-band and does not move with
                           the muscle, so a ratio against the local
                           background separates it from real signal
    x clipping        ... an amplifier at its rail reports a number that is
                           not the signal
    x neighbour corr. ... a real channel shares its motor unit action
                           potentials with the electrodes next to it; an
                           isolated noisy one does not

  (c) H Penasso. flat_channels moved here from muEccCon's
  global_amplitude.bad_channels (itself the NaN-masking of
  SP_AM_CV_HDsEMG_Analysis.m, lines 308-336). The rest written for
  hdsemg-shared by Claude Opus 5, 2026-08-19.
"""

import warnings
from typing import NamedTuple

import numpy as np
from scipy.fft import rfft, rfftfreq

from hdsemg_shared.filters.bandpass import bandpass_filter, bandpass_filter_exact_corners
from hdsemg_shared.filters.padding import pad_samples, reflect_pad, trim_pad
from hdsemg_shared.global_parameters.MDF import compute_mdf
from hdsemg_shared.global_parameters.MNF import compute_mnf
from hdsemg_shared.preprocessing.grid_map import _as_map, _check_indices

#: Band-pass applied before the amplitude and neighbour-correlation measures,
#: the same band global_amplitude uses so the numbers are comparable.
DEFAULT_BPF = {'N': 2, 'fcl': 15.0, 'fch': 450.0, 'corners': 'exact'}

#: Seconds of even reflection added at both ends before filtering.
DEFAULT_PAD_S = 0.25


class ChannelAmplitude(NamedTuple):
    """
    rms ... each channel's root-mean-square over the window [xV]
    arv ... each channel's average rectified value over the window [xV]
    """

    rms: np.ndarray
    arv: np.ndarray


class ChannelSpectrum(NamedTuple):
    """
    mnf ... each channel's mean frequency [Hz]
    mdf ... each channel's median frequency [Hz]
    """

    mnf: np.ndarray
    mdf: np.ndarray


class LineNoise(NamedTuple):
    """
    ratio         ... per channel, the LARGEST of the per-frequency ratios []
    per_frequency ... nFreqs-by-nChannels, one ratio per line frequency []
    frequencies   ... the line frequencies that were tested [Hz]
    """

    ratio: np.ndarray
    per_frequency: np.ndarray
    frequencies: np.ndarray


def flat_channels(mat, rel_tol = 1e-3, abs_tol = 0.0):
    """
    FLAT_CHANNELS finds channels that carry no signal at all

      dead = flat_channels(emg.T)

      INPUT
        mat     ... matrix, each row is one channel's signal over time
                     (rows = channels, columns = samples) [xV]
        rel_tol ... a channel counts as flat when its standard deviation
                     over time is at or below rel_tol times the median
                     standard deviation of the channels that are not
                     themselves flat, default 1e-3 []
        abs_tol ... absolute floor for the same test, default 0.0, so an
                     exactly constant channel is always caught [xV]

      OUTPUT
        dead    ... sorted list of 0-based row indices that are flat []

      WHY, AND WHY THE THRESHOLD IS SAFE
        A flat channel is not a small signal, it is no signal: a
        disconnected or saturated electrode. Left in, it contributes zero
        energy to the mean square over the grid area and therefore drags the
        global amplitude DOWN in proportion to how many of them there are.
        Measured across five subjects of the muEccCon study, flat channels
        are always EXACTLY zero and nothing else comes within 20 % of the
        grid median, so the threshold sits in a wide empty gap and cannot
        plausibly misfire. On that data every flat channel had also been
        deselected by the reviewer already, so this mostly confirms the
        review - its value is the case where a channel goes flat and was NOT
        deselected, which would otherwise pass silently.

      *INFO* ... NaN-only rows are reported as flat too: they carry no
                  signal either, and the callers treat both the same way
    """
    x = np.asarray(mat, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"mat must be 2-D (channels x samples), got shape {x.shape}.")
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("rel_tol and abs_tol must be >= 0.")

    # An all-NaN row makes nanstd warn about zero degrees of freedom; that
    # case is handled deliberately just below, so the warning is only noise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(x, axis=1)
    sd = np.where(np.isfinite(sd), sd, 0.0)  # an all-NaN row carries no signal either

    alive = sd[sd > 0]
    reference = float(np.median(alive)) if alive.size else 0.0
    threshold = max(abs_tol, rel_tol * reference)
    return [int(i) for i in np.flatnonzero(sd <= threshold)]


def channel_amplitude(mat, fs, bpf = None, window = None, pad_s = DEFAULT_PAD_S):
    """
    CHANNEL_AMPLITUDE calculates each channel's own amplitude over a window

      amp = channel_amplitude(emg.T, fs)
      amp = channel_amplitude(emg.T, fs, window = slice(1000, 3000))

      INPUT
        mat    ... matrix, each row contains one channel's RAW signal over
                    time (rows = channels, columns = samples). Filtering
                    happens here, so pass it unfiltered [xV]
        fs     ... double or int, sampling frequency [Hz]

      OPTIONAL INPUT
        bpf    ... dict with band-pass options, keys:
                    x "N"       - filter order, even and >= 2 []
                    x "fcl"     - lower corner [Hz]
                    x "fch"     - upper corner [Hz]
                    x "corners" - "exact" or "prewarp"
                    None (default) means DEFAULT_BPF
        window ... slice, boolean mask over samples, or (start, stop) tuple
                    restricting WHICH samples are reduced. The filter always
                    runs on the whole record first, so the window never
                    creates an edge transient. None (default) uses all []
        pad_s  ... seconds of even reflection at both ends before filtering,
                    default 0.25 [s]

      OUTPUT
        amp    ... ChannelAmplitude(rms, arv), each a vector of length
                    nChannels, NaN for a channel that is entirely NaN [xV]

      *INFO* ... this is a per-channel number, NOT the global amplitude of
                  the grid. For that use global_parameters.global_amplitude,
                  which reduces across the channels and roots LAST
    """
    x = _as_channels(mat)
    filtered = _bandpass(x, _merged_bpf(bpf), fs, pad_s)
    windowed = _apply_window(filtered, window)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are expected
        rms = np.sqrt(np.nanmean(windowed ** 2, axis=1))
        arv = np.nanmean(np.abs(windowed), axis=1)
    return ChannelAmplitude(rms=rms, arv=arv)


def channel_spectrum(mat, fs, window = None):
    """
    CHANNEL_SPECTRUM calculates each channel's mean and median frequency

      spec = channel_spectrum(emg.T, fs)

      INPUT
        mat    ... matrix, each row contains one channel's signal over time
                    (rows = channels, columns = samples). NOT filtered here:
                    a band-pass would move MNF/MDF by construction, and the
                    point of this measure is to compare channels of the SAME
                    grid against each other [xV]
        fs     ... double or int, sampling frequency [Hz]

      OPTIONAL INPUT
        window ... slice, boolean mask over samples, or (start, stop) tuple,
                    see channel_amplitude. None (default) uses all []

      OUTPUT
        spec   ... ChannelSpectrum(mnf, mdf), each a vector of length
                    nChannels, NaN for a channel that carries no signal [Hz]

      *INFO* ... a channel whose MNF sits far BELOW its grid's median is
                  usually movement artefact or poor contact; far ABOVE it is
                  usually noise-dominated. Turn either into a score with
                  robust_z
    """
    x = _apply_window(_as_channels(mat), window)

    mnf = np.full(x.shape[0], np.nan)
    mdf = np.full(x.shape[0], np.nan)
    for i, channel in enumerate(x):
        finite = channel[np.isfinite(channel)]
        if finite.size == 0 or np.allclose(finite, 0.0):
            continue  # no signal, so no spectrum -- stays NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mnf[i] = compute_mnf(channel, fs)
            mdf[i] = compute_mdf(channel, fs)
    return ChannelSpectrum(mnf=mnf, mdf=mdf)


def line_noise_ratio(mat, fs, freqs = (50.0,), band_hz = 2.0, outer_hz = 20.0,
                     window = None):
    """
    LINE_NOISE_RATIO measures mains pickup against the spectrum immediately
    around it

      noise = line_noise_ratio(emg.T, fs)
      noise = line_noise_ratio(emg.T, fs, freqs = (50.0, 100.0, 150.0))

      INPUT
        mat      ... matrix, each row contains one channel's signal over
                      time (rows = channels, columns = samples) [xV]
        fs       ... double or int, sampling frequency [Hz]

      OPTIONAL INPUT
        freqs    ... line frequencies to test, default (50.0,). Pass the
                      harmonics too when they matter, e.g. (50, 100, 150) [Hz]
        band_hz  ... half-width of the peak window around each line
                      frequency, default 2.0 [Hz]
        outer_hz ... outer half-width of the background ring around it,
                      default 20.0. The background is what lies between
                      band_hz and outer_hz on either side [Hz]
        window   ... slice, boolean mask or (start, stop) tuple over samples []

      OUTPUT
        noise    ... LineNoise(ratio, per_frequency, frequencies). ratio is
                      the largest per-frequency ratio of that channel, NaN
                      for a channel that carries no signal []

      HOW IT IS MEASURED
        The MEAN power within +/- band_hz of the line frequency, divided by
        the MEDIAN power of the ring band_hz..outer_hz around it. A clean
        channel lands near 1.4; mains pickup lands one to two orders of
        magnitude above.

      WHY A LOCAL RING, AND WHY A MEAN OVER THE PEAK
        x LOCAL: sEMG power falls by orders of magnitude above the signal
           band, so a background taken over the WHOLE spectrum is dominated
           by near-empty high-frequency bins. That deflates the reference
           and makes every channel look contaminated - measured at 898x on
           a synthetic, mains-free, band-limited signal.
        x MEAN over the peak window rather than its maximum: the maximum of
           K noisy bins grows like log K, so a max-based ratio drifts with
           the record length and has no fixed clean value. A mean does not:
           for noise the mean/median of an exponential is 1/ln2 = 1.44
           whatever K is, which is why a clean channel has a stable reading
           to compare against.

      PORT
        The idea of hdsemg-select's
        select_logic/auto_flagger.py::_check_frequency_peak, returned as the
        RATIO instead of as a label - the threshold is the caller's.

      DIFFERENCES TO THE hdsemg-select ORIGINAL
        select takes its background from every bin outside the peak window
        and compares the window's MAXIMUM against it; both are changed here
        for the two reasons above.
    """
    x = _apply_window(_as_channels(mat), window)
    frequencies = np.asarray(freqs, dtype=np.float64).ravel()
    if frequencies.size == 0:
        raise ValueError("freqs must name at least one line frequency.")
    if band_hz <= 0:
        raise ValueError(f"band_hz must be positive, got {band_hz}.")
    if outer_hz <= band_hz:
        raise ValueError(f"outer_hz ({outer_hz}) must exceed band_hz ({band_hz}).")
    if np.any(frequencies + band_hz >= fs / 2):
        raise ValueError(
            f"freqs {frequencies.tolist()} +/- {band_hz} Hz must stay below the "
            f"Nyquist frequency {fs / 2} Hz."
        )

    per_frequency = np.full((frequencies.size, x.shape[0]), np.nan)
    bin_freqs = rfftfreq(x.shape[1], 1.0 / fs)
    peak_masks = [np.abs(bin_freqs - f) <= band_hz for f in frequencies]
    # The ring around the peak, DC excluded: local enough that the sEMG
    # spectrum's own slope does not bias it, wide enough to have a median
    background_masks = [
        (bin_freqs > 0) & (np.abs(bin_freqs - f) > band_hz)
        & (np.abs(bin_freqs - f) <= outer_hz)
        for f in frequencies
    ]

    for i, channel in enumerate(x):
        if not np.all(np.isfinite(channel)) or np.allclose(channel, 0.0):
            continue  # NaN or dead channel -- stays NaN
        power = np.abs(rfft(channel)) ** 2
        for j, (peak, background) in enumerate(zip(peak_masks, background_masks)):
            if not np.any(peak) or not np.any(background):
                continue
            median_background = float(np.median(power[background]))
            if median_background <= 0:
                continue
            per_frequency[j, i] = float(np.mean(power[peak])) / median_background

    all_nan = np.all(np.isnan(per_frequency), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are expected
        ratio = np.nanmax(per_frequency, axis=0)
    ratio = np.where(all_nan, np.nan, ratio)
    return LineNoise(ratio=ratio, per_frequency=per_frequency, frequencies=frequencies)


def clipping_fraction(mat, rel_tol = 1e-6):
    """
    CLIPPING_FRACTION measures how much of a channel sits at its own rail

      clipped = clipping_fraction(emg.T)

      INPUT
        mat     ... matrix, each row contains one channel's RAW, UNFILTERED
                     signal over time (rows = channels, columns = samples).
                     Raw on purpose: a band-pass rounds the flat top of a
                     clipped segment off and hides exactly what is looked
                     for here [xV]
        rel_tol ... how close to the channel's own extreme a sample has to
                     be to count as sitting at the rail, as a fraction of
                     that extreme, default 1e-6 []

      OUTPUT
        clipped ... vector of length nChannels, the fraction of finite
                     samples within rel_tol of the channel's most extreme
                     value, NaN for a channel that carries no signal []

      *INFO* ... an unclipped channel touches its own maximum ONCE, so a
                  clean recording lands at about 1/nSamples, not at 0. Judge
                  it against that floor: at 2 kHz over 20 s that is 2.5e-5,
                  so a threshold of 1e-3 is roughly 40 samples at the rail
      *INFO* ... this replaces hdsemg-select's auto_flagger._detect_artifact,
                  which flags variance ABOVE 1e-9 and therefore fires on
                  every live channel. Do not port that predicate
    """
    x = _as_channels(mat)
    if rel_tol < 0:
        raise ValueError(f"rel_tol must be >= 0, got {rel_tol}.")

    fraction = np.full(x.shape[0], np.nan)
    for i, channel in enumerate(x):
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            continue
        extreme = float(np.max(np.abs(finite)))
        if extreme <= 0:
            continue  # a dead channel is flat, not clipped -- stays NaN
        at_rail = np.abs(np.abs(finite) - extreme) <= rel_tol * extreme
        fraction[i] = float(np.count_nonzero(at_rail)) / finite.size
    return fraction


def neighbor_correlation(mat, emg_map, fs, bpf = None, window = None,
                         pad_s = DEFAULT_PAD_S):
    """
    NEIGHBOR_CORRELATION measures how much a channel shares with the
    electrodes immediately next to it on the grid

      r = neighbor_correlation(emg.T, emg_map, fs, window = peak_window)

      INPUT
        mat     ... matrix, each row contains one channel's RAW signal over
                     time (rows = channels, columns = samples). Filtering
                     happens here, so pass it unfiltered [xV]
        emg_map ... list of lists or 2-D array, outer index = grid column,
                     inner index = row, base-0 channel numbers into mat, NaN
                     where no electrode sits []
        fs      ... double or int, sampling frequency [Hz]

      OPTIONAL INPUT
        bpf     ... band-pass options, see channel_amplitude []
        window  ... slice, boolean mask or (start, stop) tuple over samples.
                     *STRONGLY RECOMMENDED*, see the warning below []
        pad_s   ... reflection padding before filtering, default 0.25 [s]

      OUTPUT
        r       ... vector of length nChannels (rows of mat), the LARGEST
                     Pearson correlation of that channel with any of its up
                     to four immediate grid neighbours (row and column
                     adjacent). NaN for a channel the map does not place, for
                     one with no live neighbour, and for a dead channel []

      *WARNING* PASS A HIGH-ACTIVITY WINDOW
        Over a whole force-tracking trial, which is mostly rest, neighbouring
        monopolar channels share little but their own uncorrelated noise, so
        r collapses toward zero for GOOD channels too. Measured over the
        whole record this separates nothing and would flag an entire grid.
        Restrict it to a window where the muscle is actually active - the
        peak of the target path, or the held plateau.

      *INFO* ... calibrate the threshold on data known to be good before
                  using it to reject anything; there is no universal value
    """
    x = _as_channels(mat)
    grid = _as_map(emg_map)
    _check_indices(grid, x.shape[0])

    filtered = _apply_window(_bandpass(x, _merged_bpf(bpf), fs, pad_s), window)

    n_cols, n_rows = grid.shape
    best = np.full(x.shape[0], np.nan)
    for col in range(n_cols):
        for row in range(n_rows):
            here = grid[col, row]
            if np.isnan(here):
                continue
            channel = int(here)
            signal = filtered[channel]
            scores = []
            for d_col, d_row in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                near_col, near_row = col + d_col, row + d_row
                if not (0 <= near_col < n_cols and 0 <= near_row < n_rows):
                    continue
                there = grid[near_col, near_row]
                if np.isnan(there):
                    continue
                score = _pearson(signal, filtered[int(there)])
                if score is not None:
                    scores.append(score)
            if scores:
                best[channel] = max(scores)
    return best


def robust_z(values):
    """
    ROBUST_Z scores each value against the median and MAD of the rest

      z = robust_z(amp.rms)

      INPUT
        values ... vector of one measure, one entry per channel of ONE grid.
                    NaN entries are ignored when the median and the MAD are
                    formed and score NaN themselves []

      OUTPUT
        z      ... vector of the same length, (value - median) / (1.4826 *
                    MAD). NaN where the input is NaN, and NaN everywhere if
                    the MAD is zero (more than half the channels identical),
                    which is a degenerate grid rather than a grid of
                    outliers []

      WHY MEDIAN AND MAD RATHER THAN MEAN AND SD
        The quantity being scored is "does this channel differ from its own
        grid", and the outliers being looked for are IN the sample. A mean
        and a standard deviation are dragged by exactly those outliers, so a
        single very bad channel raises the bar enough to hide the next one.
        The 1.4826 puts the MAD on the same scale as a standard deviation
        for normally distributed data, so the usual reading of a z-score
        still applies.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.full(x.shape, np.nan)

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad <= 0:
        return np.full(x.shape, np.nan)
    return (x - median) / (1.4826 * mad)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_channels(mat):
    """The channel matrix as float64, rejecting anything not channels-by-samples."""
    x = np.asarray(mat, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"mat must be 2-D (channels x samples), got shape {x.shape}.")
    if x.shape[1] < 2:
        raise ValueError(f"mat must hold at least 2 samples, got {x.shape[1]}.")
    return x


def _merged_bpf(bpf):
    """DEFAULT_BPF with the caller's options on top, unknown keys rejected."""
    unknown = set(bpf or {}) - set(DEFAULT_BPF)
    if unknown:
        raise ValueError(
            f"Unknown bpf option(s) {sorted(unknown)}; expected {sorted(DEFAULT_BPF)}."
        )
    return {**DEFAULT_BPF, **(bpf or {})}


def _bandpass(data, bpf, fs, pad_s):
    """Band-pass with an even reflection at both ends, then trim it off again."""
    pad = pad_samples(data.shape[-1], fs, pad_s)
    padded = reflect_pad(data, pad)
    if bpf['corners'] == 'exact':
        filtered = bandpass_filter_exact_corners(padded, bpf['N'], bpf['fcl'], bpf['fch'], fs)
    elif bpf['corners'] == 'prewarp':
        filtered = bandpass_filter(padded, bpf['N'], bpf['fcl'], bpf['fch'], fs)
    else:
        raise ValueError(f"bpf 'corners' must be 'exact' or 'prewarp', got {bpf['corners']!r}.")
    return trim_pad(filtered, pad)


def _apply_window(data, window):
    """The samples the caller asked for, as a view or a copy of the same rows."""
    if window is None:
        return data
    if isinstance(window, slice):
        selected = data[:, window]
    elif isinstance(window, tuple) and len(window) == 2:
        selected = data[:, int(window[0]):int(window[1])]
    else:
        mask = np.asarray(window)
        if mask.dtype != bool or mask.shape != (data.shape[1],):
            raise ValueError(
                "window must be a slice, a (start, stop) tuple, or a boolean mask "
                f"of length {data.shape[1]}."
            )
        selected = data[:, mask]
    if selected.shape[1] < 2:
        raise ValueError(f"window selects {selected.shape[1]} sample(s); need at least 2.")
    return selected


def _pearson(a, b):
    """Pearson r over the samples both channels have, or None if undefined."""
    both = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(both) < 2:
        return None
    x, y = a[both], b[both]
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator <= 0:
        return None  # at least one of them is constant over the window
    return float(np.dot(x, y) / denominator)
