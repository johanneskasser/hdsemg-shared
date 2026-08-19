"""
GLOBAL_AMPLITUDE: the one definition of the global HDsEMG amplitude

  Everything that needs a global amplitude comes through here, so there is
  no second definition that can drift away from this one.

  THE DEFINITION, IN FIVE STEPS
    1  band-pass each channel        15-450 Hz, exact corners, zero-lag
    2  square each channel           x_i(t)^2        (ARV: |x_i(t)|)
    3  smooth each channel in time   15 Hz equivalent, zero-lag
    4  mean across the channels      -> mean square over the grid area
    5  square-root, LAST             -> global amplitude A(t)      [xV]

  WHY THE ROOT COMES LAST
    Merletti and Cerone define the RMS of a region of area a*b over an
    interval T as the root of the mean of f^2 over BOTH space and time,
    taken at the very end, which makes mean_space(RMS^2) == mean_time(RMS^2).
    That identity holds only for the SQUARED quantity, so rooting per channel
    first breaks it and makes the result depend on how many channels
    survived, by E[chi_n]/sqrt(n):

        nCh      4      8     12     24     48
        factor  .940   .969   .979   .990   .995

    That matters directly: an MVC trial and a tracking trial rarely keep the
    same channels and their ratio is the reported %MVC. Smoothing in time
    BEFORE reducing across the channels removes the dependence, residual
    spread 0.4 % from 4 to 48 channels.

  DIFFERENCE TO THE MATLAB ORIGINAL
    globEMGAmpEnv reduces across the channels at every single sample and
    smooths afterwards, A(t) = lowpass(sqrt(mean_ch x_i(t)^2)), which is the
    ordering corrected above. Every other MATLAB setting is still reachable
    through the arguments, see the global_amplitude *INFO* block.

  REFERENCES
    Merletti R, Cerone GL. Techniques for information extraction from the
      surface EMG signal, eq. 5.1 and 5.2. In: Merletti & Farina (2016/2018).
    Del Vecchio A et al. J Appl Physiol (2025),
      doi:10.1152/japplphysiol.00810.2024

  PORT of SP_AM_CV_HDsEMG_Analysis.m, local function globEMGAmpEnv
  (line 1485), (c) H Penasso Feb. 12, 2020.
  Ported and extended for hdsemg-shared by Claude Opus 5, 2026-08-19.
"""

from typing import NamedTuple

import numpy as np

from hdsemg_shared.filters.bandpass import bandpass_filter, bandpass_filter_exact_corners
from hdsemg_shared.filters.lowpass import lowpass_filter
from hdsemg_shared.filters.padding import pad_samples, reflect_pad, trim_pad
from hdsemg_shared.filters.smoothing import moving_average
from hdsemg_shared.preprocessing.grid_map import map_to_columns

#: Band-pass options. 'corners' is 'exact' or 'prewarp', see the WHY block of
#: filters.bandpass.bandpass_filter_exact_corners.
DEFAULT_BPF = {'N': 2, 'fcl': 15.0, 'fch': 450.0, 'corners': 'exact'}

#: Smoothing options. 'mode' is 'moving' (a zero-phase boxcar) or 'lowpass'
#: (a Butterworth). Given the same fc the two land on the same bandwidth.
DEFAULT_SMOOTH = {'mode': 'moving', 'fc': 15.0, 'window_s': None,
                  'kernel': 'bidirectional', 'N': 2}

_DERIVATIONS = {'MP': 0, 'SD': 1, 'DD': 2}

#: Which grid axis SD/DD difference along. 'cols' walks DOWN a column, i.e.
#: between vertically adjacent electrodes; 'rows' walks ACROSS a row.
_DIFF_AXIS = {'cols': 1, 'rows': 0}


class GlobalAmplitude(NamedTuple):
    """
    amplitude   ... global amplitude over time [xV]
    per_channel ... each channel's own amplitude envelope, rows are channels
                     in the order of positions [xV]
    positions   ... (column, row) of each kept channel in the derived grid []
    grid_shape  ... (nCols, nRows) of the derived grid []
    n_channels  ... how many channels actually contributed []
    """

    amplitude: np.ndarray
    per_channel: np.ndarray
    positions: np.ndarray
    grid_shape: tuple
    n_channels: int


def global_amplitude(emg_channels, emg_map, fs, method = 'RMS', derivation = 'MP',
                     diff_direction = 'cols', bpf = None, smooth = None, pad_s = 0.25):
    """
    GLOBAL_AMPLITUDE calculates the global HDsEMG amplitude of a grid over time

      out = global_amplitude(emg.T, emg_map, fs)
      out = global_amplitude(emg.T, emg_map, fs, method = 'ARV', derivation = 'SD')
      out = global_amplitude(emg.T, emg_map, fs, derivation = 'SD', diff_direction = 'rows')

      INPUT
        emg_channels ... matrix, each row contains one channel's raw signal
                          over time (rows = channels, columns = samples).
                          Filtering is applied here, so pass it unfiltered.
                          May hold more channels than the map refers to [xV]
        emg_map      ... list of lists or 2-D array following openhdemg's
                          custom_sorting_order: outer index = grid column,
                          inner index = row, base-0 channel numbers, NaN
                          where no electrode sits or a channel is excluded []
        fs           ... double or int, sampling frequency [Hz]

      OPTIONAL INPUT
        method       ... char, 'RMS' (default) or 'ARV' []
        derivation   ... char, 'MP' (default, monopolar), 'SD' (single
                          differential) or 'DD' (double differential). The
                          difference is taken on the raw signal, before
                          filtering []
        diff_direction ... char, which grid axis SD and DD difference along:
                          'cols' (default) walks DOWN a map column, between
                          vertically adjacent electrodes; 'rows' walks ACROSS
                          a row, between horizontally adjacent ones. Ignored
                          for derivation 'MP' []
        bpf          ... dict with bandpassfilter options, keys:
                          x 'N'       - filter order, even, default 2 []
                          x 'fcl'     - lower cutoff, default 15.0 [Hz]
                          x 'fch'     - higher cutoff, default 450.0 [Hz]
                          x 'corners' - 'exact' (default) or 'prewarp' for
                                         MATLAB-identical filtering []
        smooth       ... dict with smoothing options, keys:
                          x 'mode'     - 'moving' (default) or 'lowpass' []
                          x 'fc'       - cutoff, default 15.0 [Hz]
                          x 'window_s' - window width instead of fc, 'moving'
                                          only, default None [s]
                          x 'kernel'   - 'bidirectional' (default) or
                                          'rectangular', 'moving' only []
                          x 'N'        - filter order, 'lowpass' only,
                                          default 2 []
        pad_s        ... length of the reflection pad at each end, 0 disables
                          it. Default 0.25 [s]

      OUTPUT
        out ... GlobalAmplitude with
                 x amplitude   - global amplitude over time [xV]
                 x per_channel - each channel's own amplitude envelope [xV]
                 x positions   - (column, row) of each kept channel []
                 x grid_shape  - (nCols, nRows) of the derived grid []
                 x n_channels  - how many channels contributed []

      RAISES
        ValueError ... on an unknown method, derivation, or option key, on a
                        map too short for the derivation, or when no channel
                        survives

      *INFO* ... NaN channels are ignored. A channel is dropped when the map
                  puts NaN at its position or when its signal carries any NaN
      *INFO* ... point the differencing along the muscle fibres. 'cols' is
                  right when the map's columns run along the fibres, which is
                  how MATLAB's mat2grid and openhdemg's sort_rawemg lay a grid
                  out; use 'rows' for a grid rotated 90 deg against that
      *INFO* ... EMGFile.data is samples-by-channels, so pass its transpose.
                  Applying a selection mask is the CALLER's job: intersect it
                  with the grid's own channels first, then put NaN in the map
                  at every excluded position
      *INFO* ... per_channel is a per-channel amplitude in [xV], so the root
                  IS taken per channel for it. It is a display quantity, for
                  a density map; it is deliberately NOT how amplitude is
                  computed, which reduces across channels before rooting
      *INFO* ... the MATLAB settings are bpf = {'N': 2, 'fcl': 30.0,
                  'fch': 450.0, 'corners': 'prewarp'}, smooth = {'mode':
                  'lowpass', 'fc': 6.0, 'N': 2} and pad_s = 0. The root
                  ordering is then the only remaining difference
      *INFO* ... with a single channel MATLAB collapses all three of its
                  outputs to lowpass(|x|). Here ARV still does that, while
                  RMS stays the moving RMS of that channel - the consistent
                  extension of the definition rather than a special case
    """
    method = _check_choice(method, ('RMS', 'ARV'), 'method')
    derivation = _check_choice(derivation, tuple(_DERIVATIONS), 'derivation')
    diff_direction = _check_choice(diff_direction, tuple(_DIFF_AXIS), 'diff_direction')
    bpf = _merge(DEFAULT_BPF, bpf, 'bpf')
    smooth = _merge(DEFAULT_SMOOTH, smooth, 'smooth')
    if pad_s < 0:
        raise ValueError(f"pad_s must be >= 0, got {pad_s}.")

    # Select and order the mapped channels, then difference along the chosen axis
    grid = np.stack(map_to_columns(emg_channels, emg_map))  # (nCols, nRows, nSamples)
    grid = _differentiate(grid, derivation, diff_direction)
    grid_shape = (grid.shape[0], grid.shape[1])

    # Flatten to a channel matrix, column by column, the way MATLAB's grid2mat
    # does, and keep only the channels carrying signal
    x = grid.reshape(-1, grid.shape[-1])
    positions = np.array([(j, i) for j in range(grid_shape[0]) for i in range(grid_shape[1])])
    keep = ~np.any(np.isnan(x), axis=1)
    x, positions = x[keep], positions[keep]
    if x.shape[0] == 0:
        raise ValueError(f"No channel survived the map and the '{derivation}' derivation.")

    # Band-pass each channel, guarding both ends against the filter transient
    x = _padded(x, fs, pad_s, lambda d: _bandpass(d, bpf, fs))

    # Square (RMS) or rectify (ARV), then smooth each channel in time
    p = x ** 2 if method == 'RMS' else np.abs(x)
    p = _padded(p, fs, pad_s, lambda d: _smooth(d, smooth, fs))
    p = np.clip(p, 0.0, None)  # a butterworth can undershoot below zero

    # Mean across the channels, root LAST
    mean_p = np.mean(p, axis=0)
    amplitude = np.sqrt(mean_p) if method == 'RMS' else mean_p
    per_channel = np.sqrt(p) if method == 'RMS' else p

    return GlobalAmplitude(amplitude, per_channel, positions, grid_shape, x.shape[0])


def _differentiate(grid, derivation, diff_direction):
    """The grid, differenced along the chosen axis as many times as asked."""
    order = _DERIVATIONS[derivation]
    if order == 0:
        return grid

    axis = _DIFF_AXIS[diff_direction]
    along = 'rows per column' if diff_direction == 'cols' else 'columns per row'
    if grid.shape[axis] <= order:
        raise ValueError(
            f"'{derivation}' along '{diff_direction}' needs more than {order} "
            f"{along}, got {grid.shape[axis]}."
        )
    return np.diff(grid, n=order, axis=axis)


def _bandpass(data, bpf, fs):
    """The band-pass the 'corners' option asks for."""
    if bpf['corners'] == 'exact':
        return bandpass_filter_exact_corners(data, bpf['N'], bpf['fcl'], bpf['fch'], fs)
    if bpf['corners'] == 'prewarp':
        return bandpass_filter(data, bpf['N'], bpf['fcl'], bpf['fch'], fs)
    raise ValueError(f"bpf 'corners' must be 'exact' or 'prewarp', got {bpf['corners']!r}.")


def _smooth(data, smooth, fs):
    """The time smoother the 'mode' option asks for."""
    if smooth['mode'] == 'moving':
        return moving_average(data, fs, window_s=smooth['window_s'],
                              fc=None if smooth['window_s'] is not None else smooth['fc'],
                              kernel=smooth['kernel'])
    if smooth['mode'] == 'lowpass':
        if smooth['window_s'] is not None:
            raise ValueError("smooth 'window_s' applies to mode 'moving' only.")
        return lowpass_filter(data, smooth['N'], smooth['fc'], fs)
    raise ValueError(f"smooth 'mode' must be 'moving' or 'lowpass', got {smooth['mode']!r}.")


def _padded(data, fs, pad_s, apply):
    """Filter with an even reflection at both ends, then trim it off again."""
    pad = pad_samples(data.shape[-1], fs, pad_s)
    return trim_pad(apply(reflect_pad(data, pad)), pad)


def _merge(defaults, given, name):
    """Defaults with the caller's options on top, unknown keys rejected."""
    unknown = set(given or {}) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown {name} option(s) {sorted(unknown)}; expected {sorted(defaults)}."
        )
    return {**defaults, **(given or {})}


def _check_choice(value, allowed, name):
    """Case-insensitive choice, canonicalised to the spelling in `allowed`."""
    canonical = {choice.upper(): choice for choice in allowed}
    if not isinstance(value, str) or value.upper() not in canonical:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}.")
    return canonical[value.upper()]
