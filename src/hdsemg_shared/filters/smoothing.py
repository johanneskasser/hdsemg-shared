"""
SMOOTHING: zero-phase moving average, as an alternative to a lowpass

  A boxcar run forwards and backwards is the second way to smooth an
  envelope. Given the same fc it attenuates a sine at fc exactly as
  filters.lowpass.lowpass_filter does, so the two differ only in kernel
  shape and the choice never changes the bandwidth.

  TWO KERNELS
    "bidirectional" ... two boxcar passes, i.e. a triangular kernel. The
                         default, the smoother of the two
    "rectangular"   ... one plain boxcar. With a window as long as the
                         signal every sample returns the plain unweighted
                         mean over the whole signal, which "bidirectional"
                         does NOT - it weighs the middle of its window more
                         heavily. Use "rectangular" wherever exact agreement
                         with an epoch mean is the point

  ADDITION of the Python implementation, no MATLAB equivalent.

  (c) H Penasso. Written for hdsemg-shared by Claude Opus 5, 2026-08-19,
  from the MATLAB conventions of Global-HDsEMG-Analysis.
"""

import numpy as np
from scipy.ndimage import convolve1d
from scipy.optimize import brentq

_PASSES = {'bidirectional': 2, 'rectangular': 1}


def window_length_for_cutoff(fc, fs, kernel = 'bidirectional'):
    """
    WINDOW_LENGTH_FOR_CUTOFF returns the boxcar length whose response is
    -3 dB at fc

      w = window_length_for_cutoff(15.0, 2048.0)

      INPUT
        fc     ... wanted -3 dB cutoff frequency [Hz]
        fs     ... sampling frequency [Hz]
        kernel ... char, 'bidirectional' (default) or 'rectangular' []

      OUTPUT
        w      ... length of ONE boxcar pass, samples []

      RAISES
        ValueError ... if fc is not below the Nyquist frequency, or if no
                        window is long enough to reach fc

      *INFO* ... -3 dB means the response of the WHOLE smoother, both passes
                  included, has fallen to 1/sqrt(2). Give this function and
                  lowpass_filter the same fc and both attenuate a sine at fc
                  equally
      *INFO* ... rule of thumb w ~ 0.319 * fs / fc, so at fs = 2048 Hz and
                  fc = 15 Hz that is 44 samples, a 31.9 ms equivalent window
    """
    if fs <= 0 or fc <= 0:
        raise ValueError("fs and fc must be positive.")
    if fc >= fs / 2:
        raise ValueError(f"fc={fc} Hz must be below the Nyquist frequency fs/2={fs / 2} Hz.")

    passes = _passes(kernel)
    target = 1.0 / np.sqrt(2.0)

    # The response falls monotonically in w until the kernel's first null,
    # which sits at w = fs/fc; bracket below it to stay on that branch
    w_hi = min(fs / fc, 1e6)
    if _response(fc, w_hi, fs, passes) > target:
        raise ValueError(f"No boxcar window reaches -3 dB at fc={fc} Hz for fs={fs} Hz.")

    w = brentq(lambda ww: _response(fc, ww, fs, passes) - target, 1.0 + 1e-9, w_hi, xtol=1e-10)
    return max(1, int(round(w)))


def moving_average(data, fs, window_s = None, fc = None, kernel = 'bidirectional'):
    """
    MOVING_AVERAGE smooths each channel with a zero-phase moving window

      y = moving_average(squared, fs, fc = 15.0)
      y = moving_average(squared, fs, window_s = 3.0, kernel = 'rectangular')

      INPUT
        data     ... signal over time, numeric vector; or a matrix whose rows
                      are the channels and whose columns are the samples over
                      time, then each row is smoothed on its own [xV]
        fs       ... sampling frequency [Hz]
        window_s ... equivalent rectangular width of the window, give this OR
                      fc, not both [s]
        fc       ... wanted -3 dB cutoff, give this OR window_s [Hz]
        kernel   ... char, 'bidirectional' (default) or 'rectangular', see
                      the module docstring, TWO KERNELS []

      OUTPUT
        y        ... smoothed signal, same shape as data. Exactly zero-phase,
                      a symmetric input comes back symmetric [xV]

      RAISES
        ValueError ... if neither or both of window_s and fc are given, if
                        either is not positive, or if kernel is unknown

      *INFO* ... the ends are divided by the kernel weight that actually
                  overlapped them, so no taper towards zero is introduced
    """
    x = np.asarray(data, dtype=np.float64)
    h = _kernel(kernel, _resolve_window(fs, window_s, fc, kernel))
    if h.size == 1:
        return x.copy()

    # Convolve the signal and a matching all-ones signal with the same kernel,
    # then divide: at the ends only part of the kernel overlaps and the
    # denominator carries exactly that part
    num = convolve1d(x, h, axis=-1, mode='constant', cval=0.0)
    den = convolve1d(np.ones(x.shape[-1], dtype=np.float64), h, mode='constant', cval=0.0)
    return num / den


def _passes(kernel):
    """Number of boxcar passes a kernel name stands for."""
    if kernel not in _PASSES:
        raise ValueError(f"kernel must be one of {sorted(_PASSES)}, got {kernel!r}.")
    return _PASSES[kernel]


def _response(f, w, fs, passes):
    """Magnitude response of a boxcar of length w applied `passes` times, at f."""
    # One pass has the Dirichlet response sin(pi f w/fs) / (w sin(pi f/fs));
    # running it forwards and backwards multiplies the response by itself
    d = np.sin(np.pi * f * w / fs) / (w * np.sin(np.pi * f / fs))
    return float(d ** passes)


def _kernel(kernel, w):
    """The convolution kernel of either smoother, normalised to sum 1."""
    box = np.ones(int(w), dtype=np.float64) / float(w)
    if _passes(kernel) == 2:
        return np.convolve(box, box)

    # A plain boxcar is forced odd so it has a true centre sample
    w = int(w) + 1 if int(w) % 2 == 0 else int(w)
    return np.ones(w, dtype=np.float64) / float(w)


def _resolve_window(fs, window_s, fc, kernel):
    """One of window_s / fc, turned into a single-pass boxcar length."""
    if (window_s is None) == (fc is None):
        raise ValueError("Give exactly one of window_s or fc.")
    _passes(kernel)

    if fc is not None:
        return window_length_for_cutoff(fc, fs, kernel)
    if window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s}.")
    if kernel == 'rectangular':
        # A plain boxcar's equivalent rectangular width IS its length
        return max(1, int(round(window_s * fs)))

    # Two passes of length w have an equivalent rectangular width of 1.5*w,
    # so undo that factor to make the REQUESTED width the REALISED width
    return max(1, int(round(window_s * fs / 1.5)))
