"""
LOWPASS: zero-phase Butterworth lowpass, MATLAB-compatible

  Companion of filters.bandpass.bandpass_filter, same pre-warp, same
  bi-directional application. A SINGLE corner is the case Ton van den
  Bogert's pre-warp is exact for, so the realised -3 dB point lands on fc
  and there is no exact-corner variant to add here (verified: 15.000 Hz
  realised for N = 2 and N = 4).

  PORT of SP_AM_CV_HDsEMG_Analysis.m, local function lowpassfilter
  (line 1567), (c) H Penasso 20.02.2019.
  CREDIT Ton van den Bogert,
  https://biomch-l.isbweb.org/archive/index.php/t-26625.html

  Ported to Python for hdsemg-shared by Claude Opus 5, 2026-08-19.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def lowpass_filter(data, N, fc, fs):
    """
    LOWPASS_FILTER applies a zero-phase Butterworth lowpass to each channel

      env = lowpass_filter(rectified, 2, 15.0, 2048.0)

      INPUT
        data ... signal over time, numeric vector; or a matrix whose rows are
                  the channels and whose columns are the samples over time,
                  then each row is filtered on its own [xV]
        N    ... filter order, number. Must be EVEN and >= 2: it is halved
                  internally and doubled again by the bi-directional
                  (forwards and backwards) filtering []
        fc   ... cutoff frequency, the realised -3 dB point of the whole
                  two-pass filter [Hz]
        fs   ... sampling frequency [Hz]

      OUTPUT
        out  ... filtered signal, same shape as data [xV]

      RAISES
        ValueError ... if N is not an even integer >= 2, if fs or fc are not
                        positive, or if fs is too small for fc

      *INFO* ... MATLAB's own guard uses the UN-halved order and is therefore
                  weaker than the filter needs; butter() then errors instead.
                  Testing the normalised cutoff against 1 rejects exactly the
                  same inputs in one check
      *INFO* ... padding follows MATLAB (padtype 'odd', padlen
                  3*(max(len(a),len(b))-1)), NOT scipy's defaults. For the
                  edge bias that still leaves, see filters.padding
    """
    # Check the arguments
    if N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2 (bi-directional filtering).")
    if fs <= 0 or fc <= 0:
        raise ValueError("fs and fc must be positive.")

    # Pre-warp with the HALVED order, N is doubled again by filtfilt
    half_n = N // 2
    beta = (np.sqrt(2) - 1) ** (1 / (2 * half_n))
    Wn = float((2.0 * fc) / (fs * beta))

    # Guard the sampling frequency instead of clipping to a wrong-but-finite result
    if Wn <= 0:
        raise ValueError(
            f"fc={fc} Hz underflows to a normalised cut-off of {Wn} at "
            f"fs={fs} Hz; the filter is not representable."
        )
    if Wn >= 1.0:
        raise ValueError(
            f"Your sampling frequency is too small for the specified cut-off "
            f"frequency: fs={fs} Hz, fc={fc} Hz and N={N} give a normalised "
            f"cut-off of {Wn:.6g} >= 1. Need fs > {2.0 * fc / beta:.6g} Hz."
        )

    # Design and apply zero-phase
    b, a = butter(half_n, Wn, btype='low')
    padlen = 3 * (max(len(a), len(b)) - 1)
    return filtfilt(b, a, data, axis=-1, padtype='odd', padlen=padlen)
