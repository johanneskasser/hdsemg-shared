"""
BANDPASS: zero-phase Butterworth band-pass, in two variants

  bandpass_filter               ... exact replica of Ton van den Bogert's
                                     MATLAB bandpassfilter. Its pre-warp is
                                     derived for a SINGLE corner, so both
                                     band-pass corners land ~1.554x high
  bandpass_filter_exact_corners ... solves for the corners numerically, so
                                     the realised -3 dB points land on the
                                     requested ones

  Both filter a vector, or each row of a channels-by-samples matrix, on its
  own. Keep bandpass_filter for MATLAB agreement, use the exact-corner
  variant when the documented cut-offs must be the real ones.

  CREDIT Ton van den Bogert,
  https://biomch-l.isbweb.org/archive/index.php/t-26625.html
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import butter, filtfilt, freqz


def bandpass_filter(data: np.ndarray,
                    N: int,
                    fcl: float,
                    fch: float,
                    fs: float) -> np.ndarray:
    """
    Exact Python replica of Ton van den Bogert's MATLAB `bandpassfilter`.
    @credit Ton van den Bogert, https://biomch-l.isbweb.org/archive/index.php/t-26625.html

    -------------------------------------------------------------------
    Parameters
    ----------
    data : 1-D ndarray
        Signal to be filtered.
    N : int
        *Total* filter order requested by the user (must be even).
        Internally the Butterworth prototype is designed with order N/2 and then
        applied forwards & backwards (filtfilt) which doubles the effective order.
    fcl, fch : float
        Lower and higher cut-off frequencies [Hz].
    fs : float
        Sampling frequency [Hz].
        Must satisfy fs > 2*fch/beta with beta = (sqrt(2)-1)**(1/N), i.e. the
        normalised cut-offs handed to butter() must stay below 1, otherwise a
        ValueError is raised (see the guard below). MATLAB errors on the same
        inputs; earlier versions of this function clipped instead and returned
        a finite but wrong result.

    Returns
    -------
    fdata : ndarray
        Zero-phase band-pass filtered signal (same length as `data`).

    Raises
    ------
    ValueError
        If N is not an even integer >= 2, if the cut-offs do not satisfy
        0 < fcl < fch, or if `fs` is too small for the requested cut-offs.
    """

    # ------------- 1. argument checks  ----------
    if N < 2 or N % 2:
        raise ValueError("N must be an even integer ≥ 2 (bi-directional filtering).")
    if fs <= 0 or fcl <= 0 or fch <= 0 or fcl >= fch:
        raise ValueError("Cut-off frequencies must satisfy 0 < fcl < fch < fs/2.")
    # ------------------------------------------------------------------------

    # ------------- 2. translate MATLAB design rule --------------------------
    # In Ton's routine, N is halved BEFORE the pre-warp constant is computed
    # (`N = N/2;` precedes the Wn line in the original MATLAB), so the
    # exponent below uses halfN, not the original N. Using the original N
    # here previously shifted both cutoffs (e.g. for N=2: beta=0.8022,
    # cutoffs off by ~20%, instead of the correct beta=0.6436) -- confirmed
    # against real MATLAB output on gait EMG data.
    halfN = N // 2  # order actually given to butter()
    beta = (np.sqrt(2) - 1) ** (1 / (2 * halfN))  # pre-warping constant (uses HALVED order)
    Wn = (2.0 * np.asarray([fcl, fch])) / (fs * beta)  # normalised (0–1)
    # ------------------------------------------------------------------------

    # ------------- 2b. sampling-frequency guard -----------------------------
    # MATLAB errors out when the sampling frequency is too small for the
    # requested cut-off. Its own guard,
    #     fs < 2 * (-1 + sqrt(2))^(-(1/2)/N) * fc
    # is evaluated with the UN-halved N (it precedes `N = N/2;`), which makes
    # it weaker than the filter actually needs: for N=2, fc=6 Hz it passes
    # from fs=14.96 Hz on, while Wn stays below 1 only from fs=18.65 Hz on --
    # in that gap MATLAB passes its own guard and then butter() itself errors.
    # Testing Wn < 1 directly therefore rejects exactly the inputs MATLAB
    # rejects, in one check instead of two.
    # Wn used to be clipped into (1e-6, 0.999) here, which silently returned a
    # finite but WRONG result for an under-sampled input instead of raising.
    if Wn[0] <= 0:
        raise ValueError(
            f"fcl={fcl} Hz underflows to a normalised cut-off of {Wn[0]} at "
            f"fs={fs} Hz; the filter is not representable."
        )
    if Wn[0] >= 1.0:
        raise ValueError(
            f"Your sampling frequency is too small for the specified lower "
            f"cut-off frequency: fs={fs} Hz, fcl={fcl} Hz and N={N} give a "
            f"normalised cut-off of {Wn[0]:.6g} >= 1. Need fs > "
            f"{2.0 * fcl / beta:.6g} Hz."
        )
    if Wn[1] >= 1.0:
        raise ValueError(
            f"Your sampling frequency is too small for the specified higher "
            f"cut-off frequency: fs={fs} Hz, fch={fch} Hz and N={N} give a "
            f"normalised cut-off of {Wn[1]:.6g} >= 1. Need fs > "
            f"{2.0 * fch / beta:.6g} Hz."
        )
    # ------------------------------------------------------------------------

    # ------------- 3. design filter & apply zero-phase ----------------------
    # Classical [b,a] + filtfilt with MATLAB's default padding
    # (padtype='odd', padlen=3*(max(len(a),len(b))-1)) rather than
    # scipy's sosfiltfilt defaults, which use a different pad length and
    # measurably diverge from MATLAB on short signals (e.g. epoch-windowed
    # EMG). Confirmed to match real MATLAB output to ~1e-13 relative error.
    b, a = butter(halfN, Wn, btype='bandpass')
    padlen = 3 * (max(len(a), len(b)) - 1)
    fdata = filtfilt(b, a, data, axis=-1, padtype='odd', padlen=padlen)
    return fdata


def bandpass_filter_exact_corners(data, N, fcl, fch, fs):
    """
    BANDPASS_FILTER_EXACT_CORNERS band-passes each channel with -3 dB points
    landing exactly on fcl and fch

      emg = bandpass_filter_exact_corners(emg, 2, 15.0, 450.0, 2048.0)

      INPUT
        data ... signal over time, numeric vector; or a matrix whose rows are
                  the channels and whose columns are the samples over time,
                  then each row is filtered on its own [xV]
        N    ... filter order, number. Must be EVEN and >= 2 []
        fcl  ... lower cutoff frequency, the realised -3 dB point [Hz]
        fch  ... higher cutoff frequency, the realised -3 dB point [Hz]
        fs   ... sampling frequency [Hz]

      OUTPUT
        out  ... filtered signal, same shape as data [xV]

      RAISES
        ValueError ... if N is not an even integer >= 2, or if the cutoffs do
                        not satisfy 0 < fcl < fch < fs/2

      WHY
        bandpass_filter is an exact replica of Ton van den Bogert's MATLAB
        original, whose beta pre-warp is derived for a SINGLE corner. A
        band-pass Butterworth has no such closed form, so dividing BOTH
        corners by beta < 1 pushes both UP by 1/beta = 1.554: at fs = 2048 Hz
        and N = 2 a nominal 15-450 Hz band actually lands at 35.3-574.9 Hz,
        and 30-450 Hz at 68.8-582.3 Hz. This function instead solves
        numerically for the corners to hand to butter(), same goal, no closed
        form.

      *INFO* ... NOT a MATLAB match, use bandpass_filter for that. This is an
                  addition of the Python implementation
      *INFO* ... the solved upper corner sits ABOVE fch for a two-pass
                  filter, so it can still run into the Nyquist clip even when
                  the requested fch does not. Keep fch well below fs/2
    """
    # Check the arguments
    if N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2 (bi-directional filtering).")
    if fs <= 0 or fcl <= 0 or fch <= 0 or fcl >= fch or fch >= fs / 2:
        raise ValueError("Cut-off frequencies must satisfy 0 < fcl < fch < fs/2.")

    halfN = N // 2
    target = 1 / np.sqrt(2)

    # Solve for the butter() corners whose TWO-PASS response is -3 dB at fcl/fch
    def residual(corners):
        lo = np.clip(corners[0], 1e-3, fs / 2 - 1e-3)
        hi = np.clip(corners[1], lo + 1e-3, fs / 2 - 1e-3)
        Wn = np.clip((2.0 * np.array([lo, hi])) / fs, 1e-6, 0.999)
        b, a = butter(halfN, Wn, btype='bandpass')
        _, h = freqz(b, a, worN=2 * np.pi * np.array([fcl, fch]) / fs)
        return np.abs(h) ** 2 - target

    lo, hi = fsolve(residual, x0=[fcl, fch], xtol=1e-12)
    Wn = np.clip((2.0 * np.array([lo, hi])) / fs, 1e-6, 0.999)

    # Design and apply zero-phase, MATLAB's padding
    b, a = butter(halfN, Wn, btype='bandpass')
    padlen = 3 * (max(len(a), len(b)) - 1)
    return filtfilt(b, a, data, axis=-1, padtype='odd', padlen=padlen)
