import numpy as np
from scipy.signal import butter, sosfiltfilt


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

    Returns
    -------
    fdata : ndarray
        Zero-phase band-pass filtered signal (same length as `data`).
    """

    # ------------- 1. argument checks  ----------
    if N < 2 or N % 2:
        raise ValueError("N must be an even integer ≥ 2 (bi-directional filtering).")
    if fs <= 0 or fcl <= 0 or fch <= 0 or fcl >= fch:
        raise ValueError("Cut-off frequencies must satisfy 0 < fcl < fch < fs/2.")
    # ------------------------------------------------------------------------

    # ------------- 2. translate MATLAB design rule --------------------------
    # In Ton’s routine Wn is *pre-warped* so that after filtfilt()
    # the –3 dB point sits exactly at the user’s fcl/fch.
    halfN = N // 2  # order actually given to butter()
    beta = (np.sqrt(2) - 1) ** (1 / (2 * N))  # pre-warping constant
    Wn = (2.0 * np.asarray([fcl, fch])) / (fs * beta)  # normalised (0–1)
    Wn = np.clip(Wn, 1e-6, 0.999)  # avoid numerical issues with very low frequencies
    # ------------------------------------------------------------------------

    # ------------- 3. design filter & apply zero-phase ----------------------
    sos = butter(halfN, Wn, btype='bandpass', output='sos')  # identical poles/zeros
    fdata = sosfiltfilt(sos, data, axis=-1)  # zero-phase (like filtfilt)
    return fdata
