import numpy as np

from hdsemg_shared.filters.bandpass import bandpass_filter


def to_differential(
        mats: list[np.ndarray],
        sr: float,
        f: dict[str, float]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Compute (j-1) differential channels for each matrix in `mats` and
    apply a Butterworth bandpass filter to each differential.

    Parameters
    ----------
    mats : list of 2D np.ndarray
        Each element is a matrix of shape (j, T), where j is the number of
        channels (signals) and T is the number of samples.
    sr : float
        Sampling rate in Hz.
    f : dict
        Filter parameters:
            - 'n'   : filter order
            - 'low' : lower cutoff frequency in Hz
            - 'up'  : upper cutoff frequency in Hz

    Returns
    -------
    dmat : list of 2D np.ndarray
        Filtered differentials, each shape (j-1, T).
    dmat_no_filter : list of 2D np.ndarray
        Unfiltered differentials, each shape (j-1, T).
    """
    dmat = []
    dmat_no_filter = []

    order = int(f['n'])
    lowcut = float(f['low'])
    highcut = float(f['up'])

    for mat in mats:
        # Ensure input is a 2D array
        mat = np.asarray(mat)
        if mat.ndim != 2:
            raise ValueError("Each element of `mats` must be a 2D array.")

        # Compute unfiltered differential (row_j+1 - row_j)
        diff_no_filt = mat[1:, :] - mat[:-1, :]
        dmat_no_filter.append(diff_no_filt)

        # Apply bandpass filter along time axis for each differential row
        diff_filt = bandpass_filter(diff_no_filt, order, lowcut, highcut, sr)
        dmat.append(diff_filt)

    return dmat, dmat_no_filter
