from typing import List, Union, Dict, Tuple

import numpy as np

from hdsemg_shared.filters.bandpass import bandpass_filter


def to_differential(
    mats: List[np.ndarray],
    sr: Union[int, float],
    f: Dict[str, Union[int, float]],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Exact Python replica of MATLAB `toDifferential`.

    Parameters
    ----------
    mats : list of ndarray
        `k` matrices, each shape (j, T) where j = number of monopolar
        EMG channels and T = samples.
    sr : float
        Sampling rate [Hz].
    f : dict
        Filter options – must contain keys
            'n'   : even Butterworth order (MATLAB style)
            'low' : lower cut-off frequency [Hz]
            'up'  : upper cut-off frequency [Hz]

    Returns
    -------
    dmat        : list of ndarray
        Filtered single-differential signals, each (j-1, T).
    dmatNoF     : list of ndarray
        Unfiltered single-differential signals, each (j-1, T).
    """

    # --------------- 1. sanity checks ----------
    if not mats:
        raise ValueError("`mats` must contain at least one matrix.")
    first_rows = mats[0].shape[0]
    if first_rows < 2:
        raise ValueError("Each matrix must have at least two rows (channels).")

    order = int(f["n"])
    fcl   = float(f["low"])
    fch   = float(f["up"])

    if order < 2 or order % 2:
        raise ValueError("Filter order 'n' must be an even integer ≥ 2.")

    dmat: List[np.ndarray]       = [None] * len(mats)
    dmatNoF: List[np.ndarray]    = [None] * len(mats)

    # --------------- 2. loop -----------------------
    for k, mat in enumerate(mats):
        mat = np.asarray(mat, dtype=np.float64)
        if mat.ndim != 2:
            raise ValueError("Each element of `mats` must be 2-D (channels × samples).")
        rows, T = mat.shape
        if rows != first_rows:
            raise ValueError("All matrices must have the same number of rows as mats[0].")

        diff_raw  = np.empty((rows - 1, T), dtype=np.float64)
        diff_filt = np.empty_like(diff_raw)

        for j in range(rows - 1):
            delta = mat[j + 1, :] - mat[j, :]
            diff_raw[j, :]  = delta
            diff_filt[j, :] = bandpass_filter(delta, order, fcl, fch, sr)

        dmat[k]       = diff_filt
        dmatNoF[k]    = diff_raw

    return dmat, dmatNoF
