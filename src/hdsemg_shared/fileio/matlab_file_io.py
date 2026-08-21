# hdsemg/matlab_file_io.py
import os
import json
import logging
from pathlib import Path
import scipy.io as sio

from .units import normalize_unit

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class MatFileIO:
    @staticmethod
    def load(file_path: str):
        """
        Returns (data, time, description, sampling_frequency, file_name,
        file_size, unit).

        ``unit`` comes from the optional ``Unit`` variable (written by
        :meth:`save`); plain MATLAB exports carry no unit, so it is ``None``.
        """
        mat_data = sio.loadmat(file_path)
        data = mat_data['Data']
        time = mat_data['Time'].flatten()
        description = mat_data.get('Description', None)
        sampling_frequency = (
            mat_data.get('SamplingFrequency', [[1]])[0][0]
            if 'SamplingFrequency' in mat_data else 1
        )
        file_name = Path(file_path).name
        file_size = os.path.getsize(file_path)
        unit = normalize_unit(_scalar_text(mat_data.get('Unit')))
        return data, time, description, sampling_frequency, file_name, file_size, unit

    @staticmethod
    def save(save_file_path, data, time, description, sampling_frequency, unit=None):
        """
        Write a .mat file. ``unit`` is stored only when known, so that a
        round-trip preserves it without inventing one.
        """
        path_obj = Path(save_file_path)
        if path_obj.suffix.lower() != ".mat":
            path_obj = path_obj.with_suffix(".mat")
        final_path = str(path_obj)

        mat_dict = {
            "Data": data,
            "Time": time,
            "Description": description,
            "SamplingFrequency": sampling_frequency
        }
        if unit is not None:
            mat_dict["Unit"] = unit
        sio.savemat(final_path, mat_dict)
        logger.info(f"MAT file saved successfully: {final_path}")
        return final_path



def _scalar_text(value):
    """Unwrap the nested arrays scipy returns for a scalar string variable."""
    if value is None:
        return None
    while hasattr(value, "size"):
        if value.size != 1:
            return None
        value = value.item() if value.ndim == 0 else value.flat[0]
    return value
