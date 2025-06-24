# hdsemg/matlab_file_io.py
import os
import json
import logging
from pathlib import Path
import scipy.io as sio

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class MatFileIO:
    @staticmethod
    def load(file_path: str):
        """
        Returns exactly the same tuple as your old load_mat_file.
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
        return data, time, description, sampling_frequency, file_name, file_size

    @staticmethod
    def save(save_file_path, data, time, description, sampling_frequency):
        """
        Exactly your save logic.
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
        sio.savemat(final_path, mat_dict)
        logger.info(f"MAT file saved successfully: {final_path}")
        return final_path

