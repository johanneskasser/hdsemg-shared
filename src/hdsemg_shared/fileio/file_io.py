from pathlib import Path

import numpy as np

from .matlab_file_io import load_mat_file
from .otb_4_file_io import load_otb4_file
from .otb_plus_file_io import load_otb_file


def load_file(filepath):
    """
    Loads a file based on its file extension and extracts relevant data.

    Supported file types:
    - `.mat`: MATLAB files
    - `.otb+`: OTB+ and related files
    - `.otb4`: OTB4 and related files

    Args:
        filepath (str): The path to the file to be loaded.

    Returns:
        tuple: A tuple containing:
            - data: The loaded data. (nSamples x nChannels)
            - time: The time information associated with the data. (nSamples,)
            - description: A description of the data.
            - sampling_frequency: The sampling frequency of the data.
            - file_name: The name of the file.
            - file_size: The size of the file.

    Raises:
        ValueError: If the file type is unsupported.
    """
    file_suffix = Path(filepath).suffix
    if file_suffix == ".mat":
        data, time, description, sampling_frequency, file_name, file_size = load_mat_file(filepath)
    elif file_suffix in [".otb+", ".otb"]:
        data, time, description, sampling_frequency, file_name, file_size = load_otb_file(filepath)
    elif file_suffix == ".otb4":
        data, time, description, sampling_frequency, file_name, file_size = load_otb4_file(filepath)
    else:
        raise ValueError(f"Unsupported file type: {file_suffix}")

    # Handle case if data is int16 since we will run into issues with further processing
    if data.dtype == 'int16':
        data = data.astype(np.float32)

    data, time = _sanitize_data(data, time)

    return data, time, description, sampling_frequency, file_name, file_size


def _sanitize_data(data, time):
    """
        Ensures that the data and time arrays have the correct shape.

        - Data is converted to at least 2D and transposed if necessary so that the number of rows is greater than or equal to the number of columns.
        - Time is squeezed to 1D and reshaped if necessary to match the data.
        - Raises a ValueError if the shapes are not compatible.

        Args:
            data (np.ndarray): The data array.
            time (np.ndarray): The time array.

        Returns:
            tuple: (data, time) in compatible shapes.
        """
    data = np.atleast_2d(data)
    if data.shape[0] < data.shape[1]:
        data = data.T

    time = np.squeeze(time)
    if time.ndim == 2:
        time = time[:, 0] if time.shape[1] == 1 else time[0, :]
    if time.ndim == 1 and time.shape[0] != data.shape[0]:
        # Try swapping
        if time.shape[0] == data.shape[1]:
            time = time.T
        else:
            raise ValueError(f"Incompatible form time {time.shape} for data {data.shape}. Please check the input data.")

    return data, time
