"""
GLOBAL_PARAMETERS: amplitude, frequency and complexity measures

  The single-channel measures (ARV, RMS, IEMG, MDF, MNF, entropy) each take
  one 1-D signal. The grid-wide measure is global_amplitude, which takes a
  channel matrix plus an emg_map and reduces the whole grid to one amplitude
  over time.

  >>> from hdsemg_shared.global_parameters import global_amplitude
  >>> out = global_amplitude(emg.T, emg_map, fs, method = 'RMS')
"""

from hdsemg_shared.global_parameters.global_amplitude import (
    DEFAULT_BPF,
    DEFAULT_SMOOTH,
    GlobalAmplitude,
    global_amplitude,
)

__all__ = [
    "DEFAULT_BPF",
    "DEFAULT_SMOOTH",
    "GlobalAmplitude",
    "global_amplitude",
]
