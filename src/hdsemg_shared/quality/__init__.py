"""
QUALITY: measures that say whether a recording is fit to analyse

  Two layers, and the split between them is deliberate:

    channel_metrics ... one number per CHANNEL - is it dead, is its
                         amplitude or its spectrum unlike the rest of its
                         grid, does it pick up mains, does it clip, does it
                         share anything with the electrodes next to it
    propagation     ... one answer per GRID - along which direction the
                         action potentials travel, how fast, and whether an
                         innervation zone sits underneath

  NOTHING HERE RETURNS A VERDICT. Every function returns a measured number,
  and where the threshold sits is the caller's decision: it depends on the
  muscle, the electrode, the task and the population, not on the
  measurement. That separation is what lets a quality gate store its
  evidence next to the human answer it produced, and re-derive either one
  without the other.

  >>> from hdsemg_shared.quality import channel_amplitude, propagation
  >>> amp = channel_amplitude(emg.T, fs, window = peak_window)
  >>> prop = propagation(emg.T, emg_map, ied_mm = 10.0, fs = fs)
  >>> prop.cv_status
  'iz_split'

  (c) H Penasso. Written for hdsemg-shared by Claude Opus 5, 2026-08-19.
"""

from hdsemg_shared.quality.channel_metrics import (
    DEFAULT_BPF,
    DEFAULT_PAD_S,
    ChannelAmplitude,
    ChannelSpectrum,
    LineNoise,
    channel_amplitude,
    channel_spectrum,
    clipping_fraction,
    flat_channels,
    line_noise_ratio,
    neighbor_correlation,
    robust_z,
)
from hdsemg_shared.quality.propagation import (
    DEFAULT_ANGLES,
    MAX_CV_MS,
    MIN_CV_MS,
    MIN_VALID_PAIRS,
    PropagationResult,
    propagation,
)

__all__ = [
    "DEFAULT_BPF",
    "DEFAULT_PAD_S",
    "ChannelAmplitude",
    "ChannelSpectrum",
    "LineNoise",
    "channel_amplitude",
    "channel_spectrum",
    "clipping_fraction",
    "flat_channels",
    "line_noise_ratio",
    "neighbor_correlation",
    "robust_z",
    "DEFAULT_ANGLES",
    "MAX_CV_MS",
    "MIN_CV_MS",
    "MIN_VALID_PAIRS",
    "PropagationResult",
    "propagation",
]
