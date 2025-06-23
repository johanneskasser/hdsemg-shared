import numpy as np
from hdsemg_shared.preprocessing.differential import to_differential


def test_to_differential_output_shape_and_values():
    sr = 1000  # Hz
    f = {"n": 4, "low": 20, "up": 400}

    # Create synthetic monopolar matrices
    T = 1000
    mono1 = np.vstack([
        np.sin(2 * np.pi * 10 * np.linspace(0, 1, T)),     # below band
        np.sin(2 * np.pi * 100 * np.linspace(0, 1, T)),    # within band
        np.sin(2 * np.pi * 500 * np.linspace(0, 1, T)),    # above band
    ])

    mono2 = mono1 * 0.5  # second matrix scaled

    dmat, dmatNoF = to_differential([mono1, mono2], sr=sr, f=f)

    # Output checks
    assert isinstance(dmat, list)
    assert isinstance(dmatNoF, list)
    assert len(dmat) == 2
    assert len(dmatNoF) == 2
    assert dmat[0].shape == (2, T)
    assert dmatNoF[0].shape == (2, T)

    # Differential signal: second - first
    expected_raw = mono1[1:, :] - mono1[:-1, :]
    np.testing.assert_allclose(dmatNoF[0], expected_raw, atol=1e-10)

    # Filtered signal should differ from unfiltered
    assert not np.allclose(dmat[0], dmatNoF[0], atol=1e-3)


def test_to_differential_errors():
    sr = 1000
    f = {"n": 3, "low": 10, "up": 400}  # invalid order

    mono = np.random.randn(1, 1000)
    with np.testing.assert_raises(ValueError):
        to_differential([mono], sr, f)

    f["n"] = 4
    with np.testing.assert_raises(ValueError):
        to_differential([], sr, f)

    with np.testing.assert_raises(ValueError):
        to_differential([np.ones((1, 1000))], sr, f)

    with np.testing.assert_raises(ValueError):
        to_differential([np.ones((2, 1000)), np.ones((3, 1000))], sr, f)
