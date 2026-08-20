import numpy as np
import pytest
from scipy.special import gammaln

from hdsemg_shared.filters.bandpass import bandpass_filter_exact_corners
from hdsemg_shared.filters.smoothing import moving_average
from hdsemg_shared.global_parameters.global_amplitude import global_amplitude
from hdsemg_shared.preprocessing.grid_map import emg_map_from_indices, map_to_columns

FS = 2048.0
N_SAMPLES = int(4 * FS)


def _steady(signal):
    """The middle half of a signal, where no filter transient reaches."""
    return signal[len(signal) // 4:3 * len(signal) // 4]


def _sine_grid(amplitude, n_channels = 12):
    """One in-band sine per channel, each at its own frequency."""
    t = np.arange(N_SAMPLES) / FS
    return np.vstack(
        [amplitude * np.sqrt(2) * np.sin(2 * np.pi * (80 + 3 * i) * t) for i in range(n_channels)]
    )


def _chi_factor(n):
    """E[chi_n]/sqrt(n), the bias of rooting a mean of only n squared samples."""
    return np.exp(gammaln((n + 1) / 2) - gammaln(n / 2)) * np.sqrt(2) / np.sqrt(n)


def test_rms_of_a_known_sine_grid_is_its_amplitude_over_root_two():
    emg = _sine_grid(100.0)

    out = global_amplitude(emg, emg_map_from_indices(range(12), 4, 3), FS)

    assert _steady(out.amplitude).mean() == pytest.approx(100.0, rel=1e-3)


def test_arv_of_a_known_sine_grid_is_two_over_pi_of_the_peak():
    emg = _sine_grid(100.0)

    out = global_amplitude(emg, emg_map_from_indices(range(12), 4, 3), FS, method="ARV")

    assert _steady(out.amplitude).mean() == pytest.approx(2 * 100.0 * np.sqrt(2) / np.pi, rel=1e-3)


def test_root_is_taken_last_so_space_and_time_averaging_commute():
    # Merletti & Cerone eq. 5.1/5.2: the RMS of a region is the root of the
    # mean of f^2 over BOTH space and time, taken at the very end, which makes
    # mean_space(RMS^2) == mean_time(RMS^2). A rectangular window as long as
    # the epoch turns the amplitude time series into exactly that number.
    # This identity is what would break if the root moved earlier.
    rng = np.random.default_rng(7)
    emg = rng.standard_normal((12, N_SAMPLES)) * 50

    out = global_amplitude(
        emg,
        emg_map_from_indices(range(12), 4, 3),
        FS,
        smooth={"mode": "moving", "window_s": N_SAMPLES / FS, "kernel": "rectangular"},
        pad_s=0.0,
    )
    over_space_and_time = np.sqrt(
        np.mean(bandpass_filter_exact_corners(emg, 2, 15.0, 450.0, FS) ** 2)
    )

    assert out.amplitude[N_SAMPLES // 2] == pytest.approx(over_space_and_time, rel=1e-12)


def test_amplitude_does_not_depend_on_how_many_channels_survived():
    # Channels are taken from ONE pool so only the count varies. The MATLAB
    # ordering -- reduce across channels at every sample, smooth afterwards --
    # roots a mean of only nCh squared samples and therefore sits below the
    # true amplitude by E[chi_n]/sqrt(n). That matters because an MVC trial
    # and a tracking trial rarely keep the same channels and their ratio is
    # the reported %MVC.
    rng = np.random.default_rng(7)
    pool = rng.standard_normal((48, int(8 * FS))) * 50
    filtered = bandpass_filter_exact_corners(pool, 2, 15.0, 450.0, FS)

    root_last, matlab_order = [], []
    for k in (4, 8, 12, 24, 48):
        out = global_amplitude(pool[:k], emg_map_from_indices(range(k), k, 1), FS)
        root_last.append(_steady(out.amplitude).mean())

        collapsed = np.sqrt(np.mean(filtered[:k] ** 2, axis=0))
        matlab_order.append(_steady(moving_average(collapsed, FS, fc=15.0)).mean())

        # the shortfall follows the predicted factor to within 0.2 %
        assert matlab_order[-1] / root_last[-1] == pytest.approx(_chi_factor(k), rel=2e-3)

    spread = (max(root_last) - min(root_last)) / np.mean(root_last)
    matlab_spread = (max(matlab_order) - min(matlab_order)) / np.mean(matlab_order)
    assert spread < 0.01
    assert matlab_spread > 0.04


def test_a_nan_channel_is_ignored():
    emg = _sine_grid(100.0, n_channels=13)
    emg[12] = np.nan
    emg_map = emg_map_from_indices(range(12), 4, 3)

    without = global_amplitude(emg[:12], emg_map, FS)
    with_nan = global_amplitude(emg, np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 12]]), FS)

    assert without.n_channels == 12
    assert with_nan.n_channels == 11


def test_a_nan_map_entry_drops_that_position():
    emg = _sine_grid(100.0)
    emg_map = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, np.nan]]

    out = global_amplitude(emg, emg_map, FS)

    assert out.n_channels == 11
    assert out.grid_shape == (3, 4)
    assert out.per_channel.shape[0] == 11
    assert (3, 3) not in [tuple(p) for p in out.positions]


def test_extra_channels_outside_the_map_do_not_contribute():
    # A recording carries force and path references next to the grid.
    emg = np.vstack([_sine_grid(100.0), np.full((3, N_SAMPLES), 1e6)])

    out = global_amplitude(emg, emg_map_from_indices(range(12), 4, 3), FS)

    assert out.n_channels == 12
    assert _steady(out.amplitude).mean() == pytest.approx(100.0, rel=1e-3)


@pytest.mark.parametrize("derivation,rows", [("MP", 4), ("SD", 3), ("DD", 2)])
def test_derivation_reduces_the_row_count(derivation, rows):
    emg = _sine_grid(100.0)

    out = global_amplitude(emg, emg_map_from_indices(range(12), 4, 3), FS, derivation=derivation)

    assert out.grid_shape == (3, rows)
    assert out.n_channels == 3 * rows


def test_single_differential_is_the_plain_difference_along_a_column():
    rng = np.random.default_rng(0)
    emg = rng.standard_normal((12, N_SAMPLES)) * 50
    emg_map = emg_map_from_indices(range(12), 4, 3)

    # Feed the differences in directly as a monopolar grid and compare
    manual = np.vstack([np.diff(emg[j * 4:(j + 1) * 4], axis=0) for j in range(3)])
    from_manual = global_amplitude(manual, emg_map_from_indices(range(9), 3, 3), FS)
    from_derivation = global_amplitude(emg, emg_map, FS, derivation="SD")

    np.testing.assert_allclose(from_derivation.amplitude, from_manual.amplitude, rtol=1e-12)


def test_a_single_channel_arv_is_its_smoothed_rectified_signal():
    # MATLAB collapses all three of its outputs to lowpass(|x|) at nCh == 1.
    # ARV still does that; RMS stays the moving RMS, the consistent extension.
    t = np.arange(N_SAMPLES) / FS
    emg = np.sqrt(2) * 100.0 * np.sin(2 * np.pi * 100.0 * t).reshape(1, -1)

    arv = global_amplitude(emg, [[0]], FS, method="ARV")
    rms = global_amplitude(emg, [[0]], FS)

    assert arv.n_channels == 1
    assert _steady(arv.amplitude).mean() == pytest.approx(2 * 100.0 * np.sqrt(2) / np.pi, rel=1e-3)
    assert _steady(rms.amplitude).mean() == pytest.approx(100.0, rel=1e-2)


def test_per_channel_is_an_amplitude_not_a_squared_quantity():
    # It is what a density map plots, so it must carry the same unit as
    # amplitude. Rows of equal-amplitude sines must each sit at that amplitude.
    emg = _sine_grid(100.0)

    out = global_amplitude(emg, emg_map_from_indices(range(12), 4, 3), FS)

    for row in out.per_channel:
        assert _steady(row).mean() == pytest.approx(100.0, rel=2e-2)


def test_positions_locate_each_kept_channel_in_the_derived_grid():
    emg = _sine_grid(100.0)
    emg_map = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, np.nan]]

    out = global_amplitude(emg, emg_map, FS)

    assert out.positions.shape == (11, 2)
    assert out.positions[:, 0].max() == 2   # column index
    assert out.positions[:, 1].max() == 3   # row index


def test_matlab_settings_are_reachable_through_the_arguments():
    # Only the root ordering is irreducibly different from globEMGAmpEnv.
    emg = _sine_grid(100.0)

    out = global_amplitude(
        emg,
        emg_map_from_indices(range(12), 4, 3),
        FS,
        bpf={"N": 2, "fcl": 30.0, "fch": 450.0, "corners": "prewarp"},
        smooth={"mode": "lowpass", "fc": 6.0, "N": 2},
        pad_s=0.0,
    )

    assert np.isfinite(out.amplitude).all()
    assert out.n_channels == 12


def test_the_two_smoothing_modes_agree_on_the_bandwidth():
    rng = np.random.default_rng(3)
    emg = rng.standard_normal((12, N_SAMPLES)) * 50
    emg_map = emg_map_from_indices(range(12), 4, 3)

    moving = global_amplitude(emg, emg_map, FS, smooth={"mode": "moving", "fc": 15.0})
    lowpass = global_amplitude(emg, emg_map, FS, smooth={"mode": "lowpass", "fc": 15.0})

    assert _steady(moving.amplitude).mean() == pytest.approx(
        _steady(lowpass.amplitude).mean(), rel=0.02
    )


def test_amplitude_is_never_negative():
    # A butterworth smoother can undershoot; the result is an amplitude.
    rng = np.random.default_rng(1)
    emg = rng.standard_normal((12, N_SAMPLES)) * 50

    out = global_amplitude(
        emg, emg_map_from_indices(range(12), 4, 3), FS, smooth={"mode": "lowpass", "fc": 6.0}
    )

    assert (out.amplitude >= 0).all()
    assert (out.per_channel >= 0).all()


def test_global_amplitude_rejects_unknown_choices():
    emg = _sine_grid(100.0)
    emg_map = emg_map_from_indices(range(12), 4, 3)

    with pytest.raises(ValueError, match="method must be one of"):
        global_amplitude(emg, emg_map, FS, method="MEDIAN")
    with pytest.raises(ValueError, match="derivation must be one of"):
        global_amplitude(emg, emg_map, FS, derivation="LAPLACE")
    with pytest.raises(ValueError, match="Unknown bpf option"):
        global_amplitude(emg, emg_map, FS, bpf={"low": 15.0})
    with pytest.raises(ValueError, match="Unknown smooth option"):
        global_amplitude(emg, emg_map, FS, smooth={"cutoff": 15.0})
    with pytest.raises(ValueError, match="corners"):
        global_amplitude(emg, emg_map, FS, bpf={"corners": "warped"})
    with pytest.raises(ValueError, match="mode"):
        global_amplitude(emg, emg_map, FS, smooth={"mode": "median"})
    with pytest.raises(ValueError, match="pad_s"):
        global_amplitude(emg, emg_map, FS, pad_s=-1.0)


def test_method_and_derivation_are_case_insensitive():
    emg = _sine_grid(100.0)
    emg_map = emg_map_from_indices(range(12), 4, 3)

    lower = global_amplitude(emg, emg_map, FS, method="rms", derivation="sd")
    upper = global_amplitude(emg, emg_map, FS, method="RMS", derivation="SD")

    np.testing.assert_allclose(lower.amplitude, upper.amplitude)


def test_a_column_too_short_for_the_derivation_is_rejected():
    emg = _sine_grid(100.0, n_channels=6)

    with pytest.raises(ValueError, match="needs more than 2 rows"):
        global_amplitude(emg, emg_map_from_indices(range(6), 2, 3), FS, derivation="DD")


def test_a_grid_with_no_surviving_channel_is_rejected():
    emg = _sine_grid(100.0)
    emg[:] = np.nan

    with pytest.raises(ValueError, match="No channel survived"):
        global_amplitude(emg, emg_map_from_indices(range(12), 4, 3), FS)


def test_the_pipe_path_a_flat_index_list_from_a_grid():
    # hdsemg-pipe holds Grid.emg_indices plus rows/cols and nothing else.
    emg = np.vstack([np.full((8, N_SAMPLES), 0.0), _sine_grid(100.0)])
    emg_indices, rows, cols = list(range(8, 20)), 4, 3

    out = global_amplitude(emg, emg_map_from_indices(emg_indices, rows, cols), FS)

    assert out.n_channels == 12
    assert _steady(out.amplitude).mean() == pytest.approx(100.0, rel=1e-3)


def test_the_select_path_a_display_grid_plus_a_channel_mask():
    # hdsemg-select holds _electrode_display_grid, a (rows, cols) float array
    # whose entries are LOCAL 0-based indices into the grid's own emg_indices
    # (grid_setup_handler._apply_with_layout resolves them as
    # indices[int(display_grid[r, c])]), with NaN at absent positions. It also
    # holds a flat list[bool] selection mask indexed by GLOBAL data column,
    # whose True entries include the reference channels.
    #
    # So a caller has to do two things: resolve local -> global through
    # emg_indices, and intersect the mask with the grid's own channels. Both
    # are the caller's job; excluded positions then become NaN in the map.
    emg = np.vstack([np.full((8, N_SAMPLES), 1e6), _sine_grid(100.0)])
    emg_indices = list(range(8, 20))                 # global data columns
    display_grid = np.array([[0.0, 4.0, 8.0],        # local indices, (rows, cols)
                             [1.0, 5.0, 9.0],
                             [2.0, 6.0, 10.0],
                             [3.0, 7.0, np.nan]])
    channel_status = [True] * 20
    channel_status[13] = False                       # a deselected EMG channel

    emg_map = np.full(display_grid.T.shape, np.nan)  # select stores (rows, cols)
    for j, i in np.ndindex(emg_map.shape):
        local = display_grid[i, j]
        if np.isnan(local):
            continue
        channel = emg_indices[int(local)]
        if channel_status[channel]:
            emg_map[j, i] = channel

    out = global_amplitude(emg, emg_map, FS)

    assert out.n_channels == 10          # 12 minus the absent corner minus one deselected
    assert _steady(out.amplitude).mean() == pytest.approx(100.0, rel=1e-3)


def test_a_channel_with_a_single_nan_sample_is_dropped_whole():
    # "NaN channels are ignored" is implemented as: any NaN in a mapped
    # channel drops that channel. A partially-NaN channel cannot be filtered
    # (filtfilt would poison the whole row), so it must not reach the output.
    emg = _sine_grid(100.0)
    emg[5, N_SAMPLES // 2] = np.nan

    out = global_amplitude(emg, emg_map_from_indices(range(12), 4, 3), FS)

    assert out.n_channels == 11
    assert np.isfinite(out.amplitude).all()
    assert np.isfinite(out.per_channel).all()
    assert (1, 1) not in [tuple(p) for p in out.positions]   # channel 5 sat at col1/row1


def test_a_nan_in_an_unmapped_channel_is_harmless():
    # References the map does not touch must not affect anything.
    emg = np.vstack([_sine_grid(100.0), np.full((2, N_SAMPLES), np.nan)])

    out = global_amplitude(emg, emg_map_from_indices(range(12), 4, 3), FS)

    assert out.n_channels == 12
    assert _steady(out.amplitude).mean() == pytest.approx(100.0, rel=1e-3)


def _constant_grid(n_channels = 12):
    """Channel c holds the constant value c, so differences are exact integers."""
    return np.tile(np.arange(float(n_channels)).reshape(-1, 1), (1, N_SAMPLES))


@pytest.mark.parametrize(
    "derivation,direction,shape",
    [("SD", "cols", (3, 3)), ("DD", "cols", (3, 2)),
     ("SD", "rows", (2, 4)), ("DD", "rows", (1, 4))],
)
def test_diff_direction_picks_the_grid_axis(derivation, direction, shape):
    emg = _sine_grid(100.0)

    out = global_amplitude(emg, emg_map_from_indices(range(12), 4, 3), FS,
                           derivation=derivation, diff_direction=direction)

    assert out.grid_shape == shape
    assert out.n_channels == shape[0] * shape[1]


def test_cols_differences_down_a_column_and_rows_across_one():
    # On a 3-column x 4-row map built column-first, neighbours within a column
    # are 1 apart and neighbours across a row are nRows = 4 apart.
    emg = _constant_grid()
    emg_map = emg_map_from_indices(range(12), rows=4, cols=3)
    columns = np.stack(map_to_columns(emg, emg_map))[:, :, 0]

    np.testing.assert_allclose(np.diff(columns, axis=1), 1.0)   # 'cols'
    np.testing.assert_allclose(np.diff(columns, axis=0), 4.0)   # 'rows'


def test_diff_direction_is_the_only_difference_between_the_two_axes():
    # Transposing the map must turn one direction into the other exactly.
    rng = np.random.default_rng(11)
    emg = rng.standard_normal((12, N_SAMPLES)) * 50
    emg_map = emg_map_from_indices(range(12), rows=4, cols=3)

    along_rows = global_amplitude(emg, emg_map, FS, derivation="SD", diff_direction="rows")
    along_cols_transposed = global_amplitude(emg, np.asarray(emg_map).T, FS,
                                             derivation="SD", diff_direction="cols")

    np.testing.assert_allclose(along_rows.amplitude, along_cols_transposed.amplitude, rtol=1e-12)


def test_diff_direction_defaults_to_cols():
    emg = _sine_grid(100.0)
    emg_map = emg_map_from_indices(range(12), 4, 3)

    default = global_amplitude(emg, emg_map, FS, derivation="SD")
    explicit = global_amplitude(emg, emg_map, FS, derivation="SD", diff_direction="cols")

    np.testing.assert_allclose(default.amplitude, explicit.amplitude, rtol=1e-12)


def test_diff_direction_is_ignored_for_monopolar():
    emg = _sine_grid(100.0)
    emg_map = emg_map_from_indices(range(12), 4, 3)

    by_cols = global_amplitude(emg, emg_map, FS, diff_direction="cols")
    by_rows = global_amplitude(emg, emg_map, FS, diff_direction="rows")

    np.testing.assert_allclose(by_cols.amplitude, by_rows.amplitude, rtol=1e-12)


def test_a_grid_too_narrow_for_the_chosen_direction_is_rejected():
    emg = _sine_grid(100.0, n_channels=8)
    emg_map = emg_map_from_indices(range(8), rows=4, cols=2)

    # 2 columns cannot carry a double difference across the rows
    with pytest.raises(ValueError, match="along 'rows' needs more than 2 columns per row"):
        global_amplitude(emg, emg_map, FS, derivation="DD", diff_direction="rows")

    # but 4 rows per column can carry it down the columns
    assert global_amplitude(emg, emg_map, FS, derivation="DD").grid_shape == (2, 2)


def test_global_amplitude_rejects_an_unknown_diff_direction():
    with pytest.raises(ValueError, match="diff_direction must be one of"):
        global_amplitude(_sine_grid(100.0), emg_map_from_indices(range(12), 4, 3), FS,
                         diff_direction="columns")


def test_diff_direction_is_case_insensitive():
    emg = _sine_grid(100.0)
    emg_map = emg_map_from_indices(range(12), 4, 3)

    lower = global_amplitude(emg, emg_map, FS, derivation="SD", diff_direction="rows")
    upper = global_amplitude(emg, emg_map, FS, derivation="SD", diff_direction="ROWS")

    np.testing.assert_allclose(lower.amplitude, upper.amplitude)
