"""Synthetic tests for the opt-in INR registrar and the RPCA analysis step."""

import numpy as np
import pytest

from mvc.models.inr_registration import (
    CoordinateMLP,
    INRRegistrationConfig,
    INRVolumeRegistrar,
    low_rank_sparse_decomposition,
    mind_descriptor,
)
from mvc.models.registration import RegistrationConfig, SimpleITKRegistrar


def _phantom(shape=(16, 16), shift=(0.0, 0.0)):
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    image = np.exp(-((y - 5.0 - shift[0]) ** 2 + (x - 6.0 - shift[1]) ** 2) / 8.0)
    image += 0.7 * np.exp(-((y - 11.0 - shift[0]) ** 2 + (x - 9.0 - shift[1]) ** 2) / 12.0)
    return image + 0.05


def _fast_config(**overrides):
    values = {
        "iterations": 6,
        "spatial_samples": 64,
        "learning_rate": 0.15,
        "random_seed": 7,
    }
    values.update(overrides)
    return INRRegistrationConfig(**values)


def test_coordinate_mlp_is_four_layers_of_64_and_jacobian_is_analytic():
    mlp = CoordinateMLP(input_dim=2, rank=2, seed=1)
    assert [weight.shape[1] for weight in mlp.weights[:4]] == [64, 64, 64, 64]
    assert mlp.weights[-1].shape == (64, 2)

    coords = np.linspace(-0.8, 0.8, 8).reshape(4, 2)
    basis = mlp.evaluate(coords)
    assert basis.values.shape == (4, 2)
    assert basis.jacobian.shape == (4, 2, 2)

    step = 1e-5
    finite = np.zeros_like(basis.jacobian)
    for axis in range(2):
        plus = coords.copy()
        minus = coords.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        finite[:, :, axis] = (
            mlp.evaluate(plus).values - mlp.evaluate(minus).values
        ) / (2.0 * step)
    np.testing.assert_allclose(basis.jacobian, finite, atol=1e-8, rtol=1e-6)


def test_jacobian_parameter_gradients_match_finite_differences():
    mlp = CoordinateMLP(input_dim=2, rank=2, seed=2)
    coords = np.array([[-0.4, 0.2], [0.1, -0.3], [0.5, 0.5]], dtype=float)
    upstream = np.random.default_rng(0).normal(size=(3, 2, 2))
    weight_grads, bias_grads = mlp.jacobian_parameter_gradients(coords, upstream)

    def loss():
        return float(np.sum(upstream * mlp.evaluate(coords).jacobian))

    step = 1e-6
    for layer, weight in enumerate(mlp.weights):
        row = min(1, weight.shape[0] - 1)
        col = min(2, weight.shape[1] - 1)
        original = float(weight[row, col])
        weight[row, col] = original + step
        plus = loss()
        weight[row, col] = original - step
        minus = loss()
        weight[row, col] = original
        finite = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(weight_grads[layer][row, col], finite, atol=1e-6, rtol=1e-5)

        bias = mlp.biases[layer]
        original = float(bias[col])
        bias[col] = original + step
        plus = loss()
        bias[col] = original - step
        minus = loss()
        bias[col] = original
        finite = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(bias_grads[layer][col], finite, atol=1e-6, rtol=1e-5)


def test_rank_and_architecture_are_rejected_outside_the_specification():
    with pytest.raises(ValueError):
        INRRegistrationConfig(rank=4)
    with pytest.raises(ValueError):
        INRRegistrationConfig(hidden_units=32)
    with pytest.raises(ValueError):
        CoordinateMLP(input_dim=2, rank=1)


def test_register_volume_checks_shapes_and_stays_finite():
    volume = np.stack([_phantom(), _phantom(shift=(1.0, -0.5))], axis=0)
    result = INRVolumeRegistrar().register_volume(volume, config=_fast_config())

    assert result.status in ("ok", "warning")
    assert result.warped_real.shape == (16, 16)
    assert result.transform["warped"].shape == volume.shape
    assert result.transform["displacement"].shape == (2, 16, 16, 2)
    assert result.transform["coefficients"].shape == (2, 2)
    assert result.transform["translations"].shape == (2, 2)
    assert result.metric_name == "projection+MIND"
    assert np.isfinite(result.final_metric)
    assert np.all(np.isfinite(result.transform["warped"]))
    assert np.all(np.isfinite(result.transform["displacement"]))
    np.testing.assert_array_equal(result.transform["coefficients"][0], 0.0)
    np.testing.assert_array_equal(result.transform["translations"][0], 0.0)
    assert len(result.transform["loss_history"]) == result.iterations


def test_rank_three_three_dimensional_case_keeps_its_shapes():
    spatial = (8, 8, 8)
    z, y, x = np.mgrid[0:8, 0:8, 0:8].astype(float)
    fixed = np.exp(-((z - 3.0) ** 2 + (y - 3.0) ** 2 + (x - 4.0) ** 2) / 4.0) + 0.05
    moving = np.exp(-((z - 3.6) ** 2 + (y - 2.5) ** 2 + (x - 4.4) ** 2) / 4.0) + 0.05
    volume = np.stack([fixed, moving, fixed], axis=0)

    result = INRVolumeRegistrar().register_volume(
        volume, config=_fast_config(rank=3, iterations=3, spatial_samples=48)
    )

    assert result.status in ("ok", "warning")
    assert result.transform["coefficients"].shape == (3, 3)
    assert result.transform["translations"].shape == (3, 3)
    assert result.transform["displacement"].shape == (3,) + spatial + (3,)
    assert result.transform["warped"].shape == volume.shape
    assert np.all(np.isfinite(result.transform["displacement"]))


def test_inputs_are_not_mutated_and_components_share_one_displacement():
    fixed = _phantom()
    moving = _phantom(shift=(1.0, -0.5))
    fixed_imag = 0.25 * fixed
    moving_imag = 0.25 * moving
    before = [array.copy() for array in (fixed, moving, fixed_imag, moving_imag)]

    result = INRVolumeRegistrar().register_complex(
        fixed, fixed_imag, moving, moving_imag, _fast_config(iterations=4)
    )

    for original, snapshot in zip((fixed, moving, fixed_imag, moving_imag), before):
        np.testing.assert_array_equal(original, snapshot)
    assert result.warped_imag is not None
    np.testing.assert_allclose(result.warped_imag, 0.25 * result.warped_real, atol=1e-8)


def test_identity_pair_stays_at_the_reference():
    image = _phantom()
    result = INRVolumeRegistrar().register_volume(
        np.stack([image, image], axis=0),
        config=_fast_config(iterations=4),
    )

    metrics = result.transform["metrics"]
    assert metrics["ncc_before"] == pytest.approx(1.0)
    assert metrics["ncc_after"] == pytest.approx(1.0, abs=1e-8)
    assert result.transform["loss_history"][-1] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(result.transform["displacement"], 0.0, atol=1e-8)


def test_known_shift_reduces_the_objective_and_improves_ncc():
    fixed = _phantom(shape=(20, 20))
    moving = _phantom(shape=(20, 20), shift=(1.5, -1.0))
    result = INRVolumeRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _fast_config(iterations=20, spatial_samples=160, learning_rate=0.15),
    )

    history = result.transform["loss_history"]
    metrics = result.transform["metrics"]
    assert history[-1] < history[0]
    assert metrics["ncc_after"] > metrics["ncc_before"]
    assert metrics["finite"]
    assert result.warped_imag is None
    mean_shift = result.transform["displacement"][1].mean(axis=(0, 1))
    assert mean_shift[0] == pytest.approx(1.5, abs=0.45)
    assert mean_shift[1] == pytest.approx(-1.0, abs=0.45)


def test_same_seed_reproduces_the_warp():
    fixed = _phantom()
    moving = _phantom(shift=(1.0, 0.5))
    config = _fast_config()
    first = INRVolumeRegistrar().register_complex(fixed, None, moving, None, config)
    second = INRVolumeRegistrar().register_complex(fixed, None, moving, None, config)

    np.testing.assert_allclose(first.warped_real, second.warped_real, atol=1e-12)
    np.testing.assert_array_equal(
        first.transform["coefficients"], second.transform["coefficients"]
    )


def test_simpleitk_config_is_not_accepted_by_the_inr_registrar():
    with pytest.raises(ValueError, match="INRRegistrationConfig"):
        INRVolumeRegistrar().register_complex(
            _phantom(), None, _phantom(), None, config=RegistrationConfig()
        )


def test_bad_volumes_are_rejected_and_optimizer_errors_are_reported():
    with pytest.raises(ValueError):
        INRVolumeRegistrar().register_volume(np.ones((1, 8, 8)))
    with pytest.raises(ValueError):
        INRVolumeRegistrar().register_volume(np.array([np.nan]))

    class Broken(INRVolumeRegistrar):
        pass

    original = __import__(
        "mvc.models.inr_registration", fromlist=["_optimize"]
    )._optimize

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic optimizer failure")

    module = __import__("mvc.models.inr_registration", fromlist=["_optimize"])
    module._optimize = explode
    try:
        result = Broken().register_volume(np.stack([_phantom(), _phantom()]), _fast_config())
    finally:
        module._optimize = original

    assert result.status == "failed"
    assert result.warped_real is None
    assert "synthetic optimizer failure" in result.message


def test_default_simpleitk_configuration_is_unchanged():
    config = RegistrationConfig()

    assert config.metric == "mattes_mi"
    assert config.histogram_bins == 50
    assert config.sampling_percentage == 0.1
    assert config.random_seed is None
    assert config.iterations == 100
    assert config.shrink_factors == (4, 2, 1)
    assert config.smoothing_sigmas == (2.0, 1.0, 0.0)
    assert config.mesh_size == 8
    assert isinstance(SimpleITKRegistrar(), SimpleITKRegistrar)


def test_mind_descriptor_is_a_finite_distribution_and_shift_invariant():
    image = _phantom()
    descriptor = mind_descriptor(image, radius=1, sigma=0.5)

    assert descriptor.shape == image.shape + (8,)
    assert np.all(np.isfinite(descriptor))
    np.testing.assert_allclose(descriptor.sum(axis=-1), 1.0, atol=1e-6)
    shifted = mind_descriptor(image + 3.0, radius=1, sigma=0.5)
    np.testing.assert_allclose(descriptor, shifted, atol=1e-8)


def test_low_rank_sparse_split_is_separate_and_finds_injected_peaks():
    phases, height, width = 6, 10, 8
    time = np.linspace(0.0, 1.0, phases)
    background = (0.4 + 0.5 * time)[:, None, None] * np.linspace(0.2, 1.0, height)[None, :, None]
    background = background + (0.2 * np.sin(2.0 * np.pi * time))[:, None, None]
    series = np.broadcast_to(background, (phases, height, width)).copy()
    series[2, 3, 4] += 5.0
    series[4, 7, 1] -= 4.0

    result = low_rank_sparse_decomposition(series, rank=2, sparsity=0.02, iterations=20)

    assert result.low_rank.shape == series.shape
    assert result.sparse.shape == series.shape
    assert result.rank == 2
    # ``sparsity`` is a fraction; the implementation keeps a whole number of
    # entries, so the realized fraction can exceed it by less than one entry.
    assert result.sparse_fraction <= 0.02 + 1.0 / series.size
    assert np.all(np.isfinite(result.low_rank))
    assert np.all(np.isfinite(result.sparse))
    nonzero = {tuple(index) for index in np.argwhere(result.sparse != 0)}
    assert (2, 3, 4) in nonzero
    assert (4, 7, 1) in nonzero
    assert (0, 0, 0) not in nonzero


def test_linear_sampler_matches_map_coordinates_and_its_slope():
    from scipy.ndimage import map_coordinates

    from mvc.models.inr_registration import _sample_linear

    rng = np.random.default_rng(0)
    image = rng.normal(size=(9, 8))
    index = np.array(
        [
            [2.3, 4.7],
            [3.0, 5.0],
            [0.0, 0.0],
            [-0.4, 2.2],
            [8.2, 7.6],
            [1.0, 6.5],
        ]
    )
    values, slope = _sample_linear(image, index)
    sampled = map_coordinates(image, index.T, order=1, mode="nearest", prefilter=False)
    np.testing.assert_allclose(values[:, 0], sampled, atol=1e-8)

    step = 1e-5
    for axis in range(2):
        plus = index.copy()
        minus = index.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        right = map_coordinates(image, plus.T, order=1, mode="nearest", prefilter=False)
        left = map_coordinates(image, minus.T, order=1, mode="nearest", prefilter=False)
        finite = (right - left) / (2.0 * step)
        np.testing.assert_allclose(slope[:, 0, axis], finite, atol=1e-5, rtol=1e-4)

    volume = rng.normal(size=(5, 6, 4, 3))
    index3 = np.array([[1.2, 2.4, 0.3], [0.0, 5.5, 3.0], [4.2, -0.2, 1.7]])
    values, slope = _sample_linear(volume, index3)
    assert values.shape == (3, 3)
    assert slope.shape == (3, 3, 3)
    for channel in range(3):
        sampled = map_coordinates(
            volume[..., channel], index3.T, order=1, mode="nearest", prefilter=False
        )
        np.testing.assert_allclose(values[:, channel], sampled, atol=1e-8)


def test_localizer_profile_ignores_a_balanced_ray():
    from mvc.models.inr_registration import _projection_loss_and_intensity_grad

    shape = (6, 6)
    voxels = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
    scale = 0.5 * (np.asarray(shape) - 1.0)
    coords = voxels / scale - 1.0
    # Column means and row means both equal the reference, so every localizer
    # ray matches.  A photometric residual of these four samples would not.
    samples = np.array([0.0, 1.0, 1.0, 0.0])
    reference = np.full(4, 0.5)
    loss, gradient = _projection_loss_and_intensity_grad(
        samples, reference, coords, shape
    )
    assert loss == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(gradient, 0.0, atol=1e-12)

    shifted = np.array([0.0, 1.0, 0.0, 1.0])
    loss, gradient = _projection_loss_and_intensity_grad(
        shifted, reference, coords, shape
    )
    assert loss == pytest.approx(0.125, abs=1e-12)
    step = 1e-6
    plus = shifted.copy()
    minus = shifted.copy()
    plus[0] += step
    minus[0] -= step
    loss_plus, _ = _projection_loss_and_intensity_grad(plus, reference, coords, shape)
    loss_minus, _ = _projection_loss_and_intensity_grad(minus, reference, coords, shape)
    finite = (loss_plus - loss_minus) / (2.0 * step)
    assert gradient[0] == pytest.approx(finite, rel=1e-4, abs=1e-7)


def test_wide_layer_step_is_not_shrunk_by_an_l2_clip():
    from mvc.models.inr_registration import _update_vector

    weight = np.zeros((64, 64))
    gradient = np.ones((64, 64))
    velocity = np.zeros_like(weight)
    _update_vector(weight, gradient, velocity, step=0.15, momentum=0.8, clip=5.0)

    rms = float(np.sqrt(np.mean(np.square(weight))))
    # First momentum step is (1 - 0.8) times a unit-RMS direction.
    # An L2 cap of 5 would have multiplied this by 5/64.
    assert rms == pytest.approx(0.15 * 0.2, rel=1e-6)
    assert rms > 0.02


def test_temporal_gradient_matches_the_smoothness_term():
    from mvc.models.inr_registration import _one_temporal_gradient

    coefficients = np.array(
        [[0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [0.1, 0.4, -0.3], [-0.2, 0.2, 0.05]]
    )
    gradient = _one_temporal_gradient(coefficients, reference_index=0)
    np.testing.assert_array_equal(gradient[0], 0.0)

    def part(values):
        return 0.5 * float(np.mean(np.square(np.diff(values, axis=0))))

    step = 1e-6
    for phase, column in ((1, 0), (2, 1), (3, 2)):
        plus = coefficients.copy()
        minus = coefficients.copy()
        plus[phase, column] += step
        minus[phase, column] -= step
        finite = (part(plus) - part(minus)) / (2.0 * step)
        assert gradient[phase, column] == pytest.approx(finite, rel=1e-4, abs=1e-8)


def test_objective_gradients_match_finite_differences():
    from mvc.models.inr_registration import (
        CoordinateMLP,
        _objective_and_gradients,
        _prepare_phase_terms,
        _sample_coordinates,
    )

    config = _fast_config(
        spatial_samples=24,
        projection_weight=1.0,
        mind_weight=0.4,
        temporal_smooth_weight=0.05,
        weight_decay=0.0,
    )
    volume = np.stack(
        [
            _phantom(shape=(12, 12)),
            _phantom(shape=(12, 12), shift=(1.0, -0.4)),
            _phantom(shape=(12, 12), shift=(0.3, 0.5)),
        ],
        axis=0,
    )
    coords, _choice = _sample_coordinates(volume.shape[1:], config.spatial_samples, config.random_seed)
    mlp = CoordinateMLP(input_dim=2, rank=2, seed=3)
    coefficients = np.zeros((3, 2))
    translations = np.array([[0.0, 0.0], [0.25, -0.15], [0.1, 0.2]])
    coefficients[1] = np.array([0.04, -0.02])
    coefficients[2] = np.array([-0.03, 0.05])
    prepared = _prepare_phase_terms(volume, coords, config)

    def scalar():
        terms = _objective_and_gradients(
            mlp, coefficients, translations, coords, prepared, 0, config
        )
        return (
            config.projection_weight * terms["projection"]
            + config.mind_weight * terms["mind"]
            + config.temporal_smooth_weight * terms["temporal"]
        )

    terms = _objective_and_gradients(
        mlp, coefficients, translations, coords, prepared, 0, config
    )
    step = 1e-5
    for phase, axis in ((1, 0), (1, 1), (2, 0)):
        original = float(translations[phase, axis])
        translations[phase, axis] = original + step
        plus = scalar()
        translations[phase, axis] = original - step
        minus = scalar()
        translations[phase, axis] = original
        finite = (plus - minus) / (2.0 * step)
        assert terms["translation_grad"][phase, axis] == pytest.approx(
            finite, rel=1e-3, abs=1e-5
        )

    original = float(coefficients[1, 0])
    coefficients[1, 0] = original + step
    plus = scalar()
    coefficients[1, 0] = original - step
    minus = scalar()
    coefficients[1, 0] = original
    finite = (plus - minus) / (2.0 * step)
    assert terms["coefficient_grad"][1, 0] == pytest.approx(finite, rel=1e-3, abs=1e-5)

    weight = mlp.weights[0]
    original = float(weight[0, 0])
    weight[0, 0] = original + step
    plus = scalar()
    weight[0, 0] = original - step
    minus = scalar()
    weight[0, 0] = original
    finite = (plus - minus) / (2.0 * step)
    assert terms["weight_grads"][0][0, 0] == pytest.approx(finite, rel=1e-2, abs=1e-6)


def test_curl_free_stretch_is_not_a_pure_translation():
    from scipy.ndimage import map_coordinates

    shape = (28, 28)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    fixed = np.exp(-((y - shape[0] * 0.5) ** 2 + (x - shape[1] * 0.35) ** 2) / 8.0)
    fixed += np.exp(-((y - shape[0] * 0.5) ** 2 + (x - shape[1] * 0.68) ** 2) / 8.0)
    fixed += 0.02
    center = (shape[1] - 1) / 2.0
    # u_x = k (x - cx) is the gradient of a quadratic, so the low-rank
    # Jacobian basis can represent it.  A single translation cannot: the two
    # lobes move in opposite directions.
    stretch_x = 1.5 * (x - center) / center
    moving = map_coordinates(
        fixed,
        np.stack([y, x + stretch_x], axis=0),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    result = INRVolumeRegistrar().register_volume(
        np.stack([fixed, moving], axis=0),
        config=_fast_config(iterations=25, spatial_samples=120, learning_rate=0.05),
    )

    displacement = result.transform["displacement"][1, :, :, 1]
    metrics = result.transform["metrics"]
    assert result.status in ("ok", "warning")
    assert metrics["ncc_after"] > metrics["ncc_before"] + 0.005
    assert float(np.std(displacement)) > 0.25
    centered_x = x - float(np.mean(x))
    correlation = float(
        np.mean(centered_x * (displacement - np.mean(displacement)))
        / (np.std(centered_x) * np.std(displacement))
    )
    # moving(x) = fixed(x + stretch), so the pull field points against x.
    assert correlation < -0.5


def test_default_temperature_path_does_not_reference_the_inr_registrar():
    import inspect

    import mvc.controllers.thermometry_controller as controller
    import mvc.models.cinema_model as cinema
    import mvc.models.thermometry_model as model

    for module in (controller, cinema, model):
        source = inspect.getsource(module)
        assert "inr_registration" not in source
        assert "INRVolumeRegistrar" not in source
