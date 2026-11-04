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
