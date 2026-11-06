"""Tests for the pure registration module."""

import numpy as np
import pytest

from mvc.models.registration import (
    RegistrationConfig,
    SimpleITKRegistrar,
    _magnitude_max,
    _magnitude_rms,
    _ncc,
)


def _phantom(shape=(48, 48)):
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    image = np.exp(-((y - 16.0) ** 2 + (x - 18.0) ** 2) / 40.0)
    image += 0.6 * np.exp(-((y - 32.0) ** 2 + (x - 30.0) ** 2) / 60.0)
    rng = np.random.default_rng(0)
    image += 0.01 * rng.normal(size=shape)
    return image


def _config(**overrides):
    values = dict(
        random_seed=1234,
        iterations=5,
        shrink_factors=(2, 1),
        smoothing_sigmas=(0.0, 0.0),
    )
    values.update(overrides)
    return RegistrationConfig(**values)


def test_defaults_match_the_historical_controller_configuration():
    config = RegistrationConfig()

    assert config.metric == "mattes_mi"
    assert config.histogram_bins == 50
    assert config.sampling_percentage == 0.1
    assert config.random_seed is None  # wall clock, as before extraction
    assert config.learning_rate == 1.0
    assert config.iterations == 100
    assert config.convergence_min_value == 1e-6
    assert config.convergence_window == 10
    assert config.shrink_factors == (4, 2, 1)
    assert config.smoothing_sigmas == (2.0, 1.0, 0.0)
    assert config.mesh_size == 8
    assert config.dice_warning_threshold == 0.9
    assert config.transform_kind == "bspline"
    assert config.foreground_mask is False


def test_register_complex_returns_warped_arrays_and_metrics():
    fixed = _phantom()
    moving = np.roll(fixed, (2, -1), axis=(0, 1))
    imag = moving * 0.5

    result = SimpleITKRegistrar().register_complex(
        fixed, fixed, moving, imag, _config()
    )

    assert result.status in ("ok", "warning")
    assert result.succeeded
    assert result.warped_real is not None
    assert result.warped_imag is not None
    assert result.warped_real.shape == fixed.shape
    assert result.warped_imag.shape == imag.shape
    assert result.transform is not None
    assert result.metric_name == "MattesMutualInformation"
    assert 0.0 <= result.dice <= 1.0
    assert result.elapsed_s >= 0.0
    assert result.iterations >= 0
    assert result.stop_condition
    assert np.isfinite(result.ncc_before)
    assert np.isfinite(result.ncc_after)
    assert np.isfinite(result.mae_before)
    assert np.isfinite(result.mae_after)
    assert result.displacement_rms >= 0.0
    assert result.displacement_max >= result.displacement_rms
    assert 0.0 <= result.folding_ratio <= 1.0


def test_register_complex_does_not_mutate_inputs():
    fixed = _phantom()
    moving = np.roll(fixed, (2, -1), axis=(0, 1))
    imag = moving * 0.5
    fixed_before = fixed.copy()
    moving_before = moving.copy()
    imag_before = imag.copy()

    SimpleITKRegistrar().register_complex(fixed, fixed, moving, imag, _config())

    np.testing.assert_array_equal(fixed, fixed_before)
    np.testing.assert_array_equal(moving, moving_before)
    np.testing.assert_array_equal(imag, imag_before)


def test_real_and_imaginary_use_the_same_transform():
    fixed = _phantom()
    moving = np.roll(fixed, (2, -1), axis=(0, 1))

    result = SimpleITKRegistrar().register_complex(
        fixed, fixed, moving, moving, _config()
    )

    assert result.warped_imag is not None
    np.testing.assert_allclose(result.warped_real, result.warped_imag)


def test_registration_is_reproducible_with_a_fixed_seed():
    fixed = _phantom()
    moving = np.roll(fixed, (2, -1), axis=(0, 1))

    first = SimpleITKRegistrar().register_complex(
        fixed, fixed, moving, None, _config()
    )
    second = SimpleITKRegistrar().register_complex(
        fixed, fixed, moving, None, _config()
    )

    assert first.warped_real is not None
    assert second.warped_real is not None
    np.testing.assert_allclose(first.warped_real, second.warped_real, atol=1e-6)


def test_missing_imaginary_component_is_allowed():
    fixed = _phantom()
    moving = np.roll(fixed, (2, -1), axis=(0, 1))

    result = SimpleITKRegistrar().register_complex(
        fixed, None, moving, None, _config()
    )

    assert result.succeeded
    assert result.warped_imag is None


def test_missing_input_raises_value_error():
    with pytest.raises(ValueError):
        SimpleITKRegistrar().register_complex(None, None, np.zeros((4, 4)), None)


def test_mismatched_shapes_raise_value_error():
    with pytest.raises(ValueError):
        SimpleITKRegistrar().register_complex(
            np.zeros((4, 4)), None, np.zeros((5, 5)), None
        )


def test_ncc_is_undefined_for_a_constant_image():
    constant_a = np.full((4, 4), 5.0)
    constant_b = np.full((4, 4), 9.0)
    varying = np.arange(16, dtype=float).reshape(4, 4)

    assert np.isnan(_ncc(constant_a, constant_b))
    assert np.isnan(_ncc(constant_a, constant_a))
    assert np.isnan(_ncc(constant_a, varying))
    assert _ncc(varying, varying) == pytest.approx(1.0)
    assert _ncc(varying, -varying) == pytest.approx(-1.0)


def test_magnitude_diagnostics_stay_nan_when_the_field_cannot_be_read():
    assert np.isnan(_magnitude_rms(object()))
    assert np.isnan(_magnitude_max(object()))
    assert np.isnan(_magnitude_rms(None))
    assert np.isnan(_magnitude_max(None))


def test_rigid_and_affine_are_opt_in():
    fixed = _phantom((32, 32))
    moving = np.roll(fixed, (2, -1), axis=(0, 1))

    rigid = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(transform_kind="rigid", iterations=15),
    )
    affine = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(transform_kind="affine", iterations=15),
    )

    assert rigid.succeeded
    assert affine.succeeded
    assert rigid.warped_real.shape == fixed.shape
    assert affine.warped_real.shape == fixed.shape
    assert "Euler" in rigid.transform.GetName()
    assert "Affine" in affine.transform.GetName()


def test_foreground_mask_is_opt_in_and_can_run():
    fixed = _phantom((32, 32))
    moving = np.roll(fixed, (1, -1), axis=(0, 1))

    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(foreground_mask=True, foreground_percentile=50, iterations=5),
    )

    assert result.succeeded
    assert result.config.foreground_mask is True
    assert result.warped_real.shape == fixed.shape


def test_unknown_transform_kind_fails_without_a_partial_warp():
    fixed = _phantom((16, 16))

    result = SimpleITKRegistrar().register_complex(
        fixed, None, fixed, None, _config(transform_kind="tps")
    )

    assert result.status == "failed"
    assert result.warped_real is None
    assert "transform_kind" in result.message


def test_execute_failure_is_reported_in_the_result():
    class FailingRegistrar(SimpleITKRegistrar):
        def _execute(self, fixed_image, moving_image, config):
            raise RuntimeError("simulated registration failure")

    fixed = _phantom()
    moving = np.roll(fixed, (2, -1), axis=(0, 1))

    result = FailingRegistrar().register_complex(fixed, fixed, moving, None, _config())

    assert result.status == "failed"
    assert not result.succeeded
    assert result.warped_real is None
    assert result.warped_imag is None
    assert "simulated registration failure" in result.message
