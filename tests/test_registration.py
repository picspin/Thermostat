"""Tests for the pure registration module."""

import numpy as np
import pytest
import SimpleITK as sitk

from mvc.models.mind import (
    MIND_RADIUS,
    MIND_SAMPLE_STRIDE,
    _geometry,
    _index_to_physical,
    _parameter_gradient,
    _physical_to_index,
    _pull_through_outer,
    mind_descriptor,
    mind_distance,
)
from mvc.models.registration import (
    RegistrationConfig,
    SimpleITKRegistrar,
    _magnitude_max,
    _magnitude_rms,
    _ncc,
    default_registration_config,
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


def _resample_euler(image, angle_deg=0.0, translation=(0.0, 0.0)):
    """Move a 2D array with the same Euler convention the registrar returns."""
    source = sitk.GetImageFromArray(np.asarray(image, dtype=np.float64))
    transform = sitk.Euler2DTransform()
    transform.SetCenter(
        ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
    )
    transform.SetAngle(np.deg2rad(angle_deg))
    transform.SetTranslation(tuple(translation))
    moved = sitk.Resample(source, source, transform, sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(moved)


def _mae(left, right):
    return float(np.mean(np.abs(np.asarray(left) - np.asarray(right))))


def test_rigid_recovers_a_known_rotation_and_improves_mae():
    fixed = _phantom()
    moving = _resample_euler(fixed, angle_deg=10.0)
    before = _mae(fixed, moving)

    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(transform_kind="rigid", iterations=100),
    )

    assert result.succeeded
    assert "Euler2D" in result.transform.GetName()
    # The moving image was produced by +10 degrees, so the inverse is -10.
    recovered = np.degrees(result.transform.GetAngle())
    assert recovered == pytest.approx(-10.0, abs=1.5)
    assert _mae(fixed, result.warped_real) < before


def test_rigid_recovers_a_known_translation_and_improves_mae():
    fixed = _phantom()
    moving = _resample_euler(fixed, translation=(4.0, -3.0))
    before = _mae(fixed, moving)

    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(transform_kind="rigid", iterations=100),
    )

    assert result.succeeded
    translation = result.transform.GetTranslation()
    assert translation[0] == pytest.approx(-4.0, abs=0.5)
    assert translation[1] == pytest.approx(3.0, abs=0.5)
    assert abs(np.degrees(result.transform.GetAngle())) < 1.0
    assert _mae(fixed, result.warped_real) < before


def test_affine_rotation_improves_mae():
    fixed = _phantom()
    moving = _resample_euler(fixed, angle_deg=10.0)
    before = _mae(fixed, moving)

    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(transform_kind="affine", iterations=100),
    )

    assert result.succeeded
    assert "Affine" in result.transform.GetName()
    assert result.warped_real.shape == fixed.shape
    assert _mae(fixed, result.warped_real) < before


def test_default_optimizer_stays_gradient_descent():
    fixed = _phantom((32, 32))
    moving = np.roll(fixed, (1, 0), axis=(0, 1))

    result = SimpleITKRegistrar().register_complex(
        fixed, None, moving, None, _config()
    )

    assert "RegularStep" not in result.stop_condition
    assert "GradientDescent" in result.stop_condition


def test_foreground_mask_excludes_a_zero_background():
    image = np.zeros((30, 30))
    image[8:14, 8:14] = np.linspace(1.0, 5.0, 36).reshape(6, 6)
    mask = sitk.GetArrayFromImage(
        SimpleITKRegistrar._foreground_mask(
            sitk.GetImageFromArray(image), 80.0
        )
    )

    assert np.all(mask[image == 0] == 0)
    assert 0 < int(mask.sum()) < mask.size

    # A phantom with no empty background still runs with the mask on.
    # The sparse image above is only for the mask contents: its support is
    # too small for the metric's coarse pyramid level.
    fixed = _phantom((32, 32))
    moving = np.roll(fixed, (1, -1), axis=(0, 1))
    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(foreground_mask=True, foreground_percentile=50, iterations=10),
    )
    assert result.succeeded
    assert result.config.foreground_mask is True


def _volume_config(transform_kind):
    return RegistrationConfig(
        random_seed=1234,
        iterations=10,
        shrink_factors=(1,),
        smoothing_sigmas=(0.0,),
        sampling_percentage=0.2,
        transform_kind=transform_kind,
    )


def test_rigid_and_affine_run_in_3d():
    size = 12
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    fixed = np.exp(-((z - 5.0) ** 2 + (y - 6.0) ** 2 + (x - 4.0) ** 2) / 8.0)
    moving = np.roll(fixed, (1, -1, 2), axis=(0, 1, 2))

    rigid = SimpleITKRegistrar().register_complex(
        fixed, None, moving, None, _volume_config("rigid")
    )
    affine = SimpleITKRegistrar().register_complex(
        fixed, None, moving, None, _volume_config("affine")
    )

    assert rigid.succeeded
    assert affine.succeeded
    assert rigid.warped_real.shape == fixed.shape
    assert affine.warped_real.shape == fixed.shape
    assert "Euler3D" in rigid.transform.GetName()
    assert "Affine" in affine.transform.GetName()


def test_unknown_transform_kind_is_rejected_at_construction():
    with pytest.raises(ValueError, match="transform_kind"):
        RegistrationConfig(transform_kind="tps")


def test_liver_preset_is_the_checkbox_configuration():
    config = default_registration_config()

    assert config is not None
    assert config.transform_kind == "rigid_bspline"
    assert config.metric == "ants_neighborhood_correlation"
    assert config.neighborhood_radius == 4
    assert config.mask_kind == "otsu_dilated"
    assert config.dilation_mm == 12.0
    assert config.bspline_grid_spacing_mm == 40.0
    # The no-argument config stays the historical single stage.
    assert RegistrationConfig().transform_kind == "bspline"
    assert RegistrationConfig().metric == "mattes_mi"
    assert RegistrationConfig().mask_kind == "none"


def test_otsu_mask_drops_distant_air_and_dilates():
    image = np.zeros((40, 40))
    image[10:16, 10:16] = 8.0
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image.SetSpacing((2.0, 2.0))
    mask = sitk.GetArrayFromImage(
        SimpleITKRegistrar()._otsu_dilated_mask(sitk_image, radius_mm=4.0)
    )

    assert mask[12, 12] == 1
    assert mask[9, 12] == 1  # one 2 mm voxel outside, inside the 4 mm rim
    assert mask[0, 0] == 0


def test_cascade_recovers_a_translation():
    fixed = _phantom((40, 40))
    moving = _resample_euler(fixed, translation=(4.0, -3.0))
    before = _mae(fixed, moving)
    config = RegistrationConfig(
        random_seed=1234,
        iterations=20,
        shrink_factors=(2, 1),
        smoothing_sigmas=(0.0, 0.0),
        sampling_percentage=0.4,
        metric="mattes_mi",
        transform_kind="rigid_bspline",
        bspline_grid_spacing_mm=20.0,
    )

    result = SimpleITKRegistrar().register_complex(
        fixed, None, moving, None, config, spacing_mm=(1.0, 1.0)
    )

    assert result.succeeded
    assert "Composite" in result.transform.GetName()
    assert _mae(fixed, result.warped_real) < before


def test_local_correlation_rigid_runs():
    fixed = _phantom((32, 32))
    moving = _resample_euler(fixed, translation=(2.0, -1.0))

    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(
            metric="ants_neighborhood_correlation",
            neighborhood_radius=2,
            transform_kind="rigid",
            iterations=15,
        ),
    )

    assert result.succeeded
    assert result.metric_name == "ANTSNeighborhoodCorrelation"
    assert _mae(fixed, result.warped_real) < _mae(fixed, moving)


def test_foreground_percentile_is_rejected_outside_0_to_100():
    with pytest.raises(ValueError, match="foreground_percentile"):
        RegistrationConfig(foreground_percentile=120)


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


def test_nonunit_spacing_does_not_fail_the_overlap_check():
    fixed = _phantom((32, 32))
    moving = np.roll(fixed, (2, -1), axis=(0, 1))

    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(iterations=3),
        spacing_mm=(1.5, 1.5),
    )

    assert result.status != "failed"
    assert np.isfinite(result.dice)
    assert "physical space" not in result.message


def test_dicom_spacing_is_stored_as_x_then_y():
    from mvc.models.thermometry_model import ThermometryModel

    model = ThermometryModel.__new__(ThermometryModel)
    model.pixel_spacing_mm = None

    class Header:
        PixelSpacing = [1.5, 2.0]

    model._remember_pixel_spacing(Header())

    assert model.pixel_spacing_mm == (2.0, 1.5)


def test_mind_is_unchanged_by_a_global_intensity_shift():
    image = _phantom()
    descriptor = mind_descriptor(image)

    assert descriptor.shape == image.shape + (8,)
    np.testing.assert_allclose(descriptor.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(mind_descriptor(image + 7.5), descriptor, atol=1e-8)
    assert mind_distance(image, image + 7.5) == pytest.approx(0.0, abs=1e-8)
    assert mind_distance(image, image) == pytest.approx(0.0, abs=1e-8)

    from mvc.models.inr_registration import mind_descriptor as inr_mind

    # Same stencil as the INR descriptor.  The INR optimizer is not involved.
    np.testing.assert_allclose(descriptor, inr_mind(image), atol=1e-8)


def test_mind_distance_drops_after_undoing_a_known_translation():
    fixed = _phantom()
    moving = _resample_euler(fixed, translation=(4.0, -3.0))
    misaligned = mind_distance(fixed, moving)
    # The resample that built ``moving`` used +translation, so the inverse
    # shift is the negation.  Interpolation leaves a small rim, not equality.
    realigned = _resample_euler(moving, translation=(-4.0, 3.0))

    assert mind_distance(fixed, moving + 12.0) == pytest.approx(misaligned, abs=1e-6)
    assert mind_distance(fixed, realigned) < misaligned * 0.5


def test_mind_defaults_are_opt_in_and_do_not_change_the_other_metrics():
    historical = RegistrationConfig()
    checkbox = default_registration_config()

    assert historical.metric == "mattes_mi"
    assert historical.mind_radius == MIND_RADIUS == 1
    assert historical.mind_sample_stride == MIND_SAMPLE_STRIDE == 2
    assert checkbox.metric == "ants_neighborhood_correlation"
    assert checkbox.transform_kind == "rigid_bspline"
    assert checkbox.mask_kind == "otsu_dilated"
    with pytest.raises(ValueError, match="mind_radius"):
        RegistrationConfig(mind_radius=0)
    with pytest.raises(ValueError, match="mind_radius"):
        RegistrationConfig(mind_radius=1.9)
    with pytest.raises(ValueError, match="mind_sample_stride"):
        RegistrationConfig(mind_sample_stride=0)

    fixed = _phantom((32, 32))
    moving = np.roll(fixed, (1, -1), axis=(0, 1))
    plain = SimpleITKRegistrar().register_complex(
        fixed, None, moving, None, _config(iterations=4)
    )
    retuned = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(iterations=4, mind_radius=3, mind_sample_stride=5),
    )

    assert plain.metric_name == "MattesMutualInformation"
    assert retuned.metric_name == "MattesMutualInformation"
    assert "MIND" not in plain.stop_condition
    np.testing.assert_allclose(plain.warped_real, retuned.warped_real, atol=1e-6)

    from dataclasses import replace

    checkbox_config = replace(
        checkbox,
        random_seed=1234,
        iterations=4,
        shrink_factors=(1,),
        smoothing_sigmas=(0.0,),
    )
    checkbox_plain = SimpleITKRegistrar().register_complex(
        fixed, None, moving, None, checkbox_config
    )
    checkbox_retuned = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        replace(checkbox_config, mind_radius=3, mind_sample_stride=5),
    )

    assert checkbox_plain.succeeded
    assert checkbox_retuned.succeeded
    assert checkbox_plain.metric_name == "ANTSNeighborhoodCorrelation"
    assert checkbox_retuned.metric_name == "ANTSNeighborhoodCorrelation"
    np.testing.assert_allclose(
        checkbox_plain.warped_real, checkbox_retuned.warped_real, atol=1e-6
    )


def _resample_with_spacing(image, spacing, translation):
    source = sitk.GetImageFromArray(np.asarray(image, dtype=np.float64))
    source.SetSpacing(tuple(float(value) for value in spacing))
    transform = sitk.Euler2DTransform()
    center_index = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
    transform.SetCenter(
        (center_index[0] * spacing[0], center_index[1] * spacing[1])
    )
    transform.SetTranslation(tuple(float(value) for value in translation))
    moved = sitk.Resample(source, source, transform, sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(moved)


def test_mind_registration_recovers_a_translation_under_an_intensity_shift():
    fixed = _phantom((40, 40))
    spacing = (1.5, 2.0)
    moving = _resample_with_spacing(fixed, spacing, translation=(3.0, -2.0))
    # A global offset must not change the MIND objective.  Sum of squares
    # still has a spatial minimum here; the descriptor test above is what
    # locks the invariance.  This run checks the registrar actually minimizes
    # that descriptor on an anisotropic grid.
    shifted = moving + 20.0
    before = mind_distance(fixed, shifted)

    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        shifted,
        None,
        _config(
            metric="mind",
            transform_kind="rigid",
            iterations=25,
            shrink_factors=(1,),
            smoothing_sigmas=(0.0,),
        ),
        spacing_mm=spacing,
    )

    assert result.succeeded
    assert result.metric_name == "MIND"
    # final_metric is the descriptor distance, so the +20 offset is invisible
    # to it.  Recomputing MIND on warped_real is not: resample fills outside
    # samples with 0, and that edge is not a global shift.
    assert result.final_metric < before * 0.5
    translation = result.transform.GetTranslation()
    assert translation[0] == pytest.approx(-3.0, abs=0.5)
    assert translation[1] == pytest.approx(2.0, abs=0.5)
    assert abs(np.degrees(result.transform.GetAngle())) < 1.0
    np.testing.assert_array_equal(shifted, moving + 20.0)


def test_mind_bspline_mask_and_cascade_reduce_the_distance():
    fixed = np.zeros((36, 36))
    yy, xx = np.mgrid[0:36, 0:36]
    fixed += np.exp(-((yy - 14.0) ** 2 + (xx - 16.0) ** 2) / 18.0)
    fixed += 0.7 * np.exp(-((yy - 24.0) ** 2 + (xx - 22.0) ** 2) / 12.0)
    moving = _resample_euler(fixed, translation=(3.0, -2.0))
    before = mind_distance(fixed, moving)

    bspline = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(
            metric="mind",
            transform_kind="bspline",
            iterations=15,
            mesh_size=4,
            shrink_factors=(1,),
            smoothing_sigmas=(0.0,),
        ),
    )
    masked = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(
            metric="mind",
            transform_kind="rigid",
            iterations=20,
            mask_kind="otsu_dilated",
            dilation_mm=4.0,
            shrink_factors=(1,),
            smoothing_sigmas=(0.0,),
        ),
    )

    assert bspline.succeeded
    assert bspline.metric_name == "MIND"
    assert "BSpline" in bspline.transform.GetName()
    assert mind_distance(fixed, bspline.warped_real) < before * 0.5
    assert masked.succeeded
    assert masked.metric_name == "MIND"
    assert mind_distance(fixed, masked.warped_real) < before * 0.5

    cascade = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(
            metric="mind",
            transform_kind="rigid_bspline",
            iterations=12,
            mesh_size=4,
            shrink_factors=(1,),
            smoothing_sigmas=(0.0,),
        ),
    )
    assert cascade.succeeded
    assert cascade.metric_name == "MIND"
    assert "Composite" in cascade.transform.GetName()
    assert mind_distance(fixed, cascade.warped_real) < before * 0.5


def test_mind_rigid_runs_in_3d():
    size = 14
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    fixed = np.exp(-((z - 6.0) ** 2 + (y - 7.0) ** 2 + (x - 5.0) ** 2) / 8.0)
    source = sitk.GetImageFromArray(fixed.astype(np.float64))
    transform = sitk.Euler3DTransform()
    transform.SetCenter(((size - 1) / 2.0,) * 3)
    transform.SetTranslation((1.5, -1.0, 0.5))
    moving = sitk.GetArrayFromImage(
        sitk.Resample(source, source, transform, sitk.sitkLinear, 0.0)
    )
    before = mind_distance(fixed, moving)

    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        RegistrationConfig(
            metric="mind",
            transform_kind="rigid",
            random_seed=1234,
            iterations=15,
            shrink_factors=(1,),
            smoothing_sigmas=(0.0,),
            sampling_percentage=0.2,
        ),
    )

    assert result.succeeded
    assert result.metric_name == "MIND"
    assert "Euler3D" in result.transform.GetName()
    assert mind_distance(fixed, result.warped_real) < before
    translation = result.transform.GetTranslation()
    assert translation[0] == pytest.approx(-1.5, abs=0.75)
    assert translation[1] == pytest.approx(1.0, abs=0.75)
    assert translation[2] == pytest.approx(-0.5, abs=0.75)


def _analytical_transform_jacobian(transform, point):
    dimension = transform.GetDimension()
    points = np.asarray(point, dtype=float).reshape(1, dimension)
    jacobian = np.zeros((dimension, transform.GetNumberOfParameters()))
    for axis in range(dimension):
        upstream = np.zeros((1, dimension))
        upstream[0, axis] = 1.0
        jacobian[axis] = _parameter_gradient(transform, points, upstream)
    return jacobian


def _finite_difference_transform_jacobian(transform, point, step=1e-6):
    dimension = transform.GetDimension()
    base = np.array(transform.GetParameters(), dtype=float)
    jacobian = np.zeros((dimension, base.size))
    location = tuple(float(value) for value in point)
    for index in range(base.size):
        plus = base.copy()
        minus = base.copy()
        plus[index] += step
        minus[index] -= step
        transform.SetParameters(tuple(float(value) for value in plus))
        forward = np.array(transform.TransformPoint(location))
        transform.SetParameters(tuple(float(value) for value in minus))
        backward = np.array(transform.TransformPoint(location))
        jacobian[:, index] = (forward - backward) / (2.0 * step)
    transform.SetParameters(tuple(float(value) for value in base))
    return jacobian


def test_mind_transform_jacobians_match_finite_differences():
    euler = sitk.Euler2DTransform()
    euler.SetCenter((8.0, -3.0))
    euler.SetParameters((0.2, -1.5, 0.75))
    affine = sitk.AffineTransform(3)
    affine.SetCenter((1.0, -2.0, 0.5))
    matrix = np.eye(3) + np.array(
        [[0.1, -0.2, 0.0], [0.0, 0.05, 0.1], [-0.1, 0.0, 0.0]]
    )
    affine.SetParameters(tuple(matrix.ravel().tolist()) + (0.4, -0.3, 0.2))
    euler3 = sitk.Euler3DTransform()
    euler3.SetCenter((1.5, -2.0, 4.0))
    euler3.SetParameters((0.2, -0.15, 0.05, 0.3, -0.4, 0.2))
    image = sitk.GetImageFromArray(np.zeros((24, 24), dtype=np.float64))
    bspline = sitk.BSplineTransformInitializer(image, [3, 3])
    rng = np.random.default_rng(1)
    bspline.SetParameters(
        tuple(
            float(value)
            for value in rng.normal(
                scale=0.2, size=bspline.GetNumberOfParameters()
            )
        )
    )

    cases = (
        (euler, (4.0, 9.0)),
        (affine, (2.0, 0.5, -1.0)),
        (euler3, (0.2, 1.0, 3.0)),
        (bspline, (8.0, 12.5)),
    )
    for transform, point in cases:
        analytical = _analytical_transform_jacobian(transform, point)
        finite = _finite_difference_transform_jacobian(transform, point)
        np.testing.assert_allclose(analytical, finite, atol=1e-6, rtol=1e-5)

    outside = _analytical_transform_jacobian(bspline, (-30.0, -30.0))
    assert np.all(outside == 0.0)


def test_mind_chain_rule_matches_rigid_of_bspline():
    """The free transform is the inner map, matching the composite order."""

    image = sitk.GetImageFromArray(np.zeros((32, 32), dtype=np.float64))
    rigid = sitk.Euler2DTransform()
    rigid.SetCenter((15.5, 15.5))
    rigid.SetParameters((0.3, 2.0, -1.0))
    bspline = sitk.BSplineTransformInitializer(image, [3, 3])
    base = np.random.default_rng(0).normal(
        scale=0.3, size=bspline.GetNumberOfParameters()
    )
    bspline.SetParameters(tuple(float(value) for value in base))
    fixed_point = np.array([[8.0, 12.0]])
    inner = np.array([bspline.TransformPoint((8.0, 12.0))])
    analytical = np.zeros((2, base.size))
    for axis in range(2):
        upstream = np.zeros((1, 2))
        upstream[0, axis] = 1.0
        pulled = _pull_through_outer(rigid, inner, upstream)
        analytical[axis] = _parameter_gradient(bspline, fixed_point, pulled)

    def composed(theta):
        bspline.SetParameters(tuple(float(value) for value in theta))
        mapped = bspline.TransformPoint((8.0, 12.0))
        return np.array(rigid.TransformPoint(mapped))

    step = 1e-5
    finite = np.zeros_like(analytical)
    for index in range(base.size):
        plus = np.array(base, copy=True)
        minus = np.array(base, copy=True)
        plus[index] += step
        minus[index] -= step
        finite[:, index] = (composed(plus) - composed(minus)) / (2.0 * step)
    bspline.SetParameters(tuple(float(value) for value in base))
    np.testing.assert_allclose(analytical, finite, atol=1e-5, rtol=1e-4)


def test_mind_cascade_matches_the_resampled_order_when_bspline_moves():
    fixed = _phantom((48, 48))
    source = sitk.GetImageFromArray(np.asarray(fixed, dtype=np.float64))
    center = ((fixed.shape[1] - 1) / 2.0, (fixed.shape[0] - 1) / 2.0)
    scale = sitk.AffineTransform(2)
    scale.SetCenter(center)
    scale.SetParameters((1.12, 0.0, 0.0, 1.12, 2.5, -1.5))
    moving_image = sitk.Resample(source, source, scale, sitk.sitkLinear, 0.0)
    moving = sitk.GetArrayFromImage(moving_image)
    before = mind_distance(fixed, moving)

    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(
            metric="mind",
            transform_kind="rigid_bspline",
            iterations=20,
            mesh_size=5,
            shrink_factors=(1,),
            smoothing_sigmas=(0.0,),
        ),
    )

    assert result.succeeded
    assert result.metric_name == "MIND"
    assert mind_distance(fixed, result.warped_real) < before * 0.5
    rigid = result.transform.GetNthTransform(0)
    bspline = result.transform.GetNthTransform(1)
    assert np.max(np.abs(np.array(bspline.GetParameters()))) > 0.05
    swapped = sitk.CompositeTransform(2)
    swapped.AddTransform(bspline)
    swapped.AddTransform(rigid)
    fixed_image = sitk.GetImageFromArray(np.asarray(fixed, dtype=np.float64))
    moving_image = sitk.GetImageFromArray(np.asarray(moving, dtype=np.float64))
    swapped_warp = sitk.GetArrayFromImage(
        sitk.Resample(moving_image, fixed_image, swapped, sitk.sitkLinear, 0.0)
    )
    # The returned composite is rigid(bspline).  Optimizing the other order
    # would make this swapped warp the better one.
    assert mind_distance(fixed, result.warped_real) <= (
        mind_distance(fixed, swapped_warp) + 1e-4
    )


def test_mind_rigid_does_not_run_away_on_a_larger_shift():
    fixed = _phantom((40, 42))
    spacing = (1.5, 2.5)
    moving = _resample_with_spacing(fixed, spacing, translation=(10.0, -6.0))

    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        _config(
            metric="mind",
            transform_kind="rigid",
            iterations=40,
            shrink_factors=(1,),
            smoothing_sigmas=(0.0,),
        ),
        spacing_mm=spacing,
    )

    assert result.succeeded
    translation = result.transform.GetTranslation()
    assert translation[0] == pytest.approx(-10.0, abs=1.5)
    assert translation[1] == pytest.approx(6.0, abs=1.5)
    assert abs(translation[0]) < 40.0
    assert abs(translation[1]) < 40.0


def test_index_to_physical_matches_simpleitk_with_direction():
    image = sitk.GetImageFromArray(np.zeros((6, 5, 4), dtype=np.float64))
    image.SetSpacing((1.5, 2.0, 2.5))
    image.SetOrigin((-3.0, 4.0, 0.5))
    image.SetDirection((0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    index = np.array([[1.2, 2.5, 3.1], [0.0, 0.0, 0.0], [4.0, 3.0, 2.0]])
    physical = _index_to_physical(index, image)
    _, du_dp = _geometry(image)

    for point, continuous in zip(physical, index, strict=True):
        expected = image.TransformContinuousIndexToPhysicalPoint(
            tuple(float(value) for value in continuous)
        )
        np.testing.assert_allclose(point, expected, atol=1e-12)
    np.testing.assert_allclose(
        _physical_to_index(physical, du_dp, image), index, atol=1e-12
    )
