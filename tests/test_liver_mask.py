"""Opt-in coarse liver ROI.  The Otsu checkbox path is not this mask."""

import numpy as np
import pytest
import SimpleITK as sitk

from mvc.controllers.advanced_features_controller import AdvancedFeaturesController
from mvc.models.liver_mask import coarse_liver_mask
from mvc.models.registration import (
    RegistrationConfig,
    SimpleITKRegistrar,
    coarse_liver_requested,
    default_registration_config,
    with_coarse_liver_mask,
)


def _image(array, spacing):
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float64))
    image.SetSpacing(tuple(float(value) for value in spacing))
    return image


def _mask(array, spacing=(1.0, 1.0), wall_mm=20.0):
    image = _image(array, spacing)
    mask = coarse_liver_mask(image, wall_mm=wall_mm)
    return sitk.GetArrayFromImage(mask)


def _wall_liver_phantom():
    """Air, a bright wall, a dimmer liver, a rim spleen, and a far speck.

    Otsu on this histogram keeps the wall.  The coarse ROI should not.
    """
    image = np.zeros((100, 100), dtype=float)
    image[15:85, 15:85] = 35.0
    image[15:27, 15:85] = 150.0
    image[73:85, 15:85] = 150.0
    image[15:85, 15:27] = 150.0
    image[15:85, 73:85] = 150.0
    image[48:52, 46:50] = 0.0  # vessel inside the liver
    image[18:28, 40:52] = 180.0  # spleen sitting in the wall
    image[2, 2] = 255.0
    return image


def test_coarse_liver_drops_the_wall_that_otsu_keeps():
    phantom = _wall_liver_phantom()
    mask = _mask(phantom, wall_mm=20.0)
    otsu = sitk.GetArrayFromImage(
        sitk.OtsuThreshold(_image(phantom, (1.0, 1.0)), 0, 1)
    )

    # The documented Otsu failure: the bright wall is foreground.
    assert otsu[20, 60] == 1
    assert mask[20, 60] == 0  # wall
    assert mask[22, 46] == 0  # spleen in the rim
    assert mask[50, 55] == 1  # liver core
    assert mask[50, 48] == 1  # enclosed vessel is filled
    assert mask[2, 2] == 0  # distant speck
    assert mask[0, 50] == 0  # air
    assert int(mask.sum()) < int(otsu.sum())


def test_global_intensity_shift_does_not_change_the_coarse_mask():
    phantom = _wall_liver_phantom()

    np.testing.assert_array_equal(_mask(phantom), _mask(phantom + 17.0))


def test_thin_object_keeps_the_body_instead_of_an_empty_mask():
    image = np.zeros((40, 40), dtype=float)
    image[16:26, 16:26] = 50.0
    image[1, 1] = 200.0

    mask = _mask(image, wall_mm=20.0)

    assert mask[20, 20] == 1
    assert mask[16, 16] == 1
    assert mask[15, 20] == 0
    assert mask[1, 1] == 0
    assert int(mask.sum()) == 100


def test_flat_image_has_no_tissue_split():
    # Corners match the interior, so the air estimate is refused.  Otsu on
    # one grey level marks nothing as the high class.  There is no liver
    # to invent, and an empty mask is reported instead of a full one.
    image = np.full((40, 40), 10.0)

    with pytest.raises(ValueError, match="no tissue"):
        _mask(image, wall_mm=4.0)


def test_bright_corners_do_not_raise_and_keep_an_interior():
    # Corners are brighter than the center, so the air estimate must not run.
    # A corner threshold would see no voxel above the corner median.
    image = np.full((60, 60), 100.0)
    image[20:40, 20:40] = 10.0

    mask = _mask(image, wall_mm=6.0)

    assert mask[30, 30] == 1
    assert mask[0, 0] == 0


def test_non_finite_image_is_rejected():
    image = np.full((16, 16), np.nan)

    with pytest.raises(ValueError, match="finite"):
        _mask(image)


def test_non_finite_background_stays_out_and_an_enclosed_nan_is_filled():
    phantom = _wall_liver_phantom()
    phantom[0, 50] = np.nan
    phantom[50, 55] = np.nan

    mask = _mask(phantom)

    assert mask[0, 50] == 0
    # The liver voxel was set below the air threshold, then filled because
    # tissue encloses it.  That is the same rule as the vessel hole.
    assert mask[50, 55] == 1
    assert mask[50, 56] == 1


def test_three_dimensions_erode_in_physical_units():
    image = np.zeros((36, 36, 36), dtype=float)
    image[8:28, 8:28, 8:28] = 40.0

    mask = _mask(image, spacing=(1.0, 1.0, 1.0), wall_mm=6.0)

    assert mask[0, 0, 0] == 0
    assert mask[9, 18, 18] == 0
    assert mask[18, 18, 18] == 1


def test_anisotropic_spacing_erodes_each_axis_in_millimetres():
    image = np.zeros((40, 30), dtype=float)
    image[5:35, 5:25] = 40.0
    # spacing is (x, y) = (2 mm, 1 mm).  An 8 mm wall is 4 voxels in x
    # and 8 voxels in y.  Array axes are (y, x).
    mask = _mask(image, spacing=(2.0, 1.0), wall_mm=8.0)

    assert mask[6, 6] == 0
    assert mask[20, 15] == 1
    assert mask[0, 0] == 0


def test_non_positive_spacing_is_rejected():
    # SimpleITK refuses to store a zero spacing, so the guard is exercised
    # on the radius helper.  A zero step would make the wall an infinite
    # number of voxels.
    from mvc.models.liver_mask import _voxel_radii

    class _Image:
        def GetSpacing(self):
            return (0.0, 1.0)

    with pytest.raises(ValueError, match="spacing"):
        _voxel_radii(_Image(), 4.0, minimum=1)


def test_defaults_do_not_select_the_coarse_liver_mask():
    assert default_registration_config().mask_kind == "otsu_dilated"
    assert RegistrationConfig().mask_kind == "none"
    assert coarse_liver_requested(object()) is False


def test_with_coarse_liver_mask_replaces_only_the_mask():
    preset = RegistrationConfig.liver_respiratory()

    assert with_coarse_liver_mask(preset, False) is preset
    assert with_coarse_liver_mask(None, False) is None

    swapped = with_coarse_liver_mask(preset, True)
    assert swapped is not preset
    assert preset.mask_kind == "otsu_dilated"
    assert swapped.mask_kind == "coarse_liver"
    assert swapped.metric == preset.metric
    assert swapped.transform_kind == preset.transform_kind
    assert swapped.bspline_grid_spacing_mm == 40.0

    again = with_coarse_liver_mask(swapped, True)
    assert again is swapped


def test_missing_config_fallback_is_explicit():
    preset = RegistrationConfig.liver_respiratory()
    swapped = with_coarse_liver_mask(None, True, fallback=preset)

    assert preset.mask_kind == "otsu_dilated"
    assert swapped.mask_kind == "coarse_liver"
    assert swapped.metric == "ants_neighborhood_correlation"

    historical = with_coarse_liver_mask(None, True, fallback=RegistrationConfig())
    assert historical.metric == "mattes_mi"
    assert historical.transform_kind == "bspline"
    assert historical.mask_kind == "coarse_liver"


def test_liver_wall_mm_must_be_a_positive_finite_number():
    with pytest.raises(TypeError, match="liver_wall_mm"):
        RegistrationConfig(liver_wall_mm=True)
    with pytest.raises(ValueError, match="liver_wall_mm"):
        RegistrationConfig(liver_wall_mm=0)
    with pytest.raises(ValueError, match="liver_wall_mm"):
        RegistrationConfig(liver_wall_mm=float("nan"))
    with pytest.raises(ValueError, match="mask_kind"):
        RegistrationConfig(mask_kind="liver")


def test_registrar_uses_the_coarse_mask_and_ignores_the_knob_otherwise():
    phantom = _wall_liver_phantom()
    fixed = _image(phantom, (1.0, 1.0))
    moving = _image(phantom, (1.0, 1.0))
    registrar = SimpleITKRegistrar()

    fixed_mask, moving_mask = registrar._metric_masks(
        fixed,
        moving,
        RegistrationConfig(mask_kind="coarse_liver", liver_wall_mm=20.0),
    )
    mask = sitk.GetArrayFromImage(fixed_mask)
    assert mask[20, 60] == 0
    assert mask[50, 55] == 1
    assert sitk.GetArrayFromImage(moving_mask)[50, 55] == 1

    # foreground_mask must not override an explicit coarse liver mask.
    overridden, _ = registrar._metric_masks(
        fixed,
        moving,
        RegistrationConfig(mask_kind="coarse_liver", foreground_mask=True),
    )
    assert sitk.GetArrayFromImage(overridden)[20, 60] == 0


def test_liver_wall_knob_does_not_change_unmasked_metric_masks():
    fixed = np.zeros((32, 32), dtype=float)
    fixed[8:24, 8:24] = 30.0
    fixed[12:20, 14:22] = 80.0
    moving = np.roll(fixed, (2, -1), axis=(0, 1))
    plain = RegistrationConfig(
        random_seed=1234,
        iterations=4,
        shrink_factors=(2, 1),
        smoothing_sigmas=(0.0, 0.0),
        mask_kind="none",
        liver_wall_mm=20.0,
    )
    retuned = RegistrationConfig(
        random_seed=1234,
        iterations=4,
        shrink_factors=(2, 1),
        smoothing_sigmas=(0.0, 0.0),
        mask_kind="none",
        liver_wall_mm=6.0,
    )

    fixed_image = _image(fixed, (1.0, 1.0))
    moving_image = _image(moving, (1.0, 1.0))
    registrar = SimpleITKRegistrar()

    assert registrar._metric_masks(fixed_image, moving_image, plain) == (None, None)
    assert registrar._metric_masks(fixed_image, moving_image, retuned) == (None, None)


def test_coarse_liver_registration_recovers_a_translation():
    fixed = _wall_liver_phantom()
    moving = np.roll(fixed, (2, -1), axis=(0, 1))
    before = float(np.mean(np.abs(fixed - moving)))

    result = SimpleITKRegistrar().register_complex(
        fixed,
        None,
        moving,
        None,
        RegistrationConfig(
            random_seed=1234,
            iterations=8,
            shrink_factors=(2, 1),
            smoothing_sigmas=(0.0, 0.0),
            sampling_percentage=0.4,
            transform_kind="rigid",
            mask_kind="coarse_liver",
            liver_wall_mm=8.0,
        ),
        spacing_mm=(1.0, 1.0),
    )

    assert result.succeeded
    assert result.config.mask_kind == "coarse_liver"
    assert float(np.mean(np.abs(fixed - result.warped_real))) < before


class _Box:
    def __init__(self, checked):
        self.checked = checked

    def isChecked(self):
        return self.checked


class _View:
    def __init__(self, checked):
        self.coarse_liver_checkbox = _Box(checked)


def _cinema_controller(checked, config, registrar):
    controller = AdvancedFeaturesController.__new__(AdvancedFeaturesController)
    controller.view = _View(checked)
    controller.registration_config = config
    controller.registrar = registrar
    return controller


def test_cinema_checkbox_off_preserves_a_missing_config():
    controller = _cinema_controller(False, None, None)

    assert controller._registration_config_for_run() is None


def test_cinema_checkbox_on_keeps_the_preset_and_swaps_the_mask():
    stored = RegistrationConfig.liver_respiratory()
    controller = _cinema_controller(True, stored, object())

    config = controller._registration_config_for_run()

    assert stored.mask_kind == "otsu_dilated"
    assert config.mask_kind == "coarse_liver"
    assert config.metric == "ants_neighborhood_correlation"
    assert config.transform_kind == "rigid_bspline"
    assert config.bspline_grid_spacing_mm == 40.0


def test_cinema_checkbox_on_without_a_config_uses_the_matching_fallback():
    no_registrar = _cinema_controller(True, None, None)
    preset = no_registrar._registration_config_for_run()
    assert preset.mask_kind == "coarse_liver"
    assert preset.metric == "ants_neighborhood_correlation"
    assert preset.transform_kind == "rigid_bspline"

    with_registrar = _cinema_controller(True, None, object())
    historical = with_registrar._registration_config_for_run()
    assert historical.mask_kind == "coarse_liver"
    assert historical.metric == "mattes_mi"
    assert historical.transform_kind == "bspline"

