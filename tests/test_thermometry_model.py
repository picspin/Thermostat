"""Model state tests for non-destructive motion correction."""

import numpy as np
import pytest

import mvc.models.thermometry_model as thermometry_model
from mvc.models.registration import RegistrationConfig, RegistrationResult
from mvc.models.thermometry_model import ThermometryModel


def _model(shape=(16, 16)):
    model = ThermometryModel()
    model.te = 10.0
    model.b0 = 3.0
    model.real1 = np.ones(shape)
    model.imag1 = np.zeros(shape)
    model.real2 = np.ones(shape)
    model.imag2 = np.zeros(shape)
    return model


def _result(shape=(16, 16), phase=0.3):
    return RegistrationResult(
        warped_real=np.full(shape, np.cos(phase)),
        warped_imag=np.full(shape, np.sin(phase)),
        transform=object(),
        config=RegistrationConfig(),
        metric_name="MattesMutualInformation",
        final_metric=-1.0,
        dice=1.0,
        iterations=1,
        stop_condition="converged",
        elapsed_s=0.0,
    )


def test_registered_copy_is_used_without_overwriting_originals():
    model = _model()
    model.registration_result = _result()

    assert model.calculate_temperature(37.0, use_registration=True) is True

    # phase(exp(i*0.3)) = 0.3 -> a positive temperature change.
    assert np.mean(model.temperature_change) > 0.0
    np.testing.assert_array_equal(model.real2, np.ones((16, 16)))
    np.testing.assert_array_equal(model.imag2, np.zeros((16, 16)))


def test_original_images_are_used_without_registration():
    model = _model()
    model.registration_result = _result()

    assert model.calculate_temperature(37.0, use_registration=False) is True

    np.testing.assert_allclose(model.temperature_change, 0.0)


def test_missing_registration_result_falls_back_to_originals():
    model = _model()

    assert model.calculate_temperature(37.0, use_registration=True) is True

    np.testing.assert_allclose(model.temperature_change, 0.0)


def test_repeated_calculation_with_registration_is_stable():
    model = _model()
    model.registration_result = _result()

    assert model.calculate_temperature(37.0, use_registration=True) is True
    first = model.temperature_map.copy()
    assert model.calculate_temperature(37.0, use_registration=True) is True

    np.testing.assert_allclose(model.temperature_map, first)


def test_clear_data_clears_registration():
    model = _model()
    model.registration_result = object()

    model.clear_data()

    assert model.registration_result is None
    assert model.real2 is None
    assert model.imag2 is None


class _FakeDataset:
    def __init__(self, array):
        self.pixel_array = array
        self.EchoTime = 10.0
        self.MagneticFieldStrength = 3.0
        self.FlipAngle = 90.0


@pytest.mark.parametrize(
    "method_name",
    ["load_real1_dicom", "load_imag1_dicom", "load_real2_dicom", "load_imag2_dicom"],
)
def test_loading_any_component_invalidates_registration(monkeypatch, method_name):
    model = _model()
    model.registration_result = object()
    monkeypatch.setattr(
        thermometry_model.pydicom,
        "dcmread",
        lambda path: _FakeDataset(np.ones((4, 4))),
    )

    assert getattr(model, method_name)("ignored.dcm") is True
    assert model.registration_result is None


def test_dead_model_side_registration_is_removed():
    model = _model()
    assert not hasattr(model, "apply_motion_correction")
