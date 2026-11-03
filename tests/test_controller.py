"""Controller integration tests: registration result handling and fallbacks."""

import numpy as np

from mvc.controllers.thermometry_controller import ThermometryController
from mvc.models.registration import RegistrationConfig, RegistrationResult


class FakeView:
    def __init__(self):
        self.statuses = []
        self.errors = []
        self.temperature_updates = []

    def update_status(self, message):
        self.statuses.append(message)

    def show_error(self, message):
        self.errors.append(message)

    def update_temperature_display(self, temperature_map):
        self.temperature_updates.append(temperature_map)


class FakeModel:
    def __init__(self):
        self.real1 = np.ones((8, 8))
        self.imag1 = np.zeros((8, 8))
        self.real2 = np.full((8, 8), 2.0)
        self.imag2 = np.full((8, 8), 3.0)
        self.temperature_map = np.zeros((8, 8))
        self.registration_result = None
        self.calc_kwargs = None
        self.calculate_ok = True

    def calculate_temperature(self, **kwargs):
        self.calc_kwargs = kwargs
        return self.calculate_ok


class FakeRegistrar:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def register_complex(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _result(dice=0.95, status="ok", warped_real=None, warped_imag=None):
    shape = (8, 8)
    return RegistrationResult(
        warped_real=np.full(shape, 4.0) if warped_real is None else warped_real,
        warped_imag=np.full(shape, 5.0) if warped_imag is None else warped_imag,
        transform=object(),
        config=RegistrationConfig(),
        metric_name="MattesMutualInformation",
        final_metric=-1.0,
        dice=dice,
        iterations=5,
        stop_condition="converged",
        elapsed_s=0.01,
        status=status,
    )


def _controller(result):
    model = FakeModel()
    view = FakeView()
    registrar = FakeRegistrar(result)
    controller = ThermometryController.__new__(ThermometryController)
    controller.model = model
    controller.view = view
    controller.registrar = registrar
    controller.registration_config = RegistrationConfig()
    return controller, model, view, registrar


def test_apply_motion_correction_stores_result_without_mutating_originals():
    result = _result(dice=0.95)
    controller, model, view, registrar = _controller(result)

    assert controller.apply_motion_correction() is True

    assert model.registration_result is result
    np.testing.assert_array_equal(model.real2, np.full((8, 8), 2.0))
    np.testing.assert_array_equal(model.imag2, np.full((8, 8), 3.0))
    assert registrar.calls[0]["fixed_real"] is model.real1
    assert registrar.calls[0]["moving_real"] is model.real2
    assert registrar.calls[0]["moving_imag"] is model.imag2
    assert "Motion correction applied" in view.statuses[-1]
    assert view.errors == []


def test_repeated_correction_uses_the_original_images_each_time():
    result = _result()
    controller, model, _, registrar = _controller(result)

    assert controller.apply_motion_correction() is True
    assert controller.apply_motion_correction() is True

    assert len(registrar.calls) == 2
    assert registrar.calls[0]["moving_real"] is model.real2
    assert registrar.calls[1]["moving_real"] is model.real2
    np.testing.assert_array_equal(model.real2, np.full((8, 8), 2.0))
    np.testing.assert_array_equal(model.imag2, np.full((8, 8), 3.0))


def test_apply_motion_correction_reports_warning():
    controller, _, view, _ = _controller(_result(dice=0.8, status="warning"))

    assert controller.apply_motion_correction() is True

    assert any("Warning: Registration quality" in item for item in view.statuses)


def test_apply_motion_correction_reports_failure():
    failure = RegistrationResult(
        warped_real=None,
        warped_imag=None,
        transform=None,
        config=RegistrationConfig(),
        metric_name="MattesMutualInformation",
        final_metric=float("nan"),
        dice=float("nan"),
        iterations=0,
        stop_condition="",
        elapsed_s=0.0,
        status="failed",
        message="boom",
    )
    controller, model, view, _ = _controller(failure)

    assert controller.apply_motion_correction() is False
    assert model.registration_result is None
    assert any("boom" in item for item in view.errors)


def test_failed_registration_clears_a_stale_correction():
    failure = RegistrationResult(
        warped_real=None,
        warped_imag=None,
        transform=None,
        config=RegistrationConfig(),
        metric_name="MattesMutualInformation",
        final_metric=float("nan"),
        dice=float("nan"),
        iterations=0,
        stop_condition="",
        elapsed_s=0.0,
        status="failed",
        message="boom",
    )
    controller, model, _, _ = _controller(failure)
    model.registration_result = object()

    assert controller.apply_motion_correction() is False
    assert model.registration_result is None


def test_registrar_exception_is_handled():
    class RaisingRegistrar:
        def register_complex(self, **kwargs):
            raise ValueError("bad shapes")

    model = FakeModel()
    view = FakeView()
    controller = ThermometryController.__new__(ThermometryController)
    controller.model = model
    controller.view = view
    controller.registrar = RaisingRegistrar()
    controller.registration_config = RegistrationConfig()

    assert controller.apply_motion_correction() is False
    assert any("bad shapes" in item for item in view.errors)
    assert model.registration_result is None


def test_calculate_temperature_uses_registration_when_it_succeeds():
    controller, model, view, _ = _controller(_result())

    controller.calculate_temperature(37.0, False, False, True)

    assert model.calc_kwargs["use_registration"] is True
    assert model.calc_kwargs["original_temperature"] == 37.0
    assert view.temperature_updates
    assert view.errors == []


def test_calculate_temperature_falls_back_when_registration_fails():
    failure = RegistrationResult(
        warped_real=None,
        warped_imag=None,
        transform=None,
        config=RegistrationConfig(),
        metric_name="MattesMutualInformation",
        final_metric=float("nan"),
        dice=float("nan"),
        iterations=0,
        stop_condition="",
        elapsed_s=0.0,
        status="failed",
        message="boom",
    )
    controller, model, view, _ = _controller(failure)

    controller.calculate_temperature(37.0, False, False, True)

    assert model.calc_kwargs["use_registration"] is False
    assert any("Proceeding without it" in item for item in view.errors)
    assert view.temperature_updates


def test_calculate_temperature_without_motion_correction_does_not_register():
    controller, model, view, registrar = _controller(_result())

    controller.calculate_temperature(37.0, False, False, False)

    assert registrar.calls == []
    assert model.calc_kwargs["use_registration"] is False
