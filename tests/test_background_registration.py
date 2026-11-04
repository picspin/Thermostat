"""Background registration stays off the GUI thread and does not change a cancel."""

import threading
import time

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

from mvc.controllers.background_work import BackgroundCall
from mvc.controllers.thermometry_controller import ThermometryController
from mvc.models.registration import RegistrationConfig, RegistrationResult


class FakeView:
    def __init__(self):
        self.statuses = []
        self.errors = []
        self.temperature_updates = []
        self.status_timeouts = []

    def update_status(self, message, timeout_ms=5000):
        self.statuses.append(message)
        self.status_timeouts.append(timeout_ms)

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

    def calculate_temperature(self, **kwargs):
        self.calc_kwargs = kwargs
        return True


class ImmediateRegistrar:
    def __init__(self):
        self.calls = []

    def register_complex(self, **kwargs):
        self.calls.append(kwargs)
        return _result()


class BlockingRegistrar:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def register_complex(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        assert self.release.wait(3)
        return _result()


def _result():
    return RegistrationResult(
        warped_real=np.full((8, 8), 4.0),
        warped_imag=np.full((8, 8), 5.0),
        transform=object(),
        config=RegistrationConfig(),
        metric_name="MattesMutualInformation",
        final_metric=-0.2,
        dice=0.95,
        iterations=4,
        stop_condition="converged",
        elapsed_s=0.01,
    )


def _controller(registrar):
    controller = ThermometryController.__new__(ThermometryController)
    controller.model = FakeModel()
    controller.view = FakeView()
    controller.registrar = registrar
    controller.registration_config = RegistrationConfig()
    return controller


_QT_APP = None


def _app():
    """Keep a process-wide application.  A temporary one is destroyed at once."""
    global _QT_APP
    if QApplication.instance() is None:
        _QT_APP = QApplication([])
    elif _QT_APP is None:
        _QT_APP = QApplication.instance()
    return _QT_APP


def _pump(predicate, timeout=3.0):
    app = _app()
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_queued_signal_emitted_on_a_worker_runs_on_the_gui_thread():
    app = _app()
    seen = {}

    class Probe(QObject):
        ping = pyqtSignal(str)

    probe = Probe()
    probe.ping.connect(
        lambda text: seen.update(thread=threading.current_thread(), text=text)
    )

    def work():
        probe.ping.emit("from-worker")
        return "ok"

    job = BackgroundCall()
    job.start(
        work,
        lambda value: seen.update(done=True, value=value),
        lambda message: seen.update(done=True, error=message),
    )

    assert _pump(lambda: seen.get("done") is True)
    assert seen["value"] == "ok"
    assert seen["text"] == "from-worker"
    assert seen["thread"] is threading.main_thread()
    app.processEvents()


def test_background_registration_applies_the_same_accept_rule():
    _app()
    registrar = ImmediateRegistrar()
    controller = _controller(registrar)

    controller.begin_calculate_temperature(37.0, False, False, True)

    assert _pump(lambda: controller.model.calc_kwargs is not None)
    assert controller.model.calc_kwargs["use_registration"] is True
    assert controller.model.registration_result is not None
    assert registrar.calls[0]["moving_real"] is not controller.model.real2
    np.testing.assert_array_equal(
        registrar.calls[0]["moving_real"], np.full((8, 8), 2.0)
    )
    assert controller.view.temperature_updates
    assert controller.view.errors == []


def test_cancel_discards_the_background_registration():
    _app()
    registrar = BlockingRegistrar()
    controller = _controller(registrar)

    controller.begin_calculate_temperature(37.0, False, False, True)
    assert registrar.started.wait(3)
    controller._cancel_motion_correction()
    assert any("Cancelling" in item for item in controller.view.statuses)
    registrar.release.set()

    assert _pump(lambda: not controller._motion_job.running())
    assert controller.model.calc_kwargs is None
    assert controller.model.registration_result is None


def test_background_registration_is_discarded_when_the_images_change():
    _app()
    registrar = BlockingRegistrar()
    controller = _controller(registrar)

    controller.begin_calculate_temperature(37.0, False, False, True)
    assert registrar.started.wait(3)
    controller.model.real2 = np.zeros((8, 8))
    registrar.release.set()

    assert _pump(lambda: any("discarded" in item for item in controller.view.errors))
    assert controller.model.calc_kwargs is None
    assert controller.model.registration_result is None


def test_a_second_background_registration_is_refused_while_one_is_running():
    _app()
    registrar = BlockingRegistrar()
    controller = _controller(registrar)

    controller.begin_calculate_temperature(37.0, False, False, True)
    assert registrar.started.wait(3)
    controller.begin_calculate_temperature(37.0, False, False, True)
    assert any("already running" in item for item in controller.view.errors)
    registrar.release.set()
    assert _pump(lambda: controller.model.calc_kwargs is not None)


def test_a_stale_registration_callback_does_not_overwrite_a_newer_job():
    registrar = ImmediateRegistrar()
    controller = _controller(registrar)
    controller._motion_generation = 2
    controller._motion_cancelled = False
    controller.model.registration_result = "current"

    controller._finish_background_registration(
        _result(),
        None,
        37.0,
        False,
        False,
        controller._image_snapshot(),
        1,
    )

    assert controller.model.registration_result == "current"
    assert controller.model.calc_kwargs is None


def test_begin_without_motion_correction_stays_synchronous():
    registrar = ImmediateRegistrar()
    controller = _controller(registrar)

    controller.begin_calculate_temperature(36.5, True, False, False)

    assert registrar.calls == []
    assert controller.model.calc_kwargs["use_registration"] is False
    assert controller.model.calc_kwargs["original_temperature"] == 36.5
    assert controller.model.calc_kwargs["apply_gaussian_filter"] is True
