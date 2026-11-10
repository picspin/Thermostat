"""Sequence registration stays off unless it is explicitly requested."""

import threading

import numpy as np

from mvc.models.cinema_model import CinemaModel
from mvc.models.registration import RegistrationConfig, RegistrationResult


class RecordingModel:
    def __init__(self, fail_loads=()):
        self.fail_loads = set(fail_loads)
        self.loads = []
        self.real1 = np.ones((4, 4))
        self.imag1 = np.zeros((4, 4))
        self.real2 = np.full((4, 4), 2.0)
        self.imag2 = np.full((4, 4), 3.0)
        self.temperature_map = np.full((4, 4), 37.0)
        self.registration_result = None
        self.calls = []

    def _load(self, name, path):
        self.loads.append((name, path))
        self.registration_result = None
        if (name, path) in self.fail_loads:
            return False
        # A new array each time, so a later frame cannot alias an earlier one.
        setattr(self, name, np.full((4, 4), float(len(self.loads))))
        return True

    def load_real1_dicom(self, path):
        return self._load("real1", path)

    def load_imag1_dicom(self, path):
        return self._load("imag1", path)

    def load_real2_dicom(self, path):
        return self._load("real2", path)

    def load_imag2_dicom(self, path):
        return self._load("imag2", path)

    def calculate_temperature(self, **kwargs):
        self.calls.append(kwargs)
        self.temperature_map = np.full((4, 4), 37.0 + len(self.calls))
        return True


class _Signal:
    def __init__(self, callback):
        self.callback = callback

    def emit(self, *args):
        self.callback(*args)


def _cinema(model):
    cinema = CinemaModel.__new__(CinemaModel)
    cinema.thermometry_model = model
    cinema.temperature_sequence = []
    cinema.errors = []
    cinema.progress = []
    cinema.completed = []
    cinema.error_occurred = _Signal(cinema.errors.append)
    def record_progress(current, total):
        cinema.progress.append((current, total))

    cinema.calculation_progress = _Signal(record_progress)
    cinema.calculation_complete = _Signal(cinema.completed.append)
    return cinema


class FrameRegistrar:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def register_complex(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) in self.fail_on:
            return RegistrationResult(
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
                message="frame failed",
            )
        return RegistrationResult(
            warped_real=np.ones((4, 4)),
            warped_imag=np.zeros((4, 4)),
            transform=object(),
            config=RegistrationConfig(),
            metric_name="MattesMutualInformation",
            final_metric=-0.1,
            dice=0.95,
            iterations=3,
            stop_condition="converged",
            elapsed_s=0.01,
        )


def test_sequence_does_not_register_unless_asked():
    model = RecordingModel()
    cinema = _cinema(model)

    sequence = cinema.calculate_temperature_sequence(
        ["r0", "r1", "r2"], ["i0", "i1", "i2"]
    )

    assert sequence is not None
    assert len(sequence) == 2
    assert all(call["use_registration"] is False for call in model.calls)
    assert [item for item in model.loads if item[0] == "real1"] == [
        ("real1", "r0"),
        ("real1", "r1"),
    ]


def test_sequence_registers_every_later_frame_to_the_baseline():
    model = RecordingModel()
    cinema = _cinema(model)
    registrar = FrameRegistrar()

    sequence = cinema.calculate_temperature_sequence(
        ["r0", "r1", "r2"],
        ["i0", "i1", "i2"],
        apply_motion_correction=True,
        registrar=registrar,
        registration_config=RegistrationConfig(random_seed=4),
    )

    assert len(sequence) == 2
    assert all(frame is not None for frame in sequence)
    assert [item for item in model.loads if item[0] == "real1"] == [("real1", "r0")]
    assert [item for item in model.loads if item[0] == "real2"] == [
        ("real2", "r1"),
        ("real2", "r2"),
    ]
    assert [item for item in model.loads if item[0] == "imag2"] == [
        ("imag2", "i1"),
        ("imag2", "i2"),
    ]
    moving = [call["moving_real"] for call in registrar.calls]
    assert len(moving) == 2
    assert moving[0] is not moving[1]
    assert not np.array_equal(moving[0], moving[1])
    assert all(call["fixed_real"] is model.real1 for call in registrar.calls)
    assert all(call["use_registration"] is True for call in model.calls)
    assert registrar.calls[0]["config"].random_seed == 4


def test_a_failed_frame_does_not_remove_the_frames_around_it():
    model = RecordingModel()
    cinema = _cinema(model)
    registrar = FrameRegistrar(fail_on={1})

    sequence = cinema.calculate_temperature_sequence(
        ["r0", "r1", "r2"],
        ["i0", "i1", "i2"],
        apply_motion_correction=True,
        registrar=registrar,
    )

    assert sequence[0] is None
    assert sequence[1] is not None
    assert any("frame failed" in message for message in cinema.errors)
    assert len(model.calls) == 1


def test_a_failed_moving_load_is_skipped_instead_of_reusing_the_previous_frame():
    model = RecordingModel(fail_loads={("real2", "r2")})
    cinema = _cinema(model)
    registrar = FrameRegistrar()

    sequence = cinema.calculate_temperature_sequence(
        ["r0", "r1", "r2"],
        ["i0", "i1", "i2"],
        apply_motion_correction=True,
        registrar=registrar,
    )

    assert sequence[0] is not None
    assert sequence[1] is None
    assert len(registrar.calls) == 1
    assert registrar.calls[0]["moving_real"] is model.real2
    assert len(model.calls) == 1
    assert any("Failed to load real frame 3" in message for message in cinema.errors)
    assert ("imag2", "i2") not in model.loads


def test_a_failed_fixed_load_without_registration_is_stored_as_none():
    model = RecordingModel(fail_loads={("real1", "r1")})
    cinema = _cinema(model)

    sequence = cinema.calculate_temperature_sequence(
        ["r0", "r1", "r2"], ["i0", "i1", "i2"]
    )

    assert sequence[0] is not None
    assert sequence[1] is None
    assert len(model.calls) == 1
    assert any("Failed to load real frame 2" in message for message in cinema.errors)


def test_stop_event_keeps_frames_already_computed():
    model = RecordingModel()
    cinema = _cinema(model)
    registrar = FrameRegistrar()
    stop = threading.Event()

    def register_and_stop(**kwargs):
        outcome = FrameRegistrar.register_complex(registrar, **kwargs)
        stop.set()
        return outcome

    registrar.register_complex = register_and_stop
    sequence = cinema.calculate_temperature_sequence(
        ["r0", "r1", "r2"],
        ["i0", "i1", "i2"],
        apply_motion_correction=True,
        registrar=registrar,
        stop_event=stop,
    )

    assert sequence == [sequence[0]]
    assert sequence[0] is not None
    assert len(registrar.calls) == 1
    assert any("cancelled" in message for message in cinema.errors)
    assert len(cinema.completed) == 1


def test_stop_event_before_the_first_frame_stores_nothing():
    cinema = _cinema(RecordingModel())
    stop = threading.Event()
    stop.set()

    sequence = cinema.calculate_temperature_sequence(
        ["r0", "r1"], ["i0", "i1"], stop_event=stop
    )

    assert sequence == []
    assert any("cancelled" in message for message in cinema.errors)


def test_cinema_uses_liver_respiratory_preset():
    model = RecordingModel()
    cinema = _cinema(model)
    registrar = FrameRegistrar()
    config = RegistrationConfig.liver_respiratory()

    sequence = cinema.calculate_temperature_sequence(
        ["r0", "r1", "r2"],
        ["i0", "i1", "i2"],
        apply_motion_correction=True,
        registrar=registrar,
        registration_config=config,
    )

    assert sequence is not None
    used = registrar.calls[0]["config"]
    assert used.transform_kind == "rigid_bspline"
    assert used.metric == "ants_neighborhood_correlation"
    assert used.mask_kind == "otsu_dilated"


def test_cinema_auto_creates_registrar_when_none_is_passed():
    """The model no longer refuses a missing registrar."""
    model = RecordingModel()
    cinema = _cinema(model)

    sequence = cinema.calculate_temperature_sequence(
        ["r0", "r1"], ["i0", "i1"], apply_motion_correction=True
    )

    # No explicit registrar → the model creates a SimpleITKRegistrar
    # internally.  The synthetic 4×4 images may produce failed frames,
    # but the sequence itself is not refused as None.
    assert sequence is not None
