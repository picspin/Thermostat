"""Headless checks for the sequence workflow on the main window.

These tests use fake views or an offscreen QApplication.  They do not
require a display server and do not open the full thermometry window.
"""

from pathlib import Path

from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from mvc.controllers.advanced_features_controller import AdvancedFeaturesController
from mvc.controllers.thermometry_controller import ThermometryController
from mvc.views.thermometry_view import ThermometryView


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, value):
        self.text = value


class _StatusView:
    def __init__(self):
        self.statuses = []
        self.errors = []
        self.frame_counts = None
        self.all_real_files = []
        self.all_imag_files = []

    def update_status(self, message, timeout_ms=5000):
        self.statuses.append(message)

    def show_error(self, message):
        self.errors.append(message)

    def update_dicom_info(self, _text):
        return None

    def set_frame_counts(self, real_count, imag_count, b0_count=0):
        self.frame_counts = (real_count, imag_count, b0_count)


class _LoadModel:
    def __init__(self):
        self.real1 = None
        self.imag1 = None
        self.real2 = None
        self.imag2 = None
        self.dem_b0_map = None
        self.dicom_header_info = ""

    def load_real1_dicom(self, path):
        self.real1 = path
        return True

    def load_imag1_dicom(self, path):
        self.imag1 = path
        return True

    def load_real2_dicom(self, path):
        self.real2 = path
        return True

    def load_imag2_dicom(self, path):
        self.imag2 = path
        return True

    def load_b0_map(self, path):
        self.dem_b0_map = path
        return True


def _write_dicom(path):
    meta = Dataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "TEST"
    dataset.Modality = "MR"
    dataset.save_as(
        str(path),
        implicit_vr=False,
        little_endian=True,
        enforce_file_format=True,
    )


def _frame_view():
    view = ThermometryView.__new__(ThermometryView)
    view.frame_count_label = _Label()
    view.sequence_next_label = _Label()
    return view


def test_set_frame_counts_names_the_missing_kind():
    view = _frame_view()

    view.set_frame_counts(4, 0)
    assert view.frame_count_label.text == "Sequence: 4 real, 0 imag (missing: imag)"
    assert "imag" in view.sequence_next_label.text

    view.set_frame_counts(0, 3, 1)
    assert view.frame_count_label.text == (
        "Sequence: 0 real, 3 imag, 1 B0 (missing: real)"
    )
    assert "real" in view.sequence_next_label.text

    view.set_frame_counts(5, 5)
    assert view.frame_count_label.text == "Sequence: 5 real, 5 imag"
    assert "missing" not in view.frame_count_label.text
    assert "Generate Cinema" in view.sequence_next_label.text


def test_cinema_without_a_sequence_names_the_missing_kind():
    view = ThermometryView.__new__(ThermometryView)
    view.errors = []
    view.show_error = view.errors.append
    view.all_real_files = ["r1.dcm"]
    view.all_imag_files = []

    assert view._sequence_ready_for_cinema() is False
    assert "Load Sequence" in view.errors[-1]
    assert "Missing: imag" in view.errors[-1]

    view.all_real_files = []
    view.all_imag_files = []
    assert view._sequence_ready_for_cinema() is False
    assert "Missing: real, imag" in view.errors[-1]

    view.all_real_files = ["r1.dcm"]
    view.all_imag_files = ["i1.dcm"]
    assert view._sequence_ready_for_cinema() is False
    assert "Have 1 real and 1 imaginary" in view.errors[-1]


def test_batch_load_reports_missing_imaginary_frames(tmp_path):
    _write_dicom(Path(tmp_path) / "phase1_REAL.dcm")
    _write_dicom(Path(tmp_path) / "phase2_REAL.dcm")
    view = _StatusView()
    controller = ThermometryController.__new__(ThermometryController)
    controller.view = view
    controller.model = _LoadModel()

    controller.batch_load_dicom([str(tmp_path)])

    assert view.errors == []
    assert view.frame_counts == (2, 0, 0)
    assert any("missing: imag" in message for message in view.statuses)
    assert not any("missing: imaginary" in message for message in view.statuses)


def test_batch_load_reports_both_kinds(tmp_path):
    folder = Path(tmp_path)
    _write_dicom(folder / "phase1_REAL.dcm")
    _write_dicom(folder / "phase2_REAL.dcm")
    _write_dicom(folder / "phase1_IMAG.dcm")
    _write_dicom(folder / "phase2_IMAG.dcm")
    view = _StatusView()
    controller = ThermometryController.__new__(ThermometryController)
    controller.view = view
    controller.model = _LoadModel()

    controller.batch_load_dicom([str(folder)])

    assert view.frame_counts == (2, 2, 0)
    assert any("2 real frames and 2 imaginary frames" in message for message in view.statuses)
    assert not any("missing:" in message for message in view.statuses)


class _Checked:
    def __init__(self, checked):
        self.checked = checked

    def isChecked(self):
        return self.checked


class _Spin:
    def value(self):
        return 37.0


class _Job:
    def __init__(self):
        self.started = False

    def running(self):
        return False

    def start(self, *_args):
        self.started = True


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


class _FakeProgress:
    """Stand-in for QProgressDialog.

    Constructing a real indeterminate QProgressDialog aborts the interpreter
    when pytest has captured stderr, so this test does not open one.
    """

    def __init__(self, *_args, **_kwargs):
        self.canceled = _Signal()
        self.modality = None
        self.modal = None
        self.visible = False
        self.closed = False
        self.title = ""

    def setWindowTitle(self, title):
        self.title = title

    def setWindowModality(self, modality):
        self.modality = modality

    def setModal(self, modal):
        self.modal = modal

    def setMinimumDuration(self, _value):
        return None

    def setAutoClose(self, _value):
        return None

    def setAutoReset(self, _value):
        return None

    def show(self):
        self.visible = True

    def blockSignals(self, _blocked):
        return None

    def close(self):
        self.closed = True


def test_motion_progress_stays_non_modal(monkeypatch):
    from PyQt5.QtCore import Qt
    import mvc.controllers.thermometry_controller as controller_module

    monkeypatch.setattr(controller_module, "QProgressDialog", _FakeProgress)
    controller = ThermometryController.__new__(ThermometryController)
    controller.view = _StatusView()
    controller._wait_cursor = False
    controller._motion_generation = 1
    controller._show_motion_dialog()
    dialog = controller._motion_dialog
    assert isinstance(dialog, _FakeProgress)
    assert dialog.modality == Qt.NonModal
    assert dialog.modal is False
    assert dialog.visible is True
    assert dialog.title == "Motion correction"
    dialog.canceled.slots[0]()
    assert controller._motion_cancelled is True
    assert controller._motion_dialog is None
    assert dialog.closed is True
    assert any("Cancelling motion correction" in message for message in controller.view.statuses)


class _Widget:
    created = []

    def __init__(self, *args, **kwargs):
        _Widget.created.append(self)
        self.args = args
        self.kwargs = kwargs
        self.children = []
        self.modality = None
        self.modal = None
        self.visible = False
        self.closed = False
        self.deleted = False
        self.signals_blocked = False
        self.clicked = _Signal()
        self.rejected = _Signal()
        self.finished = _Signal()
        self.title = ""
        self._range = None
        self.value = None
        self.text = args[0] if args else ""

    def setWindowTitle(self, title):
        self.title = title

    def setText(self, text):
        self.text = text

    def setWindowModality(self, modality):
        self.modality = modality

    def setModal(self, modal):
        self.modal = modal

    def setFixedSize(self, *_args):
        return None

    def setMinimumSize(self, *_args):
        return None

    def setRange(self, start, end):
        self._range = (start, end)

    def setValue(self, value):
        self.value = value

    def addWidget(self, widget):
        self.children.append(widget)

    def addLayout(self, layout):
        self.children.append(layout)

    def show(self):
        self.visible = True

    def exec_(self):
        raise AssertionError("dialog must not be application-modal")

    def blockSignals(self, blocked):
        self.signals_blocked = blocked

    def close(self):
        self.closed = True
        if not self.signals_blocked:
            for slot in list(self.finished.slots):
                slot()

    def deleteLater(self):
        self.deleted = True


class _Axes:
    def errorbar(self, *_args, **_kwargs):
        return None

    def fill_between(self, *_args, **_kwargs):
        return None

    def set_xlabel(self, *_args, **_kwargs):
        return None

    def set_ylabel(self, *_args, **_kwargs):
        return None

    def set_title(self, *_args, **_kwargs):
        return None

    def grid(self, *_args, **_kwargs):
        return None

    def legend(self, *_args, **_kwargs):
        return None


def _install_widget_fakes(monkeypatch):
    import mvc.controllers.advanced_features_controller as features

    monkeypatch.setattr(features, "QDialog", _Widget)
    monkeypatch.setattr(features, "QVBoxLayout", _Widget)
    monkeypatch.setattr(features, "QHBoxLayout", _Widget)
    monkeypatch.setattr(features, "QLabel", _Widget)
    monkeypatch.setattr(features, "QProgressBar", _Widget)
    monkeypatch.setattr(features, "QPushButton", _Widget)
    monkeypatch.setattr(features, "FigureCanvas", lambda *_args, **_kwargs: _Widget())
    monkeypatch.setattr(features, "NavigationToolbar", lambda *_args, **_kwargs: _Widget())
    monkeypatch.setattr(features.plt, "subplots", lambda: (object(), _Axes()))


def test_cinema_progress_stays_non_modal_and_keeps_cancel(monkeypatch):
    from PyQt5.QtCore import Qt

    _install_widget_fakes(monkeypatch)
    _Widget.created = []
    view = _StatusView()
    view.all_real_files = ["r0", "r1"]
    view.all_imag_files = ["i0", "i1"]
    view.original_temp_input = _Spin()
    view.gaussian_filter_checkbox = _Checked(False)
    view.b0_correction_checkbox = _Checked(False)
    view.motion_correction_checkbox = _Checked(False)

    controller = AdvancedFeaturesController.__new__(AdvancedFeaturesController)
    controller.view = view
    controller.registrar = None
    controller.registration_config = None
    controller._cinema_stop = None
    controller._cinema_job = _Job()

    controller.generate_cinema()

    dialog = controller.progress_dialog
    assert view.errors == []
    assert controller._cinema_job.started is True
    assert dialog.modality == Qt.NonModal
    assert dialog.modal is False
    assert dialog.visible is True
    cancel = next(widget for widget in _Widget.created if widget.text == "Cancel")
    cancel.clicked.slots[0]()
    assert controller._cinema_stop.is_set()


def test_cinema_names_a_missing_kind_without_opening_a_dialog():
    view = _StatusView()
    view.all_real_files = ["r0", "r1"]
    view.all_imag_files = []
    controller = AdvancedFeaturesController.__new__(AdvancedFeaturesController)
    controller.view = view

    controller.generate_cinema()

    assert "Load Sequence" in view.errors[-1]
    assert "Missing: imag" in view.errors[-1]


def test_thermostat_curve_returns_while_non_modal(monkeypatch):
    from PyQt5.QtCore import Qt

    _install_widget_fakes(monkeypatch)
    _Widget.created = []
    view = _StatusView()

    class Model:
        def __init__(self):
            self.cleared = 0
            self.temperature_curve_data = {"kept": True}

        def clear_data(self):
            self.cleared += 1
            self.temperature_curve_data = None

    controller = AdvancedFeaturesController.__new__(AdvancedFeaturesController)
    controller.view = view
    controller.thermostat_model = Model()
    curve = {
        "time_points": [0, 1],
        "mean_temps": [37.0, 38.0],
        "std_devs": [0.1, 0.1],
        "min_temps": [36.0, 37.0],
        "max_temps": [38.0, 39.0],
    }

    controller.show_thermostat_dialog(curve)
    first = controller._thermostat_dialog
    assert first.modality == Qt.NonModal
    assert first.modal is False
    assert first.visible is True
    assert controller.thermostat_model.cleared == 0

    controller.show_thermostat_dialog(curve)
    assert controller._thermostat_dialog is not first
    assert first.closed is True
    assert first.deleted is True
    assert controller.thermostat_model.temperature_curve_data == {"kept": True}
    assert controller.thermostat_model.cleared == 0
    controller._thermostat_dialog.close()
    assert controller.thermostat_model.cleared == 1
    assert controller._thermostat_dialog is None


class _Button:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def isEnabled(self):
        return self.enabled

    def setEnabled(self, enabled):
        self.enabled = enabled


def test_running_cinema_refuses_reload_and_recalculation(monkeypatch):
    """A non-modal progress dialog must not let a second load replace the sequence."""
    _install_widget_fakes(monkeypatch)
    view = _StatusView()
    view.all_real_files = ["r0", "r1"]
    view.all_imag_files = ["i0", "i1"]
    view.frame_counts = (2, 2, 0)
    view.original_temp_input = _Spin()
    view.gaussian_filter_checkbox = _Checked(False)
    view.b0_correction_checkbox = _Checked(False)
    view.motion_correction_checkbox = _Checked(False)
    view.load_sequence_button = _Button()
    view.calc_button = _Button(enabled=False)

    cinema = AdvancedFeaturesController.__new__(AdvancedFeaturesController)
    cinema.view = view
    cinema.registrar = None
    cinema.registration_config = None
    cinema._cinema_stop = None
    cinema._cinema_job = _Job()
    cinema.generate_cinema()

    assert view.sequence_busy is True
    assert view.load_sequence_button.enabled is False
    assert view.calc_button.enabled is False
    assert view.errors == []

    thermo = ThermometryController.__new__(ThermometryController)
    thermo.view = view
    thermo.model = _LoadModel()
    thermo.model.real1 = "kept"
    calculated = []
    thermo.calculate_temperature = lambda *_args, **_kwargs: calculated.append(True)

    thermo.batch_load_dicom(["/does/not/need/to/exist"])
    assert view.all_real_files == ["r0", "r1"]
    assert view.all_imag_files == ["i0", "i1"]
    assert view.frame_counts == (2, 2, 0)
    assert thermo.model.real1 == "kept"
    assert "already running" in view.errors[-1]
    assert "loading another sequence" in view.errors[-1]

    thermo.begin_calculate_temperature(37.0, False, False, True)
    assert calculated == []
    assert "already running" in view.errors[-1]
    assert "calculating a temperature" in view.errors[-1]

    folder_view = ThermometryView.__new__(ThermometryView)
    folder_view.sequence_busy = True
    folder_view.errors = []
    folder_view.show_error = folder_view.errors.append
    folder_view.on_batch_load_folders()
    assert "loading another sequence" in folder_view.errors[-1]

    cinema._cinema_finished(["frame"])
    assert view.sequence_busy is False
    assert view.load_sequence_button.enabled is True
    # Motion correction had already disabled Calculate; Cinema must not turn it back on.
    assert view.calc_button.enabled is False
