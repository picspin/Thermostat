"""Synthetic CEM43 checks for a Cinema sequence.

The registration defaults are asserted here as well: this feature must not
change ``RegistrationConfig()``.
"""

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from mvc.controllers.advanced_features_controller import AdvancedFeaturesController
from mvc.models.cinema_model import CinemaModel
from mvc.models.registration import RegistrationConfig
from mvc.models.thermal_dose import (
    REFERENCE_CEM43_EQUIVALENT_MINUTES,
    REFERENCE_DISCLAIMER,
    REFERENCE_TEMPERATURE_C,
    AcquisitionClock,
    ThermalDoseError,
    accumulate_cem43,
    accumulate_thermal_dose,
    calculate_cinema_cem43,
    intervals_minutes,
    parse_acquisition_time,
    r_factor,
    read_acquisition_clock,
)
from mvc.models.thermometry_model import ThermometryModel


def test_one_minute_at_43c_is_one_cem43():
    dose = accumulate_cem43([np.full((2, 2), 43.0)], [1.0])

    np.testing.assert_allclose(dose, 1.0)


def test_r_factor_differs_at_42c_and_44c():
    assert r_factor(42.0) == 0.25
    assert r_factor(44.0) == 0.5
    assert r_factor(43.0) == 0.5
    assert r_factor(42.0) != r_factor(44.0)
    np.testing.assert_allclose(
        r_factor(np.array([42.0, 43.0, 44.0])),
        [0.25, 0.5, 0.5],
    )

    cool = accumulate_cem43([np.full((1, 1), 42.0)], [1.0])
    hot = accumulate_cem43([np.full((1, 1), 44.0)], [1.0])
    # 0.25^(43-42) = 0.25, and 0.5^(43-44) = 2.
    np.testing.assert_allclose(cool, 0.25)
    np.testing.assert_allclose(hot, 2.0)
    assert cool.item() != hot.item()


def test_missing_interval_fails_and_does_not_assume_equal_spacing():
    with pytest.raises(ThermalDoseError, match="equal spacing was not assumed"):
        intervals_minutes([AcquisitionClock(0.0), None])

    with pytest.raises(ThermalDoseError, match="equal spacing was not assumed"):
        accumulate_cem43([np.full((1, 1), 43.0)], [np.nan])

    blank = Dataset()
    blank.AcquisitionTime = ""
    blank.SeriesTime = "120000"
    other = Dataset()
    other.SeriesTime = "120100"
    with pytest.raises(ThermalDoseError, match="AcquisitionTime"):
        calculate_cinema_cem43([blank, other], [np.full((1, 1), 43.0)])


def test_series_time_and_acquisition_datetime_are_not_substitutes():
    first = Dataset()
    first.SeriesTime = "100000"
    first.AcquisitionDateTime = "20200101100000"
    second = Dataset()
    second.SeriesTime = "100100"
    second.AcquisitionDateTime = "20200101100100"

    assert read_acquisition_clock(first) is None
    with pytest.raises(ThermalDoseError, match="equal spacing was not assumed"):
        calculate_cinema_cem43([first, second], [np.full((1, 1), 43.0)])


def test_acquisition_time_interval_of_one_minute_at_43c():
    first = Dataset()
    first.AcquisitionTime = "100000"
    second = Dataset()
    second.AcquisitionTime = "100100"

    result = calculate_cinema_cem43([first, second], [np.full((2, 2), 43.0)])

    np.testing.assert_allclose(result.cem43, 1.0)
    np.testing.assert_allclose(result.delta_t_minutes, [1.0])
    np.testing.assert_allclose(result.elapsed_minutes, [1.0])


def test_imaginary_acquisition_time_is_used_only_when_the_real_image_has_none():
    real_baseline = Dataset()
    real_baseline.SeriesTime = "100000"
    imag_baseline = Dataset()
    imag_baseline.AcquisitionTime = "100000"
    real_later = Dataset()
    real_later.AcquisitionTime = "100200"
    # A conflicting imaginary time must not override a real AcquisitionTime.
    imag_later = Dataset()
    imag_later.AcquisitionTime = "100100"

    result = calculate_cinema_cem43(
        [real_baseline, real_later],
        [np.full((1, 1), 43.0)],
        imag_sources=[imag_baseline, imag_later],
    )

    np.testing.assert_allclose(result.delta_t_minutes, [2.0])
    np.testing.assert_allclose(result.cem43, 2.0)


def test_dicom_file_without_acquisition_time_fails(tmp_path):
    first = tmp_path / "phase0.dcm"
    second = tmp_path / "phase1.dcm"
    _write_dicom(first, series_time="100000")
    _write_dicom(second, series_time="100100", acquisition_time="100100")

    with pytest.raises(ThermalDoseError, match="AcquisitionTime"):
        calculate_cinema_cem43(
            [str(first), str(second)],
            [np.full((1, 1), 43.0)],
        )


def test_dicom_file_acquisition_time_is_one_cem43_per_minute(tmp_path):
    first = tmp_path / "phase0.dcm"
    second = tmp_path / "phase1.dcm"
    _write_dicom(first, acquisition_time="153000", series_time="000000")
    _write_dicom(second, acquisition_time="153100", series_time="235959")

    result = calculate_cinema_cem43(
        [str(first), str(second)],
        [np.full((1, 1), 43.0)],
    )

    np.testing.assert_allclose(result.cem43, 1.0)


def test_low_signal_and_invalid_voxels_do_not_accumulate():
    temperature = np.array([[43.0, 50.0], [np.nan, 43.0]])
    low_signal = np.array([[True, False], [True, True]])

    dose = accumulate_cem43([temperature], [1.0], valid_masks=[low_signal])

    np.testing.assert_allclose(dose[0, 0], 1.0)
    assert np.isnan(dose[0, 1])
    assert np.isnan(dose[1, 0])
    np.testing.assert_allclose(dose[1, 1], 1.0)
    # The input temperature is not rewritten just because dose skipped it.
    assert temperature[0, 1] == 50.0
    assert np.isnan(temperature[1, 0])


def test_low_signal_on_one_frame_does_not_block_a_later_valid_frame():
    first = np.full((1, 1), 50.0)
    second = np.full((1, 1), 43.0)
    masks = [np.array([[False]]), np.array([[True]])]

    dose = accumulate_cem43([first, second], [1.0, 1.0], valid_masks=masks)

    np.testing.assert_allclose(dose, 1.0)


def test_missing_frame_adds_no_dose_but_later_frames_still_accumulate():
    maps = [np.full((1, 1), 43.0), None, np.full((1, 1), 43.0)]

    result = accumulate_thermal_dose(maps, [1.0, 1.0, 1.0])

    np.testing.assert_allclose(result.cem43, 2.0)
    np.testing.assert_allclose(result.elapsed_minutes, [1.0, 2.0, 3.0])


def test_roi_temperature_and_cumulative_dose():
    hot = np.array([[43.0, 44.0], [37.0, 37.0]])
    roi = np.array([[True, True], [False, False]])

    result = accumulate_thermal_dose([hot], [1.0], roi_mask=roi)

    # Mean of 1 CEM43 (43°C) and 2 CEM43 (44°C).
    np.testing.assert_allclose(result.roi_mean_temperature, [43.5])
    np.testing.assert_allclose(result.roi_cumulative_mean, [1.5])
    np.testing.assert_allclose(result.roi_cumulative_max, [2.0])
    assert result.roi_mean_temperature is not None


def test_backwards_clock_is_not_turned_into_a_midnight_dose():
    # A cinema series that is out of order must not gain almost 24 hours.
    clocks = [
        AcquisitionClock(23 * 3600 + 59 * 60),
        AcquisitionClock(60.0),
    ]
    with pytest.raises(ThermalDoseError, match="not positive"):
        intervals_minutes(clocks)

    from datetime import date

    dated = [
        AcquisitionClock(23 * 3600 + 59 * 60, date(2020, 1, 1).toordinal()),
        AcquisitionClock(60.0, date(2020, 1, 2).toordinal()),
    ]
    assert intervals_minutes(dated)[0] == pytest.approx(2.0)
    partial = [
        AcquisitionClock(60.0, date(2020, 1, 1).toordinal()),
        AcquisitionClock(120.0, None),
    ]
    with pytest.raises(ThermalDoseError, match="AcquisitionDate"):
        intervals_minutes(partial)
    assert parse_acquisition_time("235900") == 23 * 3600 + 59 * 60
    assert parse_acquisition_time("000100.0") == pytest.approx(60.0)
    assert parse_acquisition_time("") is None
    assert parse_acquisition_time(None) is None


def test_fractional_acquisition_time_requires_hhmmss():
    assert parse_acquisition_time("101010.5") == pytest.approx(10 * 3600 + 10 * 60 + 10.5)
    with pytest.raises(ThermalDoseError, match="not a DICOM TM"):
        parse_acquisition_time("10.5")
    with pytest.raises(ThermalDoseError, match="not a DICOM TM"):
        parse_acquisition_time("1010.5")


def test_overflowing_temperature_is_not_stored_as_zero_dose():
    temperature = np.array([[43.0, 1.0e6]])

    dose = accumulate_cem43([temperature], [1.0])

    np.testing.assert_allclose(dose[0, 0], 1.0)
    assert np.isnan(dose[0, 1])


def test_non_positive_acquisition_interval_fails():
    with pytest.raises(ThermalDoseError, match="not positive"):
        intervals_minutes([AcquisitionClock(100.0), AcquisitionClock(100.0)])


def test_reference_lines_are_safety_guides_not_treatment_completion():
    assert REFERENCE_TEMPERATURE_C == 39.0
    assert REFERENCE_CEM43_EQUIVALENT_MINUTES == (2.0, 9.0)
    assert "van Rhoon et al., Eur Radiol 2013" in REFERENCE_DISCLAIMER
    assert "MR radiofrequency safety recommendations" in REFERENCE_DISCLAIMER
    assert "not a treatment endpoint" in REFERENCE_DISCLAIMER
    assert "治疗完成" not in REFERENCE_DISCLAIMER


def test_registration_defaults_stay_historical():
    config = RegistrationConfig()

    assert config.metric == "mattes_mi"
    assert config.transform_kind == "bspline"
    assert config.mask_kind == "none"
    assert config.mesh_size == 8
    assert config.histogram_bins == 50
    assert config.sampling_percentage == 0.1
    assert config.random_seed is None
    assert config.learning_rate == 1.0
    assert config.iterations == 100
    assert config.foreground_mask is False
    assert config.bspline_grid_spacing_mm is None


def test_low_signal_mask_does_not_rewrite_the_temperature_map():
    model = _thermometry()
    model.real1[0, 0] = 0.0
    model.imag1[0, 0] = 0.0
    model.real2[0, 0] = 0.0
    model.imag2[0, 0] = 0.0

    assert model.calculate_temperature(37.0) is True

    assert np.isfinite(model.temperature_map[0, 0])
    assert model.valid_temperature_mask[1, 1]
    assert not model.valid_temperature_mask[0, 0]

    model.clear_data()
    assert model.valid_temperature_mask is None


def test_cinema_copies_the_low_signal_mask_without_changing_temperature():
    model = _RecordingModel()
    model.valid_temperature_mask = np.zeros((4, 4), dtype=bool)
    model.valid_temperature_mask[0, 0] = True
    cinema = _cinema(model)

    sequence = cinema.calculate_temperature_sequence(
        ["r0", "r1"], ["i0", "i1"], apply_motion_correction=False
    )

    assert len(sequence) == 1
    assert len(cinema.validity_sequence) == 1
    assert cinema.validity_sequence[0][0, 0]
    assert not cinema.validity_sequence[0][1, 1]
    model.valid_temperature_mask[:] = False
    assert cinema.validity_sequence[0][0, 0]
    np.testing.assert_allclose(sequence[0], model.temperature_map)


def test_controller_reports_a_missing_acquisition_time():
    view = _FakeView()
    view.all_real_files = [Dataset(), Dataset()]
    view.all_imag_files = [Dataset(), Dataset()]
    controller = AdvancedFeaturesController.__new__(AdvancedFeaturesController)
    controller.view = view
    controller.cinema_model = type("Cinema", (), {"validity_sequence": [None]})()

    controller._show_thermal_dose([np.full((2, 2), 43.0)])

    assert view.errors
    assert "AcquisitionTime" in view.errors[0]
    assert "equal spacing was not assumed" in view.errors[0]
    assert view.status_timeouts[-1] == 0


def test_dose_dialog_keeps_reference_lines_optional():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from mvc.views.thermal_dose_dialog import create_thermal_dose_dialog

    # Keep a reference.  A temporary QApplication is destroyed immediately
    # and the next QWidget then aborts.
    global _QT_APP
    if QApplication.instance() is None:
        _QT_APP = QApplication([])
    else:
        _QT_APP = QApplication.instance()
    result = accumulate_thermal_dose(
        [np.full((2, 2), 43.0)],
        [1.0],
        roi_mask=np.ones((2, 2), dtype=bool),
    )
    dialog = create_thermal_dose_dialog(None, result)
    try:
        from PyQt5.QtWidgets import QCheckBox, QLabel

        assert dialog.windowTitle() == "CEM43 thermal dose"
        boxes = {box.text(): box for box in dialog.findChildren(QCheckBox)}
        temperature_box = boxes["39°C reference line"]
        dose_2 = boxes["CEM43 = 2 reference line"]
        dose_9 = boxes["CEM43 = 9 reference line"]
        assert temperature_box.isChecked() is False
        assert dose_2.isChecked() is False
        assert dose_9.isChecked() is False
        texts = [label.text() for label in dialog.findChildren(QLabel)]
        assert REFERENCE_DISCLAIMER in texts
        disclaimer = REFERENCE_DISCLAIMER
        assert "治疗完成" not in disclaimer
        assert "treatment is complete" in disclaimer
        assert "do not indicate" in disclaimer
        # Unchecked guides are not drawn.  The ROI series is not a horizontal guide.
        temperature_axis, dose_axis = _dialog_axes(dialog)
        assert _horizontal_values(temperature_axis) == []
        assert _horizontal_values(dose_axis) == []
        temperature_box.setChecked(True)
        dose_2.setChecked(True)
        dose_9.setChecked(True)
        assert _horizontal_values(temperature_axis) == [39.0]
        assert _horizontal_values(dose_axis) == [2.0, 9.0]
    finally:
        dialog.close()


class _FakeView:
    def __init__(self):
        self.errors = []
        self.statuses = []
        self.status_timeouts = []
        self.roi_manager = None

    def show_error(self, message):
        self.errors.append(message)

    def update_status(self, message, timeout_ms=5000):
        self.statuses.append(message)
        self.status_timeouts.append(timeout_ms)


class _Signal:
    def __init__(self, callback):
        self.callback = callback

    def emit(self, *args):
        self.callback(*args)


class _RecordingModel:
    def __init__(self):
        self.real1 = np.ones((4, 4))
        self.imag1 = np.zeros((4, 4))
        self.real2 = np.ones((4, 4))
        self.imag2 = np.zeros((4, 4))
        self.temperature_map = np.full((4, 4), 43.0)
        self.valid_temperature_mask = None
        self.registration_result = None

    def load_real1_dicom(self, _path):
        return True

    def load_imag1_dicom(self, _path):
        return True

    def load_real2_dicom(self, _path):
        return True

    def load_imag2_dicom(self, _path):
        return True

    def calculate_temperature(self, **_kwargs):
        self.temperature_map = np.full((4, 4), 43.0)
        return True


def _cinema(model):
    cinema = CinemaModel.__new__(CinemaModel)
    cinema.thermometry_model = model
    cinema.temperature_sequence = []
    cinema.validity_sequence = []
    cinema.error_occurred = _Signal(lambda _message: None)
    cinema.calculation_progress = _Signal(lambda *_args: None)
    cinema.calculation_complete = _Signal(lambda *_args: None)
    return cinema


def _thermometry():
    model = ThermometryModel()
    model.te = 10.0
    model.b0 = 3.0
    model.real1 = np.ones((4, 4))
    model.imag1 = np.zeros((4, 4))
    model.real2 = np.ones((4, 4))
    model.imag2 = np.zeros((4, 4))
    return model


def _write_dicom(path, acquisition_time=None, series_time=None):
    meta = Dataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "TEST"
    dataset.Modality = "MR"
    if series_time is not None:
        dataset.SeriesTime = series_time
    if acquisition_time is not None:
        dataset.AcquisitionTime = acquisition_time
    dataset.save_as(
        str(path),
        implicit_vr=False,
        little_endian=True,
        enforce_file_format=True,
    )


def _dialog_axes(dialog):
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    canvases = dialog.findChildren(FigureCanvas)
    # Map canvas is created first; temperature then dose follow.
    assert len(canvases) == 3
    return canvases[1].figure.axes[0], canvases[2].figure.axes[0]


def _horizontal_values(axis):
    values = []
    for line in axis.lines:
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if ydata.size >= 2 and np.allclose(ydata, ydata[0]):
            values.append(float(ydata[0]))
    return values
