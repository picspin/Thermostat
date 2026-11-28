"""Synthetic DICOM and NIfTI imports.  No real patient images."""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from mvc.controllers.thermometry_controller import ThermometryController
from mvc.models.cinema_model import CinemaModel
from mvc.models.sequence_import import import_sequences, phase_to_radians
from mvc.models.thermal_dose import ThermalDoseError, calculate_cinema_cem43
from mvc.models.thermometry_model import ThermometryModel


class _View:
    def __init__(self):
        self.statuses = []
        self.errors = []
        self.frame_counts = None
        self.all_real_files = []
        self.all_imag_files = []
        self.baseline_real_files = []
        self.baseline_imag_files = []
        self.header = ""

    def update_status(self, message, timeout_ms=5000):
        self.statuses.append(message)

    def show_error(self, message):
        self.errors.append(message)

    def update_dicom_info(self, text):
        self.header = text

    def set_frame_counts(self, real_count, imag_count, b0_count=0):
        self.frame_counts = (real_count, imag_count, b0_count)


class _Signal:
    def __init__(self):
        self.payload = []

    def emit(self, *args):
        self.payload.append(args)


def _controller(model=None):
    controller = ThermometryController.__new__(ThermometryController)
    controller.view = _View()
    controller.model = model if model is not None else ThermometryModel()
    return controller


def _write_dicom(
    path,
    pixels,
    image_type=None,
    complex_component=None,
    series_description="",
    rescale_slope=None,
    rescale_intercept=None,
    te=10.0,
    b0=3.0,
    acquisition_time=None,
):
    stored = np.asarray(pixels)
    if stored.dtype.kind == "f":
        stored = np.rint(stored).astype(np.uint16)
    elif int(np.min(stored)) < 0:
        stored = stored.astype(np.int16)
    else:
        stored = stored.astype(np.uint16)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "MR"
    dataset.PatientName = "SYNTHETIC"
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows, dataset.Columns = stored.shape
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1 if stored.dtype == np.int16 else 0
    dataset.PixelData = stored.tobytes()
    dataset.EchoTime = te
    dataset.MagneticFieldStrength = b0
    if image_type is not None:
        dataset.ImageType = list(image_type)
    if complex_component is not None:
        dataset.ComplexImageComponent = complex_component
    if series_description:
        dataset.SeriesDescription = series_description
    if rescale_slope is not None:
        dataset.RescaleSlope = float(rescale_slope)
    if rescale_intercept is not None:
        dataset.RescaleIntercept = float(rescale_intercept)
    if acquisition_time is not None:
        dataset.AcquisitionTime = acquisition_time
    dataset.save_as(
        str(path),
        implicit_vr=False,
        little_endian=True,
        enforce_file_format=True,
    )


def _write_nifti(path, array, te=10.0, b0=3.0):
    image = sitk.GetImageFromArray(np.asarray(array), isVector=False)
    image.SetSpacing(tuple(1.0 for _ in range(image.GetDimension())))
    note = f"TE={te};B0={b0}"
    image.SetMetaData("ITK_FileNotes", note)
    image.SetMetaData("aux_file", note[:24])
    sitk.WriteImage(image, str(path))


def _constant(value, shape=(4, 4)):
    return np.full(shape, value)


def test_phase_scale_uses_rescale_and_refuses_raw_integers():
    codes = np.array([[0, 2048], [4095, 1024]], dtype=np.uint16)
    mapped = phase_to_radians(codes, None, "codes")
    assert np.allclose(mapped[0, 0], -np.pi)
    assert np.allclose(mapped[0, 1], (2048 / 4095.0) * 2 * np.pi - np.pi)
    assert np.allclose(mapped[1, 0], np.pi)
    assert not np.allclose(mapped, codes.astype(float))

    class _Header:
        RescaleSlope = np.pi / 2048.0
        RescaleIntercept = -np.pi

    scaled = phase_to_radians(np.array([[2048.0]]), _Header(), "scaled")
    assert np.allclose(scaled, 0.0)

    class _Identity:
        RescaleSlope = 1.0
        RescaleIntercept = 0.0

    identity = phase_to_radians(codes, _Identity(), "identity")
    assert np.allclose(identity, mapped)

    assert np.allclose(phase_to_radians(np.array([[0.25, -0.5]]), None, "radians"), [0.25, -0.5])
    with pytest.raises(ValueError, match="not treated as radians"):
        phase_to_radians(np.array([[0, 1, 2, 3]]), None, "small-integers")


def test_real_imag_dicom_reaches_temperature(tmp_path):
    folder = Path(tmp_path)
    _write_dicom(
        folder / "t0_real.dcm",
        _constant(100),
        image_type=["ORIGINAL", "PRIMARY", "REAL"],
        acquisition_time="100000",
    )
    _write_dicom(
        folder / "t0_imag.dcm",
        _constant(0),
        image_type=["ORIGINAL", "PRIMARY", "IMAGINARY"],
        acquisition_time="100000",
    )
    _write_dicom(
        folder / "t1_real",
        _constant(80),
        image_type=["ORIGINAL", "PRIMARY", "REAL"],
        acquisition_time="100200",
    )
    _write_dicom(
        folder / "t1_imag",
        _constant(60),
        image_type=["ORIGINAL", "PRIMARY", "IMAGINARY"],
        acquisition_time="100200",
    )
    (folder / "readme.txt").write_text("not an image", encoding="utf-8")
    _write_dicom(
        folder / "unknown_slice.dcm",
        _constant(1),
        image_type=["ORIGINAL", "PRIMARY", "DERIVED"],
    )

    controller = _controller()
    controller.batch_load_dicom([str(folder)])

    joined = "\n".join(controller.view.errors)
    assert "unknown_slice.dcm" in joined
    assert "ImageType (0008,0008)" in joined
    assert "DERIVED" in joined
    assert "Complex Image Component (0008,9208)" in joined
    assert any("readme.txt" in message for message in controller.view.statuses)
    assert controller.view.frame_counts == (2, 2, 0)
    assert controller.model.te == 10.0
    assert controller.model.b0 == 3.0
    assert np.allclose(controller.model.real1, 100)
    assert np.allclose(controller.model.real2, 80)
    assert controller.model.calculate_temperature(37.0) is True
    assert np.isfinite(controller.model.temperature_map).any()


def test_magnitude_phase_without_suffix_is_not_b0_and_not_raw_radians(tmp_path):
    folder = Path(tmp_path)
    phase0 = _constant(2048)
    phase1 = _constant(3072)
    _write_dicom(
        folder / "swi_magnitude",
        _constant(50),
        image_type=["ORIGINAL", "PRIMARY", "MAGNITUDE"],
        complex_component="MAGNITUDE",
        series_description="SWI MAGNITUDE MAP",
        acquisition_time="090000",
    )
    _write_dicom(
        folder / "swi_phase",
        phase0,
        image_type=["ORIGINAL", "PRIMARY", "PHASE"],
        complex_component="PHASE",
        series_description="SWI PHASE MAP",
        acquisition_time="090000",
    )
    _write_dicom(
        folder / "swi_magnitude_late",
        _constant(50),
        complex_component="MAGNITUDE",
        series_description="SWI MAGNITUDE MAP",
        acquisition_time="090100",
    )
    _write_dicom(
        folder / "swi_phase_late",
        phase1,
        complex_component="PHASE",
        series_description="SWI PHASE MAP",
        acquisition_time="090100",
    )

    controller = _controller()
    controller.batch_load_dicom([str(folder)])

    assert controller.view.errors == []
    assert controller.view.frame_counts == (2, 2, 0)
    expected = (2048 / 4095.0) * 2 * np.pi - np.pi
    phase = np.angle(controller.model.real1 + 1j * controller.model.imag1)
    assert np.allclose(phase, expected)
    assert not np.allclose(controller.model.imag1, phase0.astype(float))
    assert controller.model.dem_b0_map is None
    assert controller.model.calculate_temperature(37.0) is True
    assert np.isfinite(controller.model.temperature_map).any()


def test_rescale_slope_is_the_phase_in_radians(tmp_path):
    folder = Path(tmp_path)
    slope = np.pi / 2048.0
    _write_dicom(
        folder / "mag.dcm",
        _constant(10),
        complex_component="MAGNITUDE",
        acquisition_time="080000",
    )
    _write_dicom(
        folder / "phase.dcm",
        _constant(2048),
        complex_component="PHASE",
        rescale_slope=slope,
        rescale_intercept=-np.pi,
        acquisition_time="080000",
    )
    _write_dicom(
        folder / "mag_b.dcm",
        _constant(10),
        complex_component="MAGNITUDE",
        acquisition_time="080100",
    )
    _write_dicom(
        folder / "phase_b.dcm",
        _constant(1024),
        complex_component="PHASE",
        rescale_slope=slope,
        rescale_intercept=-np.pi,
        acquisition_time="080100",
    )

    controller = _controller()
    controller.batch_load_dicom([str(folder)])
    phase = np.angle(controller.model.real1 + 1j * controller.model.imag1)
    assert np.allclose(phase, 0.0, atol=1e-6)
    assert controller.model.calculate_temperature(37.0) is True


def test_four_dimensional_nifti_volume_zero_is_baseline(tmp_path):
    series = np.zeros((2, 2, 4, 4), dtype=np.float32)
    series[0, 0] = 100
    series[0, 1] = 0
    series[1, 0] = 80
    series[1, 1] = 60
    path = Path(tmp_path) / "series.nii.gz"
    _write_nifti(path, series)

    controller = _controller()
    controller.batch_load_dicom([str(path)])

    assert controller.view.errors == []
    assert controller.view.frame_counts == (2, 2, 0)
    assert np.allclose(controller.model.real1, 100)
    assert np.allclose(controller.model.imag1, 0)
    assert np.allclose(controller.model.real2, 80)
    assert np.allclose(controller.model.imag2, 60)
    assert controller.model.te == 10.0
    assert controller.model.b0 == 3.0
    assert controller.model.calculate_temperature(37.0) is True
    assert np.isfinite(controller.model.temperature_map).any()

    cinema = CinemaModel.__new__(CinemaModel)
    cinema.thermometry_model = controller.model
    cinema.temperature_sequence = []
    cinema.validity_sequence = []
    cinema.error_occurred = _Signal()
    cinema.calculation_progress = _Signal()
    cinema.calculation_complete = _Signal()
    sequence = cinema.calculate_temperature_sequence(
        controller.view.all_real_files,
        controller.view.all_imag_files,
        37.0,
    )
    assert sequence is not None and len(sequence) == 1
    assert sequence[0] is not None and np.isfinite(sequence[0]).any()

    with pytest.raises(ThermalDoseError, match="equal spacing was not assumed"):
        calculate_cinema_cem43(
            controller.view.all_real_files,
            sequence,
            controller.view.all_imag_files,
        )


def test_two_3d_nifti_files_follow_filename_or_the_import_choice(tmp_path):
    magnitude = np.full((1, 4, 4), 4.0, dtype=np.float32)
    phase = np.full((1, 4, 4), 0.5, dtype=np.float32)
    named_mag = Path(tmp_path) / "vol_magnitude.nii.gz"
    named_phase = Path(tmp_path) / "vol_phase.nii.gz"
    _write_nifti(named_mag, magnitude)
    _write_nifti(named_phase, phase)
    named = import_sequences([str(named_mag), str(named_phase)])
    assert named.problems == []
    phase_image = np.angle(named.real_sources[0].array + 1j * named.imag_sources[0].array)
    assert np.allclose(phase_image, 0.5)

    first = Path(tmp_path) / "a.nii.gz"
    second = Path(tmp_path) / "b.nii.gz"
    _write_nifti(first, magnitude)
    _write_nifti(second, phase)
    missing_choice = import_sequences([str(first), str(second)])
    assert missing_choice.real_sources == []
    text = " ".join(missing_choice.problems)
    assert "a.nii.gz" in text and "b.nii.gz" in text
    assert "ImageType (0008,0008)" in text

    chosen = import_sequences(
        [str(second), str(first)], component_pair="magnitude_phase"
    )
    assert chosen.problems == []
    chosen_phase = np.angle(chosen.real_sources[0].array + 1j * chosen.imag_sources[0].array)
    assert np.allclose(chosen_phase, 0.5)


def test_baseline_first_frame_is_compared_with_sequence_first_frame(tmp_path):
    baseline = Path(tmp_path) / "baseline"
    sequence = Path(tmp_path) / "sequence"
    baseline.mkdir()
    sequence.mkdir()
    _write_dicom(baseline / "real.dcm", _constant(10), image_type=["ORIGINAL", "PRIMARY", "REAL"])
    _write_dicom(
        baseline / "imag.dcm", _constant(0), image_type=["ORIGINAL", "PRIMARY", "IMAGINARY"]
    )
    _write_dicom(
        sequence / "t0_real.dcm",
        _constant(20),
        image_type=["ORIGINAL", "PRIMARY", "REAL"],
        acquisition_time="110000",
    )
    _write_dicom(
        sequence / "t0_imag.dcm",
        _constant(0),
        image_type=["ORIGINAL", "PRIMARY", "IMAGINARY"],
        acquisition_time="110000",
    )
    _write_dicom(
        sequence / "t1_real.dcm",
        _constant(30),
        image_type=["ORIGINAL", "PRIMARY", "REAL"],
        acquisition_time="110100",
    )
    _write_dicom(
        sequence / "t1_imag.dcm",
        _constant(0),
        image_type=["ORIGINAL", "PRIMARY", "IMAGINARY"],
        acquisition_time="110100",
    )

    controller = _controller()
    controller.batch_load_dicom([str(sequence)], role="sequence")
    assert np.allclose(controller.model.real1, 20)
    assert np.allclose(controller.model.real2, 30)

    controller.batch_load_dicom([str(baseline)], role="baseline")
    assert np.allclose(controller.model.real1, 10)
    assert np.allclose(controller.model.real2, 20)
    assert len(controller.view.all_real_files) == 2
    assert controller.model.calculate_temperature(37.0) is True
    assert np.isfinite(controller.model.temperature_map).any()
