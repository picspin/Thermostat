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
    acquisition_datetime=None,
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
    if acquisition_datetime is not None:
        dataset.AcquisitionDateTime = acquisition_datetime
    dataset.save_as(
        str(path),
        implicit_vr=False,
        little_endian=True,
        enforce_file_format=True,
    )


def _write_nifti(path, array, te=10.0, b0=3.0, note=None):
    image = sitk.GetImageFromArray(np.asarray(array), isVector=False)
    image.SetSpacing(tuple(1.0 for _ in range(image.GetDimension())))
    if note is None and te is not None and b0 is not None:
        note = f"TE={te};B0={b0}"
    if note:
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
    with pytest.raises(ValueError, match="0..4095"):
        phase_to_radians(np.array([[-2048, 2047]]), None, "signed")

    class _Wide:
        RescaleSlope = 2.0
        RescaleIntercept = 0.0

    with pytest.raises(ValueError, match="outside"):
        phase_to_radians(np.array([[0.0, 4.0]]), _Wide(), "wide")


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


def test_single_letter_image_type_is_not_a_b0_map(tmp_path):
    folder = Path(tmp_path)
    _write_dicom(
        folder / "frame_a",
        _constant(40),
        image_type=["ORIGINAL", "PRIMARY", "M"],
        series_description="SWI MAGNITUDE MAP",
        acquisition_time="090000",
    )
    _write_dicom(
        folder / "frame_b",
        _constant(2048),
        image_type=["ORIGINAL", "PRIMARY", "P"],
        series_description="SWI PHASE MAP",
        acquisition_time="090000",
    )
    _write_dicom(
        folder / "frame_c.dcm",
        _constant(40),
        image_type=["ORIGINAL", "PRIMARY"],
        series_description="SWI MAGNITUDE MAP",
    )

    paired = import_sequences([str(folder / "frame_a"), str(folder / "frame_b")])
    assert paired.b0_paths == []
    assert paired.problems == []
    assert len(paired.real_sources) == 1
    phase = np.angle(paired.real_sources[0].array + 1j * paired.imag_sources[0].array)
    expected = (2048 / 4095.0) * 2 * np.pi - np.pi
    assert np.allclose(phase, expected)

    described = import_sequences([str(folder / "frame_c.dcm")])
    assert described.b0_paths == []
    assert any("frame_c.dcm" in problem and "magnitude" in problem for problem in described.problems)

    letters = Path(tmp_path) / "letters"
    letters.mkdir()
    _write_dicom(letters / "one", _constant(5), image_type=["ORIGINAL", "PRIMARY", "R"])
    _write_dicom(letters / "two", _constant(6), image_type=["ORIGINAL", "PRIMARY", "I"])
    native = import_sequences([str(letters)])
    assert native.b0_paths == []
    assert native.problems == []
    assert len(native.real_sources) == 1 and len(native.imag_sources) == 1


def test_partial_import_clears_stale_frames_and_scan_parameters(tmp_path):
    first = Path(tmp_path) / "full"
    first.mkdir()
    _write_dicom(
        first / "t0_real.dcm",
        _constant(100),
        image_type=["ORIGINAL", "PRIMARY", "REAL"],
        acquisition_time="100000",
        te=11,
        b0=1.5,
    )
    _write_dicom(
        first / "t0_imag.dcm",
        _constant(0),
        image_type=["ORIGINAL", "PRIMARY", "IMAGINARY"],
        acquisition_time="100000",
        te=11,
        b0=1.5,
    )
    _write_dicom(
        first / "t1_real.dcm",
        _constant(80),
        image_type=["ORIGINAL", "PRIMARY", "REAL"],
        acquisition_time="100200",
        te=11,
        b0=1.5,
    )
    _write_dicom(
        first / "t1_imag.dcm",
        _constant(60),
        image_type=["ORIGINAL", "PRIMARY", "IMAGINARY"],
        acquisition_time="100200",
        te=11,
        b0=1.5,
    )
    controller = _controller()
    controller.batch_load_dicom([str(first)])
    assert controller.model.te == 11.0
    assert controller.model.b0 == 1.5
    assert np.allclose(controller.model.real2, 80)

    baseline = Path(tmp_path) / "real_only"
    baseline.mkdir()
    _write_dicom(baseline / "only_real.dcm", _constant(7), image_type=["ORIGINAL", "PRIMARY", "REAL"])
    controller.batch_load_dicom([str(baseline)], role="baseline")
    assert np.allclose(controller.model.real1, 7)
    assert controller.model.imag1 is None
    assert np.allclose(controller.model.real2, 100)
    assert any("missing: imag" in message for message in controller.view.statuses)
    assert controller.model.calculate_temperature(37.0) is False

    fresh = _controller()
    fresh.batch_load_dicom([str(first)])
    assert fresh.model.te == 11.0
    # Time is the second axis.  A leading size-1 axis is the last NIfTI
    # dimension and is dropped on write, so this shape stays 4D: one volume,
    # component axis length 2.
    bare = np.zeros((2, 1, 4, 4), dtype=np.float32)
    bare[0, 0] = 3
    bare[1, 0] = 4
    nifti = Path(tmp_path) / "bare.nii.gz"
    _write_nifti(nifti, bare, te=None, b0=None)
    fresh.batch_load_dicom([str(nifti)], role="sequence")
    assert fresh.model.te is None
    assert fresh.model.b0 is None
    assert np.allclose(fresh.model.real1, 3)
    assert fresh.model.real2 is None
    assert fresh.model.imag2 is None
    assert fresh.model.calculate_temperature(37.0) is False


def test_mixed_dicom_and_nifti_are_not_concatenated(tmp_path):
    series = np.zeros((2, 2, 4, 4), dtype=np.float32)
    series[0, 0] = 1
    series[1, 0] = 2
    nifti = Path(tmp_path) / "series.nii.gz"
    _write_nifti(nifti, series)
    _write_dicom(
        Path(tmp_path) / "t0_real.dcm",
        _constant(9),
        image_type=["ORIGINAL", "PRIMARY", "REAL"],
    )
    _write_dicom(
        Path(tmp_path) / "t0_imag.dcm",
        _constant(0),
        image_type=["ORIGINAL", "PRIMARY", "IMAGINARY"],
    )
    result = import_sequences([str(tmp_path)])
    assert result.real_sources == []
    assert result.imag_sources == []
    text = " ".join(result.problems)
    assert "series.nii.gz" in text and "t0_real.dcm" in text
    assert "not combined" in text


def test_unreadable_empty_and_missing_paths_are_named(tmp_path):
    folder = Path(tmp_path) / "empty"
    folder.mkdir()
    controller = _controller()
    controller.batch_load_dicom([str(folder)])
    assert any("No DICOM or NIfTI" in message for message in controller.view.errors)

    bad_nifti = Path(tmp_path) / "broken.nii.gz"
    bad_nifti.write_bytes(b"not a nifti")
    truncated = Path(tmp_path) / "truncated.dcm"
    truncated.write_bytes(b"DICM" + b"\0" * 32)
    result = import_sequences([str(bad_nifti), str(truncated), "/no/such/frame.dcm"])
    assert "broken.nii.gz" in result.skipped
    assert "truncated.dcm" in result.skipped
    assert "frame.dcm" in result.missing
    assert result.real_sources == []

    controller.batch_load_dicom([str(bad_nifti), "/no/such/frame.dcm"])
    assert any("Skipped unreadable: broken.nii.gz" in message for message in controller.view.statuses)
    assert any("Not found: frame.dcm" in message for message in controller.view.statuses)
    assert not any("Skipped unreadable: frame.dcm" in message for message in controller.view.statuses)


def test_duplicate_path_is_imported_once_and_clocks_are_not_lexicographic(tmp_path):
    folder = Path(tmp_path) / "clocks"
    folder.mkdir()
    _write_dicom(
        folder / "late.dcm",
        _constant(1),
        image_type=["ORIGINAL", "PRIMARY", "REAL"],
        acquisition_datetime="20261128100000",
    )
    _write_dicom(
        folder / "early.dcm",
        _constant(3),
        image_type=["ORIGINAL", "PRIMARY", "REAL"],
        acquisition_datetime="20261128090000",
    )
    _write_dicom(
        folder / "time_only.dcm",
        _constant(2),
        image_type=["ORIGINAL", "PRIMARY", "REAL"],
        acquisition_time="080000",
    )
    result = import_sequences([str(folder), str(folder / "early.dcm")])
    assert [Path(path).name for path in result.real_sources] == [
        "early.dcm",
        "late.dcm",
        "time_only.dcm",
    ]


def test_bad_nifti_note_does_not_drop_the_other_file(tmp_path):
    good = np.zeros((2, 2, 4, 4), dtype=np.float32)
    good[0, 0] = 10
    good[0, 1] = 0
    good[1, 0] = 0
    good[1, 1] = 10
    bad = np.zeros((2, 2, 4, 4), dtype=np.float32)
    bad[0, 0] = 1
    _write_nifti(Path(tmp_path) / "good.nii.gz", good)
    _write_nifti(Path(tmp_path) / "bad.nii.gz", bad, note="TE=1.2.3;B0=3")
    result = import_sequences([str(tmp_path)])
    assert any("bad.nii.gz" in problem for problem in result.problems)
    assert any(np.allclose(source.array, 10) for source in result.real_sources)
    assert len(result.real_sources) == 4


def test_nifti_keeps_real_imag_when_magnitude_is_also_present(tmp_path):
    _write_nifti(Path(tmp_path) / "frame_real.nii.gz", np.full((1, 4, 4), 8, np.float32))
    _write_nifti(Path(tmp_path) / "frame_imag.nii.gz", np.full((1, 4, 4), 2, np.float32))
    _write_nifti(Path(tmp_path) / "frame_magnitude.nii.gz", np.full((1, 4, 4), 5, np.float32))
    result = import_sequences([str(tmp_path)])
    assert len(result.real_sources) == 1
    assert np.allclose(result.real_sources[0].array, 8)
    assert np.allclose(result.imag_sources[0].array, 2)
    assert any("frame_magnitude.nii.gz" in problem for problem in result.problems)


def test_shape_mismatch_does_not_calculate(tmp_path):
    baseline = Path(tmp_path) / "baseline"
    sequence = Path(tmp_path) / "sequence"
    baseline.mkdir()
    sequence.mkdir()
    _write_dicom(baseline / "real.dcm", _constant(10, (4, 4)), image_type=["ORIGINAL", "PRIMARY", "REAL"])
    _write_dicom(baseline / "imag.dcm", _constant(0, (4, 4)), image_type=["ORIGINAL", "PRIMARY", "IMAGINARY"])
    _write_dicom(sequence / "real.dcm", _constant(8, (5, 6)), image_type=["ORIGINAL", "PRIMARY", "REAL"])
    _write_dicom(sequence / "imag.dcm", _constant(1, (5, 6)), image_type=["ORIGINAL", "PRIMARY", "IMAGINARY"])
    controller = _controller()
    errors = []
    controller.model.error_occurred.connect(errors.append)
    controller.batch_load_dicom([str(sequence)])
    controller.batch_load_dicom([str(baseline)], role="baseline")
    assert controller.model.real1.shape == (4, 4)
    assert controller.model.real2.shape == (5, 6)
    assert controller.model.calculate_temperature(37.0) is False
    assert any("Error calculating temperature" in message for message in errors)
