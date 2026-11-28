"""Load a baseline or a comparison sequence from DICOM and NIfTI.

One import is a set of frames, not four single real/imag clicks.  Real and
imaginary DICOM stay as paths so Cinema and CEM43 keep reading Acquisition
Time from the file.  Magnitude/phase DICOM and NIfTI are converted to the
complex pair the PRF calculation already uses.

CEM43 is not changed here.  A NIfTI frame has no Acquisition Time; the dose
module still refuses a missing clock instead of inventing equal spacing.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pydicom
import SimpleITK as sitk

IMAGE_TYPE_TAG = (0x0008, 0x0008)
COMPLEX_COMPONENT_TAG = (0x0008, 0x9208)
ACQUISITION_DATETIME_TAG = (0x0008, 0x002A)

# 0 and 4095 are the ends of an unsigned 12-bit phase code.
# Dividing by 4095 maps code 4095 to exactly +π.  A vendor rescale of 2π/4096
# maps 4095 to just under +π (about 0.00077 rad less).  That difference is
# negligible for PRF temperature, and 4096 is not itself a stored code, so
# the unsigned map uses 4095 on purpose.  Signed codes (−2048…2047) are
# rejected rather than guessed.
_PHASE_CODE_MAX = 4095.0
_RADIAN_LIMIT = float(np.pi) + 1e-3
_COMPONENT_TOKENS = (
    ("PHASE", "phase"),
    ("IMAGINARY", "imaginary"),
    ("IMAG", "imaginary"),
    ("MAGNITUDE", "magnitude"),
    ("MAG", "magnitude"),
    ("REAL", "real"),
)
# DICOM MR Image Type value 3 (and the same letters in Complex Image
# Component): Siemens and Philips write M/P/R/I, not the full words.
_MR_VALUE3 = {
    "M": "magnitude",
    "P": "phase",
    "R": "real",
    "I": "imaginary",
}


class LoadedFrame:
    """One real or imaginary image, plus an optional header for TE and clocks.

    ``dataset`` is either a pydicom dataset or a small header with the same
    attribute names.  ``get`` lets CEM43 read Acquisition Time without a
    special case: a header that has no clock returns the default, and the
    dose module keeps its missing-time refusal.
    """

    def __init__(self, array, dataset=None, name=""):
        self.array = np.array(array, dtype=float, copy=True)
        self.dataset = dataset
        self.name = name or "in-memory frame"

    def get(self, tag, default=None):
        dataset = self.dataset
        if dataset is None or not hasattr(dataset, "get"):
            return default
        return dataset.get(tag, default)

    def __getattr__(self, name):
        dataset = self.__dict__.get("dataset")
        if dataset is not None and hasattr(dataset, name):
            return getattr(dataset, name)
        raise AttributeError(name)

    def __str__(self):
        return self.name


class ScalarHeader:
    """TE, B0, and spacing carried by a NIfTI file that has no DICOM tags."""

    def __init__(self, te=None, b0=None, pixel_spacing=None):
        if te is not None:
            self.EchoTime = float(te)
        if b0 is not None:
            self.MagneticFieldStrength = float(b0)
        if pixel_spacing is not None:
            self.PixelSpacing = pixel_spacing

    def get(self, tag, default=None):
        return default


@dataclass
class ImportResult:
    real_sources: list = field(default_factory=list)
    imag_sources: list = field(default_factory=list)
    b0_paths: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    problems: list = field(default_factory=list)


@dataclass
class _DicomItem:
    path: str
    filename: str
    kind: str
    sort_key: tuple
    order: int
    image_type: str
    component: str


def import_sequences(paths, component_pair=None):
    """Read files and folders into aligned real/imag sources.

    ``component_pair`` is ``real_imag``, ``magnitude_phase``, or None.
    None follows DICOM tags and filenames.  The choice is only used for
    NIfTI whose names do not already say which component is which.
    """

    result = ImportResult()
    files, missing = _collect_files(paths)
    result.missing.extend(missing)
    dicom_items = []
    nifti_real = []
    nifti_imag = []
    loose = []
    nifti_names = []
    for order, path in enumerate(files):
        if _is_nifti(path):
            before = len(nifti_real) + len(nifti_imag) + len(loose)
            _consume_nifti(
                path, component_pair, nifti_real, nifti_imag, loose, result
            )
            if len(nifti_real) + len(nifti_imag) + len(loose) > before:
                nifti_names.append(os.path.basename(str(path)))
            continue
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            result.skipped.append(os.path.basename(str(path)))
            continue
        filename = os.path.basename(str(path))
        image_type, component = _dicom_type_text(dataset)
        description = getattr(dataset, "SeriesDescription", "") or ""
        kind = _classify_dicom(image_type, component, filename, description)
        if kind is None:
            result.problems.append(
                _unrecognized_message(os.path.basename(str(path)), image_type, component)
            )
            continue
        if kind == "b0":
            result.b0_paths.append(str(path))
            continue
        dicom_items.append(
            _DicomItem(
                path=str(path),
                filename=os.path.basename(str(path)),
                kind=kind,
                sort_key=_dicom_sort_key(dataset),
                order=order,
                image_type=image_type,
                component=component,
            )
        )
    real_paths, imag_paths, problems = _dicom_sources(dicom_items)
    result.problems.extend(problems)
    if loose:
        paired_real, paired_imag, problems = _pair_loose_nifti(loose, component_pair)
        result.problems.extend(problems)
        nifti_real.extend(paired_real)
        nifti_imag.extend(paired_imag)
    if (nifti_real or nifti_imag) and (real_paths or imag_paths):
        dicom_names = ", ".join(
            sorted({item.filename for item in dicom_items if item.kind != "b0"})
        )
        result.problems.append(
            f"DICOM ({dicom_names}) and NIfTI ({', '.join(nifti_names)}) were "
            "selected together. They were not combined into one time series. "
            "Import one source at a time."
        )
        return result
    result.real_sources = nifti_real + real_paths
    result.imag_sources = nifti_imag + imag_paths
    return result


def phase_to_radians(pixels, dataset, filename):
    """Phase in radians.  Raw integer codes are not left as radians.

    A Rescale Slope / Intercept other than 1 and 0 is applied and then
    accepted only when the result is still inside [−π, π].  A slope that
    leaves values far outside that range is rejected, not wrapped by cos/sin.

    Without that scale, unsigned integer codes inside 0–4095 (and above π)
    map with ``code / 4095 * 2π − π``, so 0 → −π and 4095 → +π.  This is not
    the vendor ``2π/4096`` rescale: that one puts code 4095 about 0.00077 rad
    under +π.  The difference does not matter for PRF temperature.  Signed
    12-bit codes (−2048…2047) are not mapped.  Floats already inside
    [−π, π] are kept.
    """

    values = np.asarray(pixels, dtype=float)
    slope, intercept, present = _rescale(dataset)
    identity = (not present) or (slope == 1.0 and intercept == 0.0)
    if present:
        values = values * slope + intercept
    if not identity:
        if _within_radians(values):
            return values
        low = float(np.min(values)) if values.size else float("nan")
        high = float(np.max(values)) if values.size else float("nan")
        raise ValueError(
            f"{filename}: rescaled phase range {low:g}..{high:g} is outside "
            "[-pi, pi]. Rescale Slope/Intercept was not treated as radians."
        )
    if _integer_like(values):
        if values.size == 0:
            return values
        low = float(np.min(values))
        high = float(np.max(values))
        if low >= 0.0 and high <= _PHASE_CODE_MAX and high > np.pi:
            return (values / _PHASE_CODE_MAX) * (2.0 * np.pi) - np.pi
        raise ValueError(
            f"{filename}: phase codes {low:g}..{high:g} are integers and have "
            "no Rescale Slope/Intercept. Only unsigned codes 0..4095 are "
            "mapped to -pi..pi. Raw integers are not treated as radians."
        )
    if _within_radians(values):
        return values
    low = float(np.min(values)) if values.size else float("nan")
    high = float(np.max(values)) if values.size else float("nan")
    raise ValueError(
        f"{filename}: phase range {low:g}..{high:g} is not in radians and was "
        "not scaled. Raw values are not treated as radians."
    )


def _consume_nifti(path, component_pair, nifti_real, nifti_imag, loose, result):
    filename = os.path.basename(str(path))
    try:
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
    except Exception:
        result.skipped.append(filename)
        return
    try:
        header = _nifti_header(image)
    except ValueError as exc:
        result.problems.append(f"{filename}: {exc}")
        header = None
    try:
        if np.iscomplexobj(array) or getattr(array, "ndim", 0) == 4:
            frames = _frames_from_series(array, filename, component_pair, header)
            nifti_real.extend(frame[0] for frame in frames)
            nifti_imag.extend(frame[1] for frame in frames)
            return
        image2d = _as_2d(array, filename)
    except ValueError as exc:
        result.problems.append(str(exc))
        return
    loose.append(
        {
            "kind": _nifti_file_kind(filename),
            "array": image2d,
            "filename": filename,
            "header": header,
        }
    )


def _frames_from_series(array, filename, component_pair, header):
    """Time-major frames.  Volume 0 is the baseline of this file."""

    if np.iscomplexobj(array):
        volumes = _complex_volumes(array, filename)
        frames = []
        for index, volume in enumerate(volumes):
            real = np.real(volume).astype(float)
            imag = np.imag(volume).astype(float)
            name = f"{filename} volume {index}"
            frames.append(
                (
                    LoadedFrame(real, header, name=name + " real"),
                    LoadedFrame(imag, header, name=name + " imag"),
                )
            )
        return frames
    time_major = _time_major_components(np.asarray(array), filename)
    kind = _series_component_kind(filename, component_pair)
    frames = []
    for index in range(time_major.shape[0]):
        first = np.asarray(time_major[index, 0], dtype=float)
        second = np.asarray(time_major[index, 1], dtype=float)
        real, imag = _component_pair_to_complex(
            first, second, kind, f"{filename} volume {index}"
        )
        name = f"{filename} volume {index}"
        frames.append(
            (
                LoadedFrame(real, header, name=name + " real"),
                LoadedFrame(imag, header, name=name + " imag"),
            )
        )
    return frames


def _complex_volumes(array, filename):
    arr = np.squeeze(np.asarray(array))
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(
        f"{filename}: complex NIfTI shape {tuple(np.asarray(array).shape)} is not "
        "a single-slice time series. Volume 0 would be the baseline."
    )


def _time_major_components(array, filename):
    """Return (time, 2, row, col).  Axis length 2 is the component axis.

    SimpleITK's array is (time, component, row, col) when the file was
    written that way.  If axis 1 has length 2 it wins, even when axis 0 is
    also 2.  A 2-volume × 2-slice magnitude series with shape (2, 2, H, W)
    is therefore read as two real/imag frames, not two slices.  Name the
    file or pass ``component_pair`` when that default is wrong.
    """

    if array.ndim != 4:
        raise ValueError(
            f"{filename}: expected a 4D NIfTI time series, got shape {tuple(array.shape)}."
        )
    if array.shape[1] == 2:
        time_major = array
    elif array.shape[0] == 2:
        time_major = np.moveaxis(array, 0, 1)
    else:
        raise ValueError(
            f"{filename}: 4D NIfTI shape {tuple(array.shape)} has no component "
            "axis of length 2 for real/imag or magnitude/phase. Volume 0 is "
            "the baseline only when that axis is present."
        )
    if time_major.shape[0] < 1:
        raise ValueError(f"{filename}: NIfTI time series has no volumes.")
    return time_major


def _series_component_kind(filename, component_pair):
    text_kind = _kind_from_text(filename)
    if text_kind in ("magnitude", "phase"):
        return "magnitude_phase"
    if text_kind in ("real", "imaginary"):
        return "real_imag"
    if component_pair in ("real_imag", "magnitude_phase"):
        return component_pair
    return "real_imag"


def _component_pair_to_complex(first, second, kind, filename):
    if first.shape != second.shape:
        raise ValueError(
            f"{filename}: component shapes {first.shape} and {second.shape} differ."
        )
    if kind == "magnitude_phase":
        phase = phase_to_radians(second, None, filename)
        magnitude = np.asarray(first, dtype=float)
        return magnitude * np.cos(phase), magnitude * np.sin(phase)
    return np.asarray(first, dtype=float), np.asarray(second, dtype=float)


def _pair_loose_nifti(loose, component_pair):
    """Pair 3D NIfTI volumes into one complex frame each."""

    named = [item for item in loose if item["kind"] is not None]
    unnamed = [item for item in loose if item["kind"] is None]
    problems = []
    real_sources = []
    imag_sources = []
    if named:
        family_problems = _append_named_nifti(named, real_sources, imag_sources)
        problems.extend(family_problems)
    if unnamed:
        if component_pair not in ("real_imag", "magnitude_phase"):
            names = ", ".join(item["filename"] for item in unnamed)
            problems.append(
                f"{names}: NIfTI component is not real/imag or magnitude/phase. "
                "Rename the file or choose a pair type when importing. "
                f"ImageType (0008,0008)=absent; Complex Image Component (0008,9208)=absent."
            )
        elif len(unnamed) != 2:
            names = ", ".join(item["filename"] for item in unnamed)
            problems.append(
                f"{names}: choosing {component_pair} needs exactly two NIfTI "
                "volumes when the filenames do not name the component."
            )
        else:
            ordered = sorted(unnamed, key=lambda item: item["filename"])
            first, second = ordered
            try:
                real, imag = _component_pair_to_complex(
                    first["array"],
                    second["array"],
                    component_pair,
                    f"{first['filename']} + {second['filename']}",
                )
            except ValueError as exc:
                problems.append(str(exc))
            else:
                header = first["header"] or second["header"]
                real_sources.append(LoadedFrame(real, header, name=first["filename"]))
                imag_sources.append(LoadedFrame(imag, header, name=second["filename"]))
    return real_sources, imag_sources, problems


def _append_named_nifti(named, real_sources, imag_sources):
    problems = []
    reals = [item for item in named if item["kind"] == "real"]
    imags = [item for item in named if item["kind"] == "imaginary"]
    mags = [item for item in named if item["kind"] == "magnitude"]
    phases = [item for item in named if item["kind"] == "phase"]
    if (reals or imags) and (mags or phases):
        names = ", ".join(item["filename"] for item in mags + phases)
        problems.append(
            f"{names}: magnitude/phase NIfTI was not mixed into a "
            "real/imaginary series."
        )
        _zip_components(reals, imags, "real_imag", real_sources, imag_sources, problems)
        return problems
    if reals or imags:
        _zip_components(reals, imags, "real_imag", real_sources, imag_sources, problems)
    else:
        _zip_components(
            mags, phases, "magnitude_phase", real_sources, imag_sources, problems
        )
    return problems


def _zip_components(first_items, second_items, kind, real_sources, imag_sources, problems):
    first_items = sorted(first_items, key=lambda item: item["filename"])
    second_items = sorted(second_items, key=lambda item: item["filename"])
    count = min(len(first_items), len(second_items))
    for index in range(count):
        first = first_items[index]
        second = second_items[index]
        try:
            real, imag = _component_pair_to_complex(
                first["array"],
                second["array"],
                kind,
                f"{first['filename']} + {second['filename']}",
            )
        except ValueError as exc:
            problems.append(str(exc))
            continue
        header = first["header"] or second["header"]
        real_sources.append(LoadedFrame(real, header, name=first["filename"]))
        imag_sources.append(LoadedFrame(imag, header, name=second["filename"]))
    extra = first_items[count:] + second_items[count:]
    for item in extra:
        problems.append(
            f"{item['filename']}: unpaired NIfTI {item['kind']} volume."
        )


def _dicom_sources(items):
    reals = [item for item in items if item.kind == "real"]
    imags = [item for item in items if item.kind == "imaginary"]
    mags = [item for item in items if item.kind == "magnitude"]
    phases = [item for item in items if item.kind == "phase"]
    problems = []
    if reals or imags:
        if mags or phases:
            names = ", ".join(item.filename for item in mags + phases)
            problems.append(
                f"{names}: magnitude/phase DICOM was not mixed into a "
                "real/imaginary series."
            )
        return (
            [item.path for item in _sort_items(reals)],
            [item.path for item in _sort_items(imags)],
            problems,
        )
    return _magnitude_phase_sources(_sort_items(mags), _sort_items(phases), problems)


def _magnitude_phase_sources(mags, phases, problems):
    real_sources = []
    imag_sources = []
    count = min(len(mags), len(phases))
    for index in range(count):
        mag = mags[index]
        phase = phases[index]
        try:
            mag_ds = pydicom.dcmread(mag.path)
            phase_ds = pydicom.dcmread(phase.path)
            magnitude = _apply_rescale(mag_ds.pixel_array, mag_ds)
            radians = phase_to_radians(phase_ds.pixel_array, phase_ds, phase.filename)
        except Exception as exc:
            problems.append(f"{phase.filename}: {exc}" if phase.filename not in str(exc) else str(exc))
            continue
        if magnitude.shape != radians.shape:
            problems.append(
                f"{mag.filename} and {phase.filename}: magnitude shape "
                f"{magnitude.shape} does not match phase shape {radians.shape}."
            )
            continue
        real_sources.append(
            LoadedFrame(magnitude * np.cos(radians), mag_ds, name=mag.filename)
        )
        imag_sources.append(
            LoadedFrame(magnitude * np.sin(radians), phase_ds, name=phase.filename)
        )
    for item in mags[count:]:
        problems.append(
            f"{item.filename}: unpaired magnitude DICOM. ImageType "
            f"(0008,0008)={item.image_type or 'absent'}; Complex Image Component "
            f"(0008,9208)={item.component or 'absent'}."
        )
    for item in phases[count:]:
        problems.append(
            f"{item.filename}: unpaired phase DICOM. ImageType "
            f"(0008,0008)={item.image_type or 'absent'}; Complex Image Component "
            f"(0008,9208)={item.component or 'absent'}."
        )
    return real_sources, imag_sources, problems


def _sort_items(items):
    """Dated clocks, then time-of-day clocks, then files with no clock."""
    return sorted(items, key=lambda item: (item.sort_key, item.order))


def _dicom_sort_key(dataset):
    """Sort key ``(group, seconds)``.  Group 0 is dated, 1 is time-only, 2 is missing.

    AcquisitionDateTime (YYYYMMDDHHMMSS) and AcquisitionDate+AcquisitionTime
    become seconds from a date ordinal.  A bare AcquisitionTime is seconds
    from midnight and sorts after every dated clock, not by string order
    against a YYYYMMDD prefix.  SeriesDate/SeriesTime are the last fallback.
    Unparseable clocks sort last and do not fail the import.
    """

    datetime_text = _tag_text(dataset, ACQUISITION_DATETIME_TAG)
    if not datetime_text and getattr(dataset, "AcquisitionDateTime", None):
        datetime_text = str(dataset.AcquisitionDateTime)
    absolute = _parse_dt_seconds(datetime_text)
    if absolute is not None:
        return (0, absolute)
    dated = _dated_clock(
        getattr(dataset, "AcquisitionDate", None),
        getattr(dataset, "AcquisitionTime", None),
    )
    if dated is not None:
        return dated
    dated = _dated_clock(
        getattr(dataset, "SeriesDate", None),
        getattr(dataset, "SeriesTime", None),
    )
    if dated is not None:
        return dated
    return (2, 0.0)


def _unrecognized_message(filename, image_type, component):
    return (
        f"{filename}: ImageType (0008,0008)={image_type or 'absent'}; "
        f"Complex Image Component (0008,9208)={component or 'absent'}."
    )


def _dicom_type_text(dataset):
    image_type = _tag_text(dataset, IMAGE_TYPE_TAG)
    if not image_type and getattr(dataset, "ImageType", None):
        value = dataset.ImageType
        if isinstance(value, (list, tuple)):
            image_type = " ".join(str(part) for part in value)
        else:
            image_type = str(value)
    component = _tag_text(dataset, COMPLEX_COMPONENT_TAG)
    if not component and getattr(dataset, "ComplexImageComponent", None):
        component = str(dataset.ComplexImageComponent)
    return image_type.strip(), component.strip()


def _tag_text(dataset, tag):
    if tag not in dataset:
        return ""
    value = dataset[tag].value
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value)
    return str(value).strip()


def _tokens(text):
    tokens = set(re.split(r"[^A-Z0-9]+", (text or "").upper()))
    tokens.discard("")
    return tokens


def _kind_from_text(text):
    tokens = _tokens(text)
    for token, kind in _COMPONENT_TOKENS:
        if token in tokens:
            return kind
    return None


def _kind_from_value3(image_type, component):
    """Single-letter MR Image Type value 3, or a one-letter complex component.

    Filenames are not checked.  ``ORIGINAL\\PRIMARY\\M`` is magnitude;
    ``MPR`` is not, because the third value must be exactly M, P, R, or I.
    """

    parts = [part for part in re.split(r"[^A-Za-z0-9]+", image_type or "") if part]
    if len(parts) >= 3 and parts[2].upper() in _MR_VALUE3:
        return _MR_VALUE3[parts[2].upper()]
    component_text = (component or "").strip().upper()
    if component_text in _MR_VALUE3:
        return _MR_VALUE3[component_text]
    return None


def _classify_dicom(image_type, component, filename, description):
    """Component first, including M/P/R/I.  B0 is only a whole-word fallback.

    ``MAP`` inside a magnitude or phase description must not win.  A
    description that itself says MAGNITUDE or PHASE is that component even
    when Image Type is empty.  ``MAP`` matches only as its own token, so
    ``MAPPING`` is not a B0 map.
    """

    tagged = f"{image_type} {component}"
    kind = _kind_from_text(tagged)
    if kind is None:
        kind = _kind_from_value3(image_type, component)
    if kind is None:
        kind = _kind_from_text(filename)
    if kind is None:
        kind = _kind_from_text(description)
    if kind is None and _tokens(f"{description} {filename}") & {"B0", "FIELD", "MAP"}:
        kind = "b0"
    return kind


def _nifti_file_kind(filename):
    return _kind_from_text(filename)


def _rescale(dataset):
    if dataset is None:
        return 1.0, 0.0, False
    slope = getattr(dataset, "RescaleSlope", None)
    intercept = getattr(dataset, "RescaleIntercept", None)
    if slope is None and intercept is None:
        return 1.0, 0.0, False
    slope_value = 1.0 if slope is None else float(slope)
    intercept_value = 0.0 if intercept is None else float(intercept)
    return slope_value, intercept_value, True


def _apply_rescale(pixels, dataset):
    values = np.asarray(pixels, dtype=float)
    slope, intercept, present = _rescale(dataset)
    if not present:
        return values
    return values * slope + intercept


def _integer_like(values):
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False
    return bool(np.all(np.abs(values - np.round(values)) <= 1e-6))


def _within_radians(values):
    if values.size == 0:
        return True
    if not np.all(np.isfinite(values)):
        return False
    return float(np.max(np.abs(values))) <= _RADIAN_LIMIT


def _as_2d(array, filename):
    squeezed = np.squeeze(np.asarray(array))
    if squeezed.ndim == 2:
        return np.asarray(squeezed, dtype=float)
    raise ValueError(
        f"{filename}: expected one 2D image (a 3D NIfTI may use a singleton "
        f"slice), got shape {tuple(np.asarray(array).shape)}."
    )


def _nifti_header(image):
    te, b0 = _parse_te_b0(_nifti_note(image))
    spacing = image.GetSpacing()
    pixel_spacing = None
    if len(spacing) >= 2 and float(spacing[0]) > 0.0 and float(spacing[1]) > 0.0:
        pixel_spacing = (float(spacing[1]), float(spacing[0]))
    if te is None and b0 is None and pixel_spacing is None:
        return None
    return ScalarHeader(te=te, b0=b0, pixel_spacing=pixel_spacing)


def _nifti_note(image):
    parts = []
    for key in ("ITK_FileNotes", "aux_file", "descrip"):
        try:
            if key in image.GetMetaDataKeys():
                parts.append(str(image.GetMetaData(key)))
        except Exception:
            continue
    return " ".join(parts)


def _parse_te_b0(text):
    te = None
    b0 = None
    if not text:
        return te, b0
    for match in re.finditer(
        r"\b(TE|B0)\s*[:=]\s*([-+0-9.eE]+)", text, flags=re.IGNORECASE
    ):
        try:
            value = float(match.group(2))
        except ValueError as exc:
            raise ValueError(
                f"NIfTI note {match.group(0)!r} is not a TE or B0 number."
            ) from exc
        if match.group(1).upper() == "TE" and te is None:
            te = value
        elif match.group(1).upper() == "B0" and b0 is None:
            b0 = value
    return te, b0


def _collect_files(paths):
    files = []
    missing = []
    seen = set()

    def add_file(path):
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            return
        seen.add(key)
        files.append(path)

    for raw in paths or []:
        if raw is None:
            continue
        path = os.fspath(raw)
        if os.path.isdir(path):
            for root, dirs, names in os.walk(path):
                dirs.sort()
                for name in sorted(names):
                    add_file(os.path.join(root, name))
            continue
        if os.path.isfile(path):
            add_file(path)
            continue
        missing.append(os.path.basename(path) or path)
    return files, missing


def _parse_tm_seconds(value):
    text = str(value or "").strip()
    if not text:
        return None
    fraction = 0.0
    whole = text
    if "." in text:
        whole, frac = text.split(".", 1)
        if frac == "" or not frac.isdigit():
            return None
        fraction = float("0." + frac)
    if not whole.isdigit() or len(whole) not in (2, 4, 6):
        return None
    hour = int(whole[0:2])
    minute = int(whole[2:4]) if len(whole) >= 4 else 0
    second = int(whole[4:6]) if len(whole) == 6 else 0
    if hour > 23 or minute > 59 or second >= 61:
        return None
    return hour * 3600.0 + minute * 60.0 + second + fraction


def _parse_da_ordinal(value):
    text = str(value or "").strip()
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return date(int(text[0:4]), int(text[4:6]), int(text[6:8])).toordinal()
    except ValueError:
        return None


def _parse_dt_seconds(value):
    text = str(value or "").strip()
    if not text:
        return None
    fraction = 0.0
    whole = text
    if "." in text:
        whole, frac = text.split(".", 1)
        if frac and not frac.isdigit():
            return None
        if frac:
            fraction = float("0." + frac)
    if not whole.isdigit() or len(whole) < 8 or len(whole) > 14 or len(whole) % 2 != 0:
        return None
    ordinal = _parse_da_ordinal(whole[0:8])
    if ordinal is None:
        return None
    seconds = 0.0
    if len(whole) > 8:
        seconds = _parse_tm_seconds(whole[8:])
        if seconds is None:
            return None
    return ordinal * 86400.0 + seconds + fraction


def _dated_clock(date_value, time_value):
    if not date_value and not time_value:
        return None
    seconds = _parse_tm_seconds(time_value)
    if seconds is None:
        return None
    ordinal = _parse_da_ordinal(date_value) if date_value else None
    if ordinal is None:
        return (1, seconds)
    return (0, ordinal * 86400.0 + seconds)


def _is_nifti(path):
    name = os.path.basename(str(path)).lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")
