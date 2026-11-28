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

import numpy as np
import pydicom
import SimpleITK as sitk

IMAGE_TYPE_TAG = (0x0008, 0x0008)
COMPLEX_COMPONENT_TAG = (0x0008, 0x9208)
ACQUISITION_DATETIME_TAG = (0x0008, 0x002A)

# 0 and 4095 are the ends of a 12-bit phase code.  4096 is not the top code.
_PHASE_CODE_MAX = 4095.0
_COMPONENT_TOKENS = (
    ("PHASE", "phase"),
    ("IMAGINARY", "imaginary"),
    ("IMAG", "imaginary"),
    ("MAGNITUDE", "magnitude"),
    ("MAG", "magnitude"),
    ("REAL", "real"),
)


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
    problems: list = field(default_factory=list)


@dataclass
class _DicomItem:
    path: str
    filename: str
    kind: str
    sort_key: str
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
    result.skipped.extend(missing)
    dicom_items = []
    nifti_real = []
    nifti_imag = []
    loose = []
    for order, path in enumerate(files):
        if _is_nifti(path):
            _consume_nifti(
                path, component_pair, nifti_real, nifti_imag, loose, result
            )
            continue
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            result.skipped.append(os.path.basename(str(path)))
            continue
        image_type, component = _dicom_type_text(dataset)
        kind = _kind_from_text(image_type + " " + component)
        if kind is None:
            kind = _kind_from_text(os.path.basename(str(path)))
        if kind is None and _looks_like_b0(dataset, os.path.basename(str(path))):
            kind = "b0"
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
    result.real_sources = nifti_real + real_paths
    result.imag_sources = nifti_imag + imag_paths
    return result


def phase_to_radians(pixels, dataset, filename):
    """Phase in radians.  Raw 12-bit integers are not left as radians.

    A Rescale Slope / Intercept other than 1 and 0 is the phase in radians.
    Without that scale, integer codes whose range sits inside 0–4095 and
    above π are mapped from 0 → −π and 4095 → π.  Floats already inside
    [−π, π] are kept.  Anything else raises; it is not treated as radians.
    """

    values = np.asarray(pixels, dtype=float)
    slope, intercept, present = _rescale(dataset)
    identity = (not present) or (slope == 1.0 and intercept == 0.0)
    if present:
        values = values * slope + intercept
    if not identity:
        return values
    if _integer_like(values):
        if values.size == 0:
            return values
        low = float(np.min(values))
        high = float(np.max(values))
        if low >= 0.0 and high <= _PHASE_CODE_MAX and high > np.pi:
            return (values / _PHASE_CODE_MAX) * (2.0 * np.pi) - np.pi
        raise ValueError(
            f"{filename}: phase codes {low:g}..{high:g} are integers and have "
            "no Rescale Slope/Intercept. Raw integers are not treated as radians."
        )
    if values.size == 0 or (
        np.all(np.isfinite(values)) and float(np.max(np.abs(values))) <= np.pi + 1e-3
    ):
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
    header = _nifti_header(image)
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
    """Return (time, 2, row, col).  Axis length 2 is the component axis."""

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
        names = ", ".join(item["filename"] for item in named)
        problems.append(
            f"{names}: NIfTI import mixes real/imag with magnitude/phase. "
            "Import one pair type at a time."
        )
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
    decorated = []
    for index, item in enumerate(items):
        missing = item.sort_key is None
        decorated.append((missing, item.sort_key or "", item.order, index, item))
    decorated.sort()
    return [item for _, _, _, _, item in decorated]


def _dicom_sort_key(dataset):
    text = _tag_text(dataset, ACQUISITION_DATETIME_TAG)
    if text:
        return text
    if getattr(dataset, "AcquisitionDateTime", None):
        return str(dataset.AcquisitionDateTime)
    acquisition = getattr(dataset, "AcquisitionTime", None)
    if acquisition:
        date = getattr(dataset, "AcquisitionDate", None)
        return f"{date or ''}{acquisition}"
    series_time = getattr(dataset, "SeriesTime", None)
    if series_time:
        series_date = getattr(dataset, "SeriesDate", None)
        return f"{series_date or ''}{series_time}"
    return None


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


def _kind_from_text(text):
    tokens = set(re.split(r"[^A-Z0-9]+", (text or "").upper()))
    tokens.discard("")
    for token, kind in _COMPONENT_TOKENS:
        if token in tokens:
            return kind
    return None


def _nifti_file_kind(filename):
    return _kind_from_text(filename)


def _looks_like_b0(dataset, filename):
    description = getattr(dataset, "SeriesDescription", "") or ""
    blob = f"{description} {filename}".upper()
    return any(token in blob for token in ("B0", "FIELD", "MAP"))


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
        value = float(match.group(2))
        if match.group(1).upper() == "TE" and te is None:
            te = value
        elif match.group(1).upper() == "B0" and b0 is None:
            b0 = value
    return te, b0


def _collect_files(paths):
    files = []
    missing = []
    for raw in paths or []:
        if raw is None:
            continue
        path = os.fspath(raw)
        if os.path.isdir(path):
            for root, dirs, names in os.walk(path):
                dirs.sort()
                for name in sorted(names):
                    files.append(os.path.join(root, name))
            continue
        if os.path.isfile(path):
            files.append(path)
            continue
        missing.append(os.path.basename(path) or path)
    return files, missing


def _is_nifti(path):
    name = os.path.basename(str(path)).lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")
