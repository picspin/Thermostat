"""Cumulative equivalent minutes at 43°C (CEM43) for a Cinema sequence.

The dose is the Sapareto & Dewey 1984 isoeffect integral, evaluated per voxel:

    CEM43 = Σ Δtᵢ · R^(43 − Tᵢ)

R is 0.25 below 43°C and 0.5 at or above 43°C.  Δtᵢ is in minutes.  This
module does not register images and does not assume equally spaced frames.
A missing Acquisition Time stops the calculation.

The sum is a rectangle.  The whole Δtᵢ is multiplied by R^(43−Tᵢ) at the
temperature at the end of that interval.  Heating inside the interval is
not modeled.

Low-signal voxels are excluded only when the caller passes a validity mask
(the thermometry model records the same 10% magnitude mask it already uses
for PRF).  A missing mask does not mean those voxels are low-SNR; only
non-finite temperatures are then excluded.  Excluded voxels do not add dose
for that frame and are not replaced with a default temperature.  A missing
frame (``None``) adds nothing for its interval; the other frames are kept.

Cinema maps are used as stored.  With motion correction off, each map is the
baseline plus the change between that pair of frames, not temperature versus
the first phase.  This module does not rewrite those maps.  The controller
must not present that number as CEM43; see
``CONSECUTIVE_PAIR_DOSE_MESSAGE``.

An ROI curve uses only voxels that are valid on every frame that has a
temperature.  A voxel that joins late has a shorter history, so it is left
out of both the temperature curve and the cumulative-dose curve.  The map
still keeps every voxel that was valid on at least one frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np
import pydicom

# Sapareto SA, Dewey WC. Int J Radiat Oncol Biol Phys. 1984;10:787-800.
BREAKPOINT_CELSIUS = 43.0
R_BELOW_BREAKPOINT = 0.25
R_AT_OR_ABOVE_BREAKPOINT = 0.5

# Optional display guides only.  van Rhoon et al., Eur Radiol 2013, proposed
# these as MR radiofrequency safety references, not as a treatment endpoint.
REFERENCE_TEMPERATURE_C = 39.0
REFERENCE_CEM43_EQUIVALENT_MINUTES = (2.0, 9.0)
REFERENCE_SOURCE = "van Rhoon et al., Eur Radiol 2013"
REFERENCE_DISCLAIMER = (
    "Optional reference only, from van Rhoon et al., Eur Radiol 2013 "
    "(MR radiofrequency safety recommendations). "
    "These lines are not a treatment endpoint and do not indicate that "
    "treatment is complete."
)

# Above this fraction, a black dose map is no longer self-explanatory.
# The 10% magnitude mask uses the brightest pixel, so one artifact can
# exclude almost the whole image.
HIGH_MASK_EXCLUSION_FRACTION = 0.95

CONSECUTIVE_PAIR_DOSE_MESSAGE = (
    "Motion correction is off. Each Cinema temperature map is the baseline "
    "plus the change between that pair of frames, not absolute temperature "
    "versus the first phase. CEM43 was not shown, because using those maps "
    "would systematically underestimate the dose. Turn motion correction on "
    "so every frame is compared with the first phase, then run the sequence "
    "again. Equal spacing was not assumed."
)

ACQUISITION_TIME_TAG = (0x0008, 0x0032)
ACQUISITION_DATE_TAG = (0x0008, 0x0022)


class ThermalDoseError(ValueError):
    """CEM43 cannot be calculated.

    Equal spacing is never substituted for a missing Acquisition Time.
    """


@dataclass(frozen=True)
class AcquisitionClock:
    """One phase clock taken from DICOM Acquisition Time (0008,0032).

    ``date_ordinal`` comes from Acquisition Date when that tag is present on
    the same image.  It is not read from Series Time or Acquisition DateTime.
    """

    seconds: float
    date_ordinal: Optional[int] = None


@dataclass(frozen=True)
class ThermalDoseResult:
    """Voxel CEM43 map and, when an ROI was supplied, its curves.

    ``elapsed_minutes`` is the time from the first phase's Acquisition Time
    to the end of each frame's interval.  ROI arrays are None when no ROI
    mask was supplied.  Cumulative ROI values are equivalent minutes.
    """

    cem43: np.ndarray
    delta_t_minutes: np.ndarray
    elapsed_minutes: np.ndarray
    roi_mean_temperature: Optional[np.ndarray]
    roi_cumulative_mean: Optional[np.ndarray]
    roi_cumulative_max: Optional[np.ndarray]
    map_max_cem43: float
    # Share of voxels that had a finite temperature and still never
    # accumulated.  With a low-signal mask, that is the mask (or an
    # overflowing rate), not a missing Acquisition Time.
    mask_excluded_fraction: float
    low_signal_mask_applied: bool


def r_factor(temperature):
    """Return the Sapareto & Dewey R at ``temperature`` (°C).

    A scalar input returns a float.  An array returns an array of the same
    shape.  This does not by itself decide whether a voxel is accumulated.
    """

    temp = np.asarray(temperature, dtype=float)
    values = np.where(
        temp < BREAKPOINT_CELSIUS,
        R_BELOW_BREAKPOINT,
        R_AT_OR_ABOVE_BREAKPOINT,
    ).astype(float)
    if values.shape == ():
        return float(values)
    return values


def accumulate_cem43(temperature_maps, delta_t_minutes, valid_masks=None):
    """Accumulate CEM43 and return the final map in equivalent minutes.

    ``temperature_maps`` is one absolute-temperature array per Cinema frame.
    ``None`` means that frame has no usable temperature: it adds nothing.
    ``delta_t_minutes`` must contain one finite, positive interval per frame.
    A NaN interval is a missing time and raises ``ThermalDoseError``.

    Voxels that are never valid are NaN, not zero, so an unaccumulated
    background is not drawn as a real dose of 0.
    """

    return accumulate_thermal_dose(
        temperature_maps, delta_t_minutes, valid_masks=valid_masks
    ).cem43


def accumulate_thermal_dose(
    temperature_maps,
    delta_t_minutes,
    valid_masks=None,
    roi_mask=None,
) -> ThermalDoseResult:
    """Voxel-wise CEM43 plus optional ROI temperature and cumulative dose."""

    maps = list(temperature_maps)
    if not maps:
        raise ThermalDoseError(
            "No temperature maps to accumulate. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        )
    delta_t = _require_intervals(delta_t_minutes, len(maps))
    masks = _align_masks(valid_masks, len(maps))
    shape = _sequence_shape(maps)
    roi = _align_roi(roi_mask, shape)

    running = np.zeros(shape, dtype=float)
    ever_valid = np.zeros(shape, dtype=bool)
    finite_somewhere = np.zeros(shape, dtype=bool)
    frame_valid = [] if roi is not None else None
    mask_applied = False

    for index, temperature in enumerate(maps):
        if temperature is None:
            valid = np.zeros(shape, dtype=bool)
        else:
            observed = np.asarray(temperature, dtype=float)
            if observed.shape != shape:
                raise ThermalDoseError(
                    f"Temperature frame {index + 1} has shape {observed.shape}, "
                    f"expected {shape}."
                )
            finite_somewhere |= np.isfinite(observed)
            if masks[index] is not None:
                mask_applied = True
            valid = _valid_voxels(observed, masks[index])
            increment = np.zeros(shape, dtype=float)
            if np.any(valid):
                # Overflow is not a real dose.  Drop those voxels for this
                # frame instead of storing a zero contribution as valid.
                rate = _rate(observed[valid])
                usable = np.isfinite(rate)
                if not np.all(usable):
                    rejected = np.flatnonzero(valid.ravel())[~usable]
                    valid.ravel()[rejected] = False
                    rate = rate[usable]
                increment[valid] = delta_t[index] * rate
            running = running + increment
            ever_valid |= valid
        if frame_valid is not None:
            # Copy after overflow rejection, so the ROI cohort matches the map.
            frame_valid.append(np.array(valid, copy=True))

    cem43 = np.where(ever_valid, running, np.nan)
    finite = cem43[np.isfinite(cem43)]
    map_max = float(np.max(finite)) if finite.size else float("nan")
    elapsed = np.cumsum(delta_t)
    n_voxels = float(ever_valid.size)
    # Finite somewhere, but never accumulated: the mask or an overflowing rate.
    mask_excluded_fraction = float(
        np.count_nonzero(finite_somewhere & ~ever_valid) / n_voxels
    )
    if roi is None:
        roi_temperature = None
        roi_mean = None
        roi_max = None
    else:
        roi_temperature, roi_mean, roi_max = _roi_curves(
            maps, delta_t, frame_valid, roi
        )
    return ThermalDoseResult(
        cem43=cem43,
        delta_t_minutes=delta_t,
        elapsed_minutes=elapsed,
        roi_mean_temperature=roi_temperature,
        roi_cumulative_mean=roi_mean,
        roi_cumulative_max=roi_max,
        map_max_cem43=map_max,
        mask_excluded_fraction=mask_excluded_fraction,
        low_signal_mask_applied=mask_applied,
    )


def calculate_cinema_cem43(
    real_sources,
    temperature_maps,
    imag_sources=None,
    valid_masks=None,
    roi_mask=None,
) -> ThermalDoseResult:
    """CEM43 for a Cinema sequence using each phase's Acquisition Time.

    ``real_sources`` is the phase order used by Cinema (one entry per real
    image, including the baseline).  There must be one more phase than
    temperature frame.  When a real image has no Acquisition Time, the
    imaginary image at the same index is tried.  Series Time and
    Acquisition DateTime are not used.  A missing or non-positive interval
    raises ``ThermalDoseError`` instead of inventing equal spacing.
    """

    maps = list(temperature_maps)
    clocks = clocks_for_phases(real_sources, len(maps) + 1, imag_sources)
    delta_t = intervals_minutes(clocks)
    return accumulate_thermal_dose(
        maps,
        delta_t,
        valid_masks=valid_masks,
        roi_mask=roi_mask,
    )


def clocks_for_phases(real_sources, n_phases, imag_sources=None):
    """Read Acquisition Time for the first ``n_phases`` phases."""

    if n_phases < 2:
        raise ThermalDoseError(
            "At least two phases with AcquisitionTime are required. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        )
    if real_sources is None or len(real_sources) < n_phases:
        found = 0 if not real_sources else len(real_sources)
        raise ThermalDoseError(
            f"Need {n_phases} real images to read AcquisitionTime, "
            f"found {found} files. This is a file count, not a count of "
            "phases that already have AcquisitionTime. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        )
    clocks = []
    for index in range(n_phases):
        imag_source = None
        if imag_sources is not None and index < len(imag_sources):
            imag_source = imag_sources[index]
        try:
            clock = read_acquisition_clock(real_sources[index])
            if clock is None and imag_source is not None:
                clock = read_acquisition_clock(imag_source)
        except ThermalDoseError:
            raise
        except Exception as exc:
            raise ThermalDoseError(
                f"Could not read AcquisitionTime for phase {index + 1}: {exc}"
            ) from exc
        clocks.append(clock)
    return clocks


def intervals_minutes(clocks: Sequence[Optional[AcquisitionClock]]) -> np.ndarray:
    """Minutes between consecutive phase clocks.

    A missing clock or a non-positive interval is an error.  A backwards
    time-of-day is not treated as midnight, because that would invent nearly
    a day of dose.  Put Acquisition Date on every phase if the series crosses
    midnight.  Dates on only some phases are also an error.
    """

    if len(clocks) < 2:
        raise ThermalDoseError(
            "At least two AcquisitionTime values are required. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        )
    missing = [index + 1 for index, clock in enumerate(clocks) if clock is None]
    if missing:
        listed = ", ".join(str(index) for index in missing)
        raise ThermalDoseError(
            "AcquisitionTime (0008,0032) is missing for phase "
            f"{listed}. CEM43 was not calculated, and equal spacing was not assumed."
        )
    dated = [clock.date_ordinal is not None for clock in clocks]
    if any(dated) and not all(dated):
        raise ThermalDoseError(
            "AcquisitionDate is present on some phases but missing on others. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        )
    use_dates = all(dated)
    absolute = []
    for clock in clocks:
        if use_dates:
            absolute.append(clock.date_ordinal * 86400.0 + clock.seconds)
        else:
            absolute.append(clock.seconds)
    deltas = []
    for index, (start, end) in enumerate(zip(absolute, absolute[1:])):
        seconds = float(end - start)
        if not np.isfinite(seconds) or seconds <= 0.0:
            raise ThermalDoseError(
                "AcquisitionTime interval between phase "
                f"{index + 1} and {index + 2} is not positive. "
                "CEM43 was not calculated, and equal spacing was not assumed."
            )
        deltas.append(seconds / 60.0)
    return np.asarray(deltas, dtype=float)


def read_acquisition_clock(source) -> Optional[AcquisitionClock]:
    """Return Acquisition Time for a path or a dataset.

    ``None`` means the Acquisition Time tag is absent or empty.  An
    unreadable file, or a tag that is present but not a DICOM TM, raises
    ``ThermalDoseError``.  Series Time is ignored even if it is present.
    """

    if source is None:
        return None
    dataset = _as_dataset(source)
    raw_time = _tag_value(dataset, "AcquisitionTime", ACQUISITION_TIME_TAG)
    if raw_time is None:
        return None
    seconds = parse_acquisition_time(raw_time)
    if seconds is None:
        return None
    raw_date = _tag_value(dataset, "AcquisitionDate", ACQUISITION_DATE_TAG)
    ordinal = None if raw_date is None else parse_acquisition_date(raw_date)
    return AcquisitionClock(seconds=seconds, date_ordinal=ordinal)


def parse_acquisition_time(value) -> Optional[float]:
    """Parse a DICOM TM (HH, HHMM, or HHMMSS, optional fraction) to seconds.

    A blank value is missing and returns None.  A non-blank value that is
    not a TM raises ``ThermalDoseError``.
    """

    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(text)
            hour = int(parts[0])
            minute = int(parts[1])
            second = float(parts[2]) if len(parts) == 3 else 0.0
        else:
            fraction = 0.0
            whole = text
            if "." in text:
                whole, frac = text.split(".", 1)
                # DICOM puts the fraction on HHMMSS only.  "10.5" is not a TM.
                if len(whole) != 6 or frac == "" or not frac.isdigit():
                    raise ValueError(text)
                fraction = float("0." + frac)
            if not whole.isdigit() or len(whole) not in (2, 4, 6):
                raise ValueError(text)
            hour = int(whole[0:2])
            minute = int(whole[2:4]) if len(whole) >= 4 else 0
            second = float(int(whole[4:6]) if len(whole) == 6 else 0) + fraction
    except (TypeError, ValueError) as exc:
        raise ThermalDoseError(
            f"AcquisitionTime {text!r} is not a DICOM TM value. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        ) from exc
    if hour < 0 or hour > 23 or minute < 0 or minute > 59 or second < 0.0 or second >= 61.0:
        raise ThermalDoseError(
            f"AcquisitionTime {text!r} is outside the DICOM TM range. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        )
    return hour * 3600.0 + minute * 60.0 + second


def parse_acquisition_date(value) -> Optional[int]:
    """Parse a DICOM DA (YYYYMMDD) to a date ordinal.  Blank returns None."""

    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if len(text) != 8 or not text.isdigit():
        raise ThermalDoseError(
            f"AcquisitionDate {text!r} is not a DICOM DA value. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        )
    try:
        ordinal = date(int(text[0:4]), int(text[4:6]), int(text[6:8])).toordinal()
    except ValueError as exc:
        raise ThermalDoseError(
            f"AcquisitionDate {text!r} is not a real calendar date. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        ) from exc
    return ordinal


def format_dose_summary(result: ThermalDoseResult) -> str:
    """One status sentence for the dose dialog.  No treatment-completion claim."""

    if np.isfinite(result.map_max_cem43):
        summary = (
            f"Map maximum: {result.map_max_cem43:.3f} CEM43 (equivalent minutes)."
        )
    else:
        summary = "Map maximum: no valid voxels were accumulated."
    if result.roi_cumulative_mean is None:
        summary += " ROI temperature curve and cumulative dose need an ROI on the main view."
    elif not np.any(np.isfinite(result.roi_cumulative_mean)):
        summary += " The ROI has no valid temperature voxels, so no dose was accumulated there."
    else:
        summary += (
            " ROI cumulative dose: mean "
            f"{result.roi_cumulative_mean[-1]:.3f} CEM43, max "
            f"{result.roi_cumulative_max[-1]:.3f} CEM43."
            " The ROI curves count only voxels valid on every frame."
        )
    note = mask_exclusion_note(result)
    if note:
        summary += " " + note
    return summary


def mask_exclusion_note(result: ThermalDoseResult):
    """Why a nearly black dose map happened, or None when it did not.

    The thermometry mask is ``magnitude > 0.1 * max(magnitude)``.  One bright
    pixel raises that line.  The dose map then goes black while the
    temperature values are still finite.  Callers should say so instead of
    leaving "no valid voxels" unexplained.
    """

    if not result.low_signal_mask_applied:
        return None
    if result.mask_excluded_fraction <= HIGH_MASK_EXCLUSION_FRACTION:
        return None
    percent = 100.0 * result.mask_excluded_fraction
    return (
        f"{percent:.1f}% of voxels had a finite temperature but were not "
        "accumulated. One possible cause is a single bright pixel or artifact "
        "raising the 10% magnitude threshold until almost the whole map is "
        "excluded. Black is not a dose of zero."
    )


def _rate(temperature: np.ndarray) -> np.ndarray:
    """R^(43−T).  Non-finite results stay NaN so the caller can reject them."""

    temp = np.asarray(temperature, dtype=float)
    ratio = r_factor(temp)
    with np.errstate(over="ignore", invalid="ignore"):
        rate = np.power(ratio, BREAKPOINT_CELSIUS - temp)
    return np.asarray(rate, dtype=float)


def _valid_voxels(temperature: np.ndarray, mask) -> np.ndarray:
    valid = np.isfinite(temperature)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != temperature.shape:
            raise ThermalDoseError(
                f"Validity mask shape {mask_array.shape} does not match "
                f"temperature shape {temperature.shape}."
            )
        valid &= mask_array
    return valid


def _roi_curves(maps, delta_t, frame_valid, roi):
    """Temperature and cumulative dose for one stable ROI cohort.

    The cohort is the ROI intersection of voxels valid on every frame that
    has a temperature.  A missing frame does not shrink it.  Both curves use
    that same set, so a voxel with only part of the history cannot pull the
    mean down.  An empty cohort leaves NaN; it does not fall back to "voxels
    valid so far".
    """

    n_frames = len(maps)
    temperatures = np.full(n_frames, np.nan, dtype=float)
    means = np.full(n_frames, np.nan, dtype=float)
    maxima = np.full(n_frames, np.nan, dtype=float)
    cohort = np.array(roi, dtype=bool, copy=True)
    saw_frame = False
    for temperature, valid in zip(maps, frame_valid):
        if temperature is None:
            continue
        cohort &= valid
        saw_frame = True
    if not saw_frame or not np.any(cohort):
        return temperatures, means, maxima
    dose = np.zeros(int(np.count_nonzero(cohort)), dtype=float)
    for index, temperature in enumerate(maps):
        if temperature is not None:
            observed = np.asarray(temperature, dtype=float)[cohort]
            temperatures[index] = float(np.mean(observed))
            rate = _rate(observed)
            usable = np.isfinite(rate)
            step = np.zeros_like(dose)
            step[usable] = delta_t[index] * rate[usable]
            dose = dose + step
        means[index] = float(np.mean(dose))
        maxima[index] = float(np.max(dose))
    return temperatures, means, maxima


def _require_intervals(delta_t_minutes, n_frames) -> np.ndarray:
    try:
        delta_t = np.asarray(delta_t_minutes, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ThermalDoseError(
            "Time intervals are not numeric. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        ) from exc
    if delta_t.shape != (n_frames,):
        raise ThermalDoseError(
            f"Need {n_frames} AcquisitionTime intervals, found {delta_t.size}. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        )
    if not np.all(np.isfinite(delta_t)) or np.any(delta_t <= 0.0):
        raise ThermalDoseError(
            "A time interval is missing or not positive. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        )
    return delta_t


def _align_masks(valid_masks, n_frames):
    if valid_masks is None:
        return [None] * n_frames
    masks = list(valid_masks)
    if len(masks) != n_frames:
        raise ThermalDoseError(
            f"Need {n_frames} validity masks, found {len(masks)}."
        )
    return masks


def _sequence_shape(maps):
    for index, temperature in enumerate(maps):
        if temperature is None:
            continue
        shape = np.shape(temperature)
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise ThermalDoseError(
                f"Temperature frame {index + 1} is not a 2-D map (shape {shape})."
            )
        return shape
    raise ThermalDoseError(
        "Every temperature frame is missing, so CEM43 was not calculated."
    )


def _align_roi(roi_mask, shape):
    if roi_mask is None:
        return None
    roi = np.asarray(roi_mask, dtype=bool)
    if roi.shape != shape:
        raise ThermalDoseError(
            f"ROI mask shape {roi.shape} does not match temperature shape {shape}."
        )
    if not np.any(roi):
        raise ThermalDoseError("The ROI mask is empty, so no ROI dose was calculated.")
    return roi


def _as_dataset(source):
    if not isinstance(source, (str, bytes, os.PathLike)):
        return source
    path = os.fspath(source)
    try:
        return pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=[ACQUISITION_TIME_TAG, ACQUISITION_DATE_TAG],
        )
    except Exception as exc:
        raise ThermalDoseError(
            f"Could not read AcquisitionTime from {path}: {exc}. "
            "CEM43 was not calculated, and equal spacing was not assumed."
        ) from exc


def _tag_value(dataset, keyword, tag):
    """Return one DICOM value, or None when that tag is absent or blank.

    Only the requested tag is read.  A neighbouring time tag must not fill in.
    """

    element = None
    if hasattr(dataset, "get"):
        try:
            element = dataset.get(tag, None)
        except Exception:
            element = None
    if element is None and hasattr(dataset, keyword):
        element = getattr(dataset, keyword)
    if element is None:
        return None
    value = getattr(element, "value", element)
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="ignore")
    if value is None or str(value).strip() == "":
        return None
    return value
