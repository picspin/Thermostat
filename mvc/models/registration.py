"""Volume registration used for motion correction.

This module deliberately has no Qt imports: the registration can be exercised
with plain numpy arrays in headless tests.  The controller is responsible for
deciding what to do with the result and the model is responsible for keeping
the original acquired images intact.

The defaults mirror the historical controller configuration so that the
extraction in this change is behavior preserving.  Algorithmic defaults will
be revisited separately once the motion model / similarity metric / reference
frame have been agreed on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import SimpleITK as sitk


# Public status values for RegistrationResult.status.
STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class RegistrationConfig:
    """Parameters for the SimpleITK registration.

    Attributes:
        metric: ``"mattes_mi"`` (historical default) or ``"correlation"``.
        histogram_bins: Number of histogram bins for Mattes MI.
        sampling_percentage: Fraction of pixels sampled by the metric.
        random_seed: Seed for random sampling.  ``None`` keeps SimpleITK's
            wall-clock seed (the historical behavior).  Note that SimpleITK
            uses ``sitkWallClock == 0``; pass a non-zero value to make a run
            reproducible.
        learning_rate: Gradient-descent learning rate.
        iterations: Maximum optimizer iterations per resolution level.
        convergence_min_value: Convergence checker minimum value.
        convergence_window: Convergence checker window size.
        shrink_factors: Shrink factors per resolution level.
        smoothing_sigmas: Smoothing sigmas per resolution level.
        smoothing_sigmas_in_physical_units: Whether sigmas are in physical
            units (historical behavior: yes).
        mesh_size: B-spline mesh size per dimension.
        interpolator: ``"linear"``, ``"nearest"`` or ``"bspline"``.
        dice_warning_threshold: DICE below this value produces a warning.
    """

    metric: str = "mattes_mi"
    histogram_bins: int = 50
    sampling_percentage: float = 0.1
    random_seed: Optional[int] = None
    learning_rate: float = 1.0
    iterations: int = 100
    convergence_min_value: float = 1e-6
    convergence_window: int = 10
    shrink_factors: Tuple[int, ...] = (4, 2, 1)
    smoothing_sigmas: Tuple[float, ...] = (2.0, 1.0, 0.0)
    smoothing_sigmas_in_physical_units: bool = True
    mesh_size: int = 8
    interpolator: str = "linear"
    dice_warning_threshold: float = 0.9


@dataclass
class RegistrationResult:
    """Outcome of one registration.

    ``warped_real`` / ``warped_imag`` are ``None`` when ``status`` is
    ``"failed"``.  The result is a value object; the registrar never mutates
    the input arrays.
    """

    warped_real: Optional[np.ndarray]
    warped_imag: Optional[np.ndarray]
    transform: Any
    config: RegistrationConfig
    metric_name: str
    final_metric: float
    dice: float
    iterations: int
    stop_condition: str
    elapsed_s: float
    status: str = STATUS_OK
    message: str = ""
    # Diagnostics added after the historical fields.  They are not used to
    # decide the B-spline status, so a default configuration still warps the
    # same arrays and returns the same status as before.
    ncc_before: float = float("nan")
    ncc_after: float = float("nan")
    mae_before: float = float("nan")
    mae_after: float = float("nan")
    displacement_rms: float = float("nan")
    displacement_max: float = float("nan")
    folding_ratio: float = float("nan")

    @property
    def succeeded(self) -> bool:
        return self.status != STATUS_FAILED


class SimpleITKRegistrar:
    """SimpleITK B-spline registrar for 2D/3D motion correction.

    The transform is estimated from the real component and then applied to
    both real and imaginary components.  Estimating one transform for both is
    required for phase-consistent thermometry.
    """

    _METRIC_NAMES = {
        "mattes_mi": "MattesMutualInformation",
        "correlation": "Correlation",
    }

    _INTERPOLATORS = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline,
    }

    def register_complex(
        self,
        fixed_real: np.ndarray,
        fixed_imag: Optional[np.ndarray],
        moving_real: np.ndarray,
        moving_imag: Optional[np.ndarray],
        config: Optional[RegistrationConfig] = None,
    ) -> RegistrationResult:
        """Register ``moving_real`` to ``fixed_real`` and warp both components.

        ``fixed_imag`` is accepted so callers can pass the complete complex
        pair; the current implementation estimates the transform from the real
        component only (matching the historical behavior).
        """
        config = config or RegistrationConfig()
        if fixed_real is None or moving_real is None:
            raise ValueError("fixed_real and moving_real are required")
        if np.shape(fixed_real) != np.shape(moving_real):
            raise ValueError("fixed_real and moving_real must have the same shape")

        fixed_array = np.asarray(fixed_real)
        moving_array = np.asarray(moving_real)
        if moving_imag is not None and np.shape(moving_imag) != np.shape(moving_array):
            raise ValueError("moving_imag must have the same shape as moving_real")

        start = time.perf_counter()
        try:
            fixed_image = sitk.GetImageFromArray(fixed_array)
            moving_image = sitk.GetImageFromArray(moving_array)

            final_transform, method = self._execute(
                fixed_image, moving_image, config
            )
            interpolator = self._interpolator(config.interpolator)

            warped_real = self._resample(
                moving_image, fixed_image, final_transform, interpolator
            )
            warped_imag = None
            if moving_imag is not None:
                moving_imag_image = sitk.GetImageFromArray(np.asarray(moving_imag))
                warped_imag = self._resample(
                    moving_imag_image, fixed_image, final_transform, interpolator
                )

            dice = self._dice(fixed_image, fixed_array, warped_real)
            diagnostics = _similarity_diagnostics(
                fixed_array, moving_array, warped_real, final_transform, fixed_image
            )
            if dice < config.dice_warning_threshold:
                status = STATUS_WARNING
                message = (
                    f"DICE coefficient {dice:.3f} is below "
                    f"{config.dice_warning_threshold:.3f}"
                )
            else:
                status = STATUS_OK
                message = ""

            return RegistrationResult(
                warped_real=warped_real,
                warped_imag=warped_imag,
                transform=final_transform,
                config=config,
                metric_name=self._METRIC_NAMES.get(config.metric, config.metric),
                final_metric=float(method.GetMetricValue()),
                dice=float(dice),
                iterations=int(method.GetOptimizerIteration()),
                stop_condition=str(method.GetOptimizerStopConditionDescription()),
                elapsed_s=time.perf_counter() - start,
                status=status,
                message=message,
                ncc_before=diagnostics["ncc_before"],
                ncc_after=diagnostics["ncc_after"],
                mae_before=diagnostics["mae_before"],
                mae_after=diagnostics["mae_after"],
                displacement_rms=diagnostics["displacement_rms"],
                displacement_max=diagnostics["displacement_max"],
                folding_ratio=diagnostics["folding_ratio"],
            )
        except Exception as exc:  # noqa: BLE001 - surfaced through the result
            return RegistrationResult(
                warped_real=None,
                warped_imag=None,
                transform=None,
                config=config,
                metric_name=self._METRIC_NAMES.get(config.metric, config.metric),
                final_metric=float("nan"),
                dice=float("nan"),
                iterations=0,
                stop_condition="",
                elapsed_s=time.perf_counter() - start,
                status=STATUS_FAILED,
                message=str(exc),
            )

    # ------------------------------------------------------------------
    # SimpleITK configuration
    # ------------------------------------------------------------------
    def _execute(self, fixed_image, moving_image, config):
        """Build and run the registration; returns (transform, method)."""
        method = sitk.ImageRegistrationMethod()
        self._configure_metric(method, config)
        self._configure_optimizer(method, config)
        self._configure_pyramid(method, config)
        method.SetInterpolator(self._interpolator(config.interpolator))

        mesh_size = [int(config.mesh_size)] * fixed_image.GetDimension()
        transform = sitk.BSplineTransformInitializer(fixed_image, mesh_size)
        method.SetInitialTransform(transform)

        final_transform = method.Execute(fixed_image, moving_image)
        return final_transform, method

    def _configure_metric(self, method, config):
        if config.metric == "mattes_mi":
            method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=int(config.histogram_bins)
            )
        elif config.metric == "correlation":
            method.SetMetricAsCorrelation()
        else:
            raise ValueError(f"Unsupported metric: {config.metric!r}")

        method.SetMetricSamplingStrategy(method.RANDOM)
        if config.random_seed is None:
            method.SetMetricSamplingPercentage(float(config.sampling_percentage))
        else:
            method.SetMetricSamplingPercentage(
                float(config.sampling_percentage), int(config.random_seed)
            )

    def _configure_optimizer(self, method, config):
        method.SetOptimizerAsGradientDescent(
            learningRate=float(config.learning_rate),
            numberOfIterations=int(config.iterations),
            convergenceMinimumValue=float(config.convergence_min_value),
            convergenceWindowSize=int(config.convergence_window),
        )
        method.SetOptimizerScalesFromPhysicalShift()

    def _configure_pyramid(self, method, config):
        method.SetShrinkFactorsPerLevel(
            shrinkFactors=[int(value) for value in config.shrink_factors]
        )
        method.SetSmoothingSigmasPerLevel(
            smoothingSigmas=[float(value) for value in config.smoothing_sigmas]
        )
        if config.smoothing_sigmas_in_physical_units:
            method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        else:
            method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    def _interpolator(self, name):
        try:
            return self._INTERPOLATORS[name]
        except KeyError:
            raise ValueError(f"Unsupported interpolator: {name!r}") from None

    @staticmethod
    def _resample(moving_image, fixed_image, transform, interpolator):
        return sitk.GetArrayFromImage(
            sitk.Resample(
                moving_image,
                fixed_image,
                transform,
                interpolator,
                0.0,
                moving_image.GetPixelID(),
            )
        )

    @staticmethod
    def _dice(fixed_image, fixed_array, warped_array):
        """Historical DICE sanity check: foreground is above the image mean."""
        fixed_binary = sitk.BinaryThreshold(
            fixed_image,
            lowerThreshold=float(np.mean(fixed_array)),
            upperThreshold=float(np.max(fixed_array)),
        )
        warped_image = sitk.GetImageFromArray(warped_array)
        moving_binary = sitk.BinaryThreshold(
            warped_image,
            lowerThreshold=float(np.mean(warped_array)),
            upperThreshold=float(np.max(warped_array)),
        )
        overlap = sitk.LabelOverlapMeasuresImageFilter()
        overlap.Execute(fixed_binary, moving_binary)
        return float(overlap.GetDiceCoefficient())


def _similarity_diagnostics(fixed, moving, warped, transform, reference_image):
    """Record quality numbers without changing the registration decision.

    NCC and MAE compare the real component before and after resampling.
    Displacement and folding are sampled from the transform on the reference
    grid.  A transform that cannot report either one leaves that field as
    NaN; it does not fail the registration.  None of these numbers is
    compared with a threshold.
    """
    field = _displacement_grid(transform, reference_image)
    return {
        "ncc_before": _ncc(fixed, moving),
        "ncc_after": _ncc(fixed, warped),
        "mae_before": _mae(fixed, moving),
        "mae_after": _mae(fixed, warped),
        "displacement_rms": _magnitude_rms(field),
        "displacement_max": _magnitude_max(field),
        "folding_ratio": _folding_ratio(field),
    }


def _ncc(left, right):
    """Pearson correlation.  A constant image makes it undefined.

    Centering first would turn every constant image into zeros, so two
    different constants would compare equal and incorrectly return 1.
    """
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    if left.shape != right.shape or left.size == 0:
        return float("nan")
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return float("nan")
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denom)


def _mae(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(np.mean(np.abs(left - right)))


def _displacement_grid(transform, reference_image):
    """Sample the transform on the reference grid.

    ``TransformToDisplacementField`` has no image overload: without an
    explicit size it silently uses a 64³ default, which is neither the
    registered image nor a useful diagnostic.
    """
    try:
        field = sitk.TransformToDisplacementField(
            transform,
            sitk.sitkVectorFloat64,
            reference_image.GetSize(),
            reference_image.GetOrigin(),
            reference_image.GetSpacing(),
            reference_image.GetDirection(),
        )
    except Exception:  # noqa: BLE001 - diagnostic only; registration already ran
        return None
    return field


def _magnitude_rms(field):
    if field is None:
        return float("nan")
    try:
        values = sitk.GetArrayFromImage(field)
        magnitude = np.linalg.norm(values, axis=-1)
        if magnitude.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(magnitude))))
    except Exception:  # noqa: BLE001 - diagnostic only; registration already ran
        return float("nan")


def _magnitude_max(field):
    if field is None:
        return float("nan")
    try:
        values = sitk.GetArrayFromImage(field)
        magnitude = np.linalg.norm(values, axis=-1)
        if magnitude.size == 0:
            return float("nan")
        return float(np.max(magnitude))
    except Exception:  # noqa: BLE001 - diagnostic only; registration already ran
        return float("nan")


def _folding_ratio(field):
    """Fraction of voxels whose Jacobian determinant is not positive."""
    if field is None:
        return float("nan")
    try:
        jacobian = sitk.DisplacementFieldJacobianDeterminant(field)
    except Exception:  # noqa: BLE001 - diagnostic only
        return float("nan")
    values = sitk.GetArrayFromImage(jacobian)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values <= 0.0))
