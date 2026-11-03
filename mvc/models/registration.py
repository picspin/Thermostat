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
            raise ValueError(f"Unsupported interpolator: {name!r}")

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
