"""Volume registration used for motion correction.

This module deliberately has no Qt imports: the registration can be exercised
with plain numpy arrays in headless tests.  The controller is responsible for
deciding what to do with the result and the model is responsible for keeping
the original acquired images intact.

``RegistrationConfig()`` is still the historical single-stage B-spline with
Mattes MI.  The motion-correction checkbox uses
``RegistrationConfig.liver_respiratory()`` instead: rigid, then a coarse
B-spline, with local correlation and an Otsu mask.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
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
        metric: ``"mattes_mi"`` (historical), ``"correlation"``, or
            ``"ants_neighborhood_correlation"`` (local NCC).
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
        mesh_size: B-spline mesh size per dimension.  Used only for
            ``transform_kind="bspline"``.
        interpolator: ``"linear"``, ``"nearest"`` or ``"bspline"``.
        dice_warning_threshold: DICE below this value produces a warning.
        transform_kind: ``"bspline"`` (historical), ``"rigid"``, ``"affine"``,
            or ``"rigid_bspline"`` (rigid stage, then a B-spline).
        foreground_mask: When true, and ``mask_kind`` is ``"none"``, the
            metric uses the percentile mask.  The historical default is off.
        foreground_percentile: Percentile of the non-background signal used
            by that mask.
        neighborhood_radius: Voxel radius for local NCC.  Ignored by the
            other metrics.  4 is inside the requested 3–5 voxel range.
        mask_kind: ``"none"``, ``"percentile"``, or ``"otsu_dilated"``.
            The dilated Otsu mask is a coarse tissue mask, not a liver
            segmentation.  It drops air and adds a rim around the tissue.
        dilation_mm: Physical radius of that dilation.
        bspline_grid_spacing_mm: When set, the B-spline mesh is sized from
            the image's physical extent.  ``None`` keeps ``mesh_size``.
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
    transform_kind: str = "bspline"
    foreground_mask: bool = False
    foreground_percentile: float = 80.0
    neighborhood_radius: int = 4
    mask_kind: str = "none"
    dilation_mm: float = 12.0
    bspline_grid_spacing_mm: Optional[float] = None

    def __post_init__(self):
        transforms = ("bspline", "rigid", "affine", "rigid_bspline")
        if self.transform_kind not in transforms:
            raise ValueError(
                "Unsupported transform_kind: "
                f"{self.transform_kind!r}. Expected one of {transforms}."
            )
        masks = ("none", "percentile", "otsu_dilated")
        if self.mask_kind not in masks:
            raise ValueError(
                f"Unsupported mask_kind: {self.mask_kind!r}. "
                f"Expected one of {masks}."
            )
        if not 0.0 <= float(self.foreground_percentile) <= 100.0:
            raise ValueError(
                "foreground_percentile must be between 0 and 100, "
                f"got {self.foreground_percentile!r}"
            )
        if int(self.neighborhood_radius) < 1:
            raise ValueError(
                "neighborhood_radius must be at least 1, "
                f"got {self.neighborhood_radius!r}"
            )
        if self.bspline_grid_spacing_mm is not None:
            if float(self.bspline_grid_spacing_mm) <= 0.0:
                raise ValueError(
                    "bspline_grid_spacing_mm must be positive, "
                    f"got {self.bspline_grid_spacing_mm!r}"
                )

    @classmethod
    def liver_respiratory(cls) -> "RegistrationConfig":
        """Checkbox preset for same-sequence respiratory motion.

        Stage 1 is rigid, not affine: without a true liver mask, affine
        shear follows the abdominal wall.  Stage 2 is a B-spline whose
        cells are about 40 mm, inside the requested 35–45 mm range.
        Local NCC uses a 4-voxel radius.  The mask is Otsu plus a 12 mm
        dilation, inside the requested 10–15 mm range.  SimpleITK has no
        bending-energy metric, so the coarse grid is the regularizer.
        """
        return cls(
            metric="ants_neighborhood_correlation",
            neighborhood_radius=4,
            transform_kind="rigid_bspline",
            mask_kind="otsu_dilated",
            dilation_mm=12.0,
            bspline_grid_spacing_mm=40.0,
            histogram_bins=32,
        )


def default_registration_config() -> RegistrationConfig:
    """Configuration used when the user turns motion correction on."""
    return RegistrationConfig.liver_respiratory()


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
    """SimpleITK registrar for 2D/3D motion correction.

    The default transform is the historical B-spline, optimized with
    gradient descent.  Rigid and affine are opt-in.  They are initialized
    at the image center and optimized with regular-step gradient descent,
    because the B-spline step size does not recover rotation.  That choice
    does not change the default B-spline path.  The transform is estimated
    from the real component and then applied to both components.
    """

    _METRIC_NAMES = {
        "mattes_mi": "MattesMutualInformation",
        "correlation": "Correlation",
        "ants_neighborhood_correlation": "ANTSNeighborhoodCorrelation",
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
        spacing_mm: Optional[Tuple[float, ...]] = None,
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
            _assign_spacing(fixed_image, spacing_mm)
            _assign_spacing(moving_image, spacing_mm)

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
                _assign_spacing(moving_imag_image, spacing_mm)
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
            if config.transform_kind == "rigid_bspline" and not message:
                message = "rigid then coarse bspline"

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
    def _execute(
        self,
        fixed_image,
        moving_image,
        config,
        moving_initial_transform=None,
    ):
        """Build and run one stage; returns (transform, method)."""
        if config.transform_kind == "rigid_bspline":
            return self._execute_cascade(fixed_image, moving_image, config)
        method = sitk.ImageRegistrationMethod()
        self._configure_metric(method, config)
        # The historical single-stage B-spline keeps gradient descent.
        # The second stage of a rigid-then-B-spline cascade starts from an
        # alignment that is already close, and that same step size overshoots.
        self._configure_optimizer(
            method,
            config,
            regular_step=moving_initial_transform is not None,
        )
        self._configure_pyramid(method, config)
        method.SetInterpolator(self._interpolator(config.interpolator))
        self._configure_mask(method, fixed_image, moving_image, config)

        transform = self._initial_transform(fixed_image, moving_image, config)
        if moving_initial_transform is not None:
            method.SetMovingInitialTransform(moving_initial_transform)
        # B-spline keeps the historical in-place flag (false).  Rigid and
        # affine must be updated in place so the centered parameters are the
        # ones returned to the caller.
        if config.transform_kind == "bspline":
            method.SetInitialTransform(transform)
        else:
            method.SetInitialTransform(transform, True)

        final_transform = method.Execute(fixed_image, moving_image)
        return final_transform, method

    def _execute_cascade(self, fixed_image, moving_image, config):
        """Rigid alignment, then a B-spline on that result."""
        rigid_tx, _ = self._execute(
            fixed_image,
            moving_image,
            replace(config, transform_kind="rigid"),
        )
        bspline_tx, method = self._execute(
            fixed_image,
            moving_image,
            replace(config, transform_kind="bspline"),
            moving_initial_transform=rigid_tx,
        )
        composite = sitk.CompositeTransform(fixed_image.GetDimension())
        composite.AddTransform(rigid_tx)
        composite.AddTransform(bspline_tx)
        return composite, method

    def _initial_transform(self, fixed_image, moving_image, config):
        """Initial transform.  The default kind is still a B-spline."""
        kind = config.transform_kind
        dimension = fixed_image.GetDimension()
        if kind == "bspline":
            mesh = self._bspline_mesh(fixed_image, config)
            return sitk.BSplineTransformInitializer(fixed_image, mesh)
        if kind == "rigid":
            if dimension == 2:
                transform = sitk.Euler2DTransform()
            elif dimension == 3:
                transform = sitk.Euler3DTransform()
            else:
                raise ValueError(
                    "rigid registration does not support "
                    f"dimension {dimension}"
                )
        elif kind == "affine":
            if dimension not in (2, 3):
                raise ValueError(
                    "affine registration does not support "
                    f"dimension {dimension}"
                )
            transform = sitk.AffineTransform(dimension)
        else:
            raise ValueError(f"Unsupported transform_kind: {kind!r}")
        return sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            transform,
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

    def _bspline_mesh(self, image, config):
        if config.bspline_grid_spacing_mm is None:
            return [int(config.mesh_size)] * image.GetDimension()
        mesh = []
        step_mm = float(config.bspline_grid_spacing_mm)
        for length, spacing in zip(
            image.GetSize(), image.GetSpacing(), strict=True
        ):
            physical = float(length) * float(spacing)
            mesh.append(max(1, int(round(physical / step_mm))))
        return mesh

    def _configure_mask(self, method, fixed_image, moving_image, config):
        kind = _active_mask_kind(config)
        if kind == "none":
            return
        if kind == "percentile":
            fixed_mask = self._foreground_mask(
                fixed_image, config.foreground_percentile
            )
            moving_mask = self._foreground_mask(
                moving_image, config.foreground_percentile
            )
        else:
            fixed_mask = self._otsu_dilated_mask(fixed_image, config.dilation_mm)
            moving_mask = self._otsu_dilated_mask(
                moving_image, config.dilation_mm
            )
        method.SetMetricFixedMask(fixed_mask)
        method.SetMetricMovingMask(moving_mask)

    def _otsu_dilated_mask(self, image, radius_mm):
        """Coarse tissue mask: Otsu, then a physical dilation.

        This is not a liver segmentation.  Bright chest wall stays inside
        the Otsu foreground.  The dilation only adds a rim so the diaphragm
        next to the tissue is not cut off.
        """
        mask = sitk.OtsuThreshold(image, 0, 1)
        radius = [
            max(1, int(round(float(radius_mm) / float(step))))
            for step in image.GetSpacing()
        ]
        dilated = sitk.BinaryDilate(mask, radius)
        return sitk.Cast(dilated, sitk.sitkUInt8)

    @staticmethod
    def _foreground_mask(image, percentile):
        array = sitk.GetArrayFromImage(image)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError("foreground mask needs a finite image")
        # A percentile of the whole image is a no-op when most voxels are
        # background: the 80th percentile of a 90% zero image is still zero,
        # and the mask then includes every voxel.
        peak = float(np.max(finite))
        if peak > 0.0:
            signal = finite[finite > peak * 1e-6]
        else:
            signal = finite
        if signal.size == 0:
            signal = finite
        threshold = float(np.percentile(signal, float(percentile)))
        upper = float(np.max(signal))
        mask = sitk.BinaryThreshold(
            image,
            lowerThreshold=threshold,
            upperThreshold=upper,
            insideValue=1,
            outsideValue=0,
        )
        return sitk.Cast(mask, sitk.sitkUInt8)

    def _configure_metric(self, method, config):
        if config.metric == "mattes_mi":
            method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=int(config.histogram_bins)
            )
        elif config.metric == "correlation":
            method.SetMetricAsCorrelation()
        elif config.metric == "ants_neighborhood_correlation":
            method.SetMetricAsANTSNeighborhoodCorrelation(
                radius=int(config.neighborhood_radius)
            )
        else:
            raise ValueError(f"Unsupported metric: {config.metric!r}")

        method.SetMetricSamplingStrategy(method.RANDOM)
        if config.random_seed is None:
            method.SetMetricSamplingPercentage(float(config.sampling_percentage))
        else:
            method.SetMetricSamplingPercentage(
                float(config.sampling_percentage), int(config.random_seed)
            )

    def _configure_optimizer(self, method, config, regular_step=False):
        if config.transform_kind == "bspline" and not regular_step:
            # Historical optimizer.  Do not change this for a lone B-spline.
            method.SetOptimizerAsGradientDescent(
                learningRate=float(config.learning_rate),
                numberOfIterations=int(config.iterations),
                convergenceMinimumValue=float(config.convergence_min_value),
                convergenceWindowSize=int(config.convergence_window),
            )
        else:
            # Gradient descent with the B-spline step recovers a rigid
            # rotation with the wrong translation and a worse MAE.  Regular
            # step stays inside the overlap and reaches the known angle.
            learning_rate = float(config.learning_rate)
            method.SetOptimizerAsRegularStepGradientDescent(
                learningRate=learning_rate,
                minStep=min(1e-4, abs(learning_rate) * 1e-4),
                numberOfIterations=int(config.iterations),
                relaxationFactor=0.5,
                gradientMagnitudeTolerance=float(config.convergence_min_value),
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
        # GetImageFromArray defaults to spacing 1.  The fixed image may
        # carry DICOM spacing, and the overlap filter rejects that mismatch.
        warped_image.CopyInformation(fixed_image)
        moving_binary = sitk.BinaryThreshold(
            warped_image,
            lowerThreshold=float(np.mean(warped_array)),
            upperThreshold=float(np.max(warped_array)),
        )
        overlap = sitk.LabelOverlapMeasuresImageFilter()
        overlap.Execute(fixed_binary, moving_binary)
        return float(overlap.GetDiceCoefficient())


def _active_mask_kind(config):
    if config.mask_kind == "otsu_dilated":
        return "otsu_dilated"
    if config.mask_kind == "percentile" or config.foreground_mask:
        return "percentile"
    return "none"


def _assign_spacing(image, spacing_mm):
    if spacing_mm is None:
        return
    values = tuple(float(value) for value in spacing_mm)
    if len(values) != image.GetDimension():
        raise ValueError(
            "spacing_mm has "
            f"{len(values)} values for a {image.GetDimension()}D image"
        )
    image.SetSpacing(values)


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
