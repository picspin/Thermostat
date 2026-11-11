"""Opt-in MIND similarity for the SimpleITK registrar.

MIND (modality-independent neighbourhood descriptor) replaces each voxel
with a small self-similarity distribution.  A global intensity shift cancels
inside the pairwise squared differences, and the softmax then normalizes the
remaining pattern, so the descriptor itself does not change.

SimpleITK's ``ImageRegistrationMethod`` has no custom-metric hook and rejects
vector images, so the descriptor cannot be handed to Mattes, correlation, or
mean squares.  This module keeps the SimpleITK transform (rigid, affine, or
cubic B-spline) and minimizes the descriptor distance with L-BFGS-B.  The
caller still resamples the original real and imaginary components with that
transform.  Nothing here is used unless ``RegistrationConfig.metric`` is
``"mind"``.

Two defaults are fixed:

* ``MIND_RADIUS = 1``.  The neighbourhood is the Chebyshev ball of radius 1
  (8 neighbours in 2D, 26 in 3D).  It is the smallest stencil that includes
  diagonals, so the descriptor sees the local pattern rather than only the
  axis-aligned edges.  Radius 2 would be a 5×5×5 stencil.  That is no longer
  a small neighbourhood, and it mixes tissue a one-voxel shift should keep
  apart.  The same radius is used by the separate INR descriptor.

* ``MIND_SAMPLE_STRIDE = 2``.  A radius-1 stencil already overlaps every
  adjacent voxel, so neighbouring descriptors are redundant.  Taking every
  second voxel leaves one sample inside each stencil and cuts the 2D cost by
  four (eight in 3D).  The Mattes 10% random draw is the wrong sampler here:
  it can miss a structure smaller than the draw, and it would import
  SimpleITK's wall-clock seed into a metric that is otherwise deterministic.

``MIND_SIGMA = 0.5`` is not a third knob.  It is the one-voxel Gaussian on
the squared differences: wide enough that one noisy voxel does not dominate
the stencil, and narrow enough that it does not turn radius 1 into a larger
pattern.

The image pyramid is not applied.  ``MIND_RADIUS`` is defined on the original
voxel grid; shrinking the image first would silently change which tissue the
stencil compares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize


MIND_RADIUS = 1
MIND_SAMPLE_STRIDE = 2
MIND_SIGMA = 0.5


@dataclass(frozen=True)
class MindOptimization:
    """Scalar outcome of one MIND solve.  The transform is updated in place."""

    value: float
    iterations: int
    stop_condition: str


def mind_descriptor(
    image: np.ndarray,
    radius: int = MIND_RADIUS,
    sigma: float = MIND_SIGMA,
) -> np.ndarray:
    """Self-similarity distribution, shape ``image.shape + (neighbours,)``.

    Each voxel is compared with the Chebyshev neighbourhood of ``radius``.
    Distances are Gaussian-smoothed squared differences.  Subtracting the
    per-voxel minimum and dividing by the mean makes a global map
    ``I -> a*I + b`` (``a != 0``) cancel, which is why a pure intensity
    offset leaves the descriptor unchanged.  The result sums to one along
    the neighbour axis.

    This is the same definition as the INR module's descriptor.  It is
    duplicated so the SimpleITK path does not import the INR registrar.
    """

    array = np.asarray(image, dtype=float)
    if array.ndim not in (2, 3):
        raise ValueError("MIND supports 2D or 3D images")
    if int(radius) < 1:
        raise ValueError(f"radius must be at least 1, got {radius!r}")
    if float(sigma) <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    if not np.all(np.isfinite(array)):
        raise ValueError("MIND input contains a non-finite value")

    offsets = _neighbour_offsets(array.ndim, int(radius))
    distances = np.empty(array.shape + (len(offsets),), dtype=float)
    for index, offset in enumerate(offsets):
        shifted = map_coordinates(
            array,
            _offset_coordinates(array.shape, offset),
            order=1,
            mode="nearest",
            prefilter=False,
        ).reshape(array.shape)
        distances[..., index] = gaussian_filter(
            np.square(array - shifted), sigma=float(sigma), mode="nearest"
        )
    centered = distances - np.min(distances, axis=-1, keepdims=True)
    weights = np.exp(-centered / (float(np.mean(centered)) + 1e-8))
    return weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-12)


def mind_distance(
    left: np.ndarray,
    right: np.ndarray,
    radius: int = MIND_RADIUS,
    sample_stride: int = MIND_SAMPLE_STRIDE,
    sigma: float = MIND_SIGMA,
) -> float:
    """Mean squared Euclidean distance between two MIND descriptors.

    Samples are the same stride-``sample_stride`` lattice the registrar
    optimizes, inset by ``radius`` so the stencil is not scored on the
    extrapolated rim.  Zero means the sampled self-similarity agrees,
    including the case where one image is a global intensity shift of the
    other.  A spatial misalignment increases it.  This compares arrays that
    already share a grid; it does not estimate a transform.
    """

    if int(sample_stride) < 1:
        raise ValueError(
            f"sample_stride must be at least 1, got {sample_stride!r}"
        )
    left_desc = mind_descriptor(left, radius=radius, sigma=sigma)
    right_desc = mind_descriptor(right, radius=radius, sigma=sigma)
    if left_desc.shape != right_desc.shape:
        raise ValueError(
            f"MIND distance shape {left_desc.shape} does not match "
            f"{right_desc.shape}"
        )
    samples = _sample_rows(left_desc.shape[:-1], int(radius), int(sample_stride))
    delta = left_desc[tuple(samples.T)] - right_desc[tuple(samples.T)]
    return float(np.mean(np.sum(np.square(delta), axis=-1)))


def optimize_mind(
    fixed_image,
    moving_image,
    transform,
    moving_initial_transform=None,
    fixed_mask=None,
    moving_mask=None,
    radius: int = MIND_RADIUS,
    sample_stride: int = MIND_SAMPLE_STRIDE,
    iterations: int = 100,
    convergence: float = 1e-6,
) -> MindOptimization:
    """Minimize MIND on ``transform`` and leave the parameters in place.

    ``moving_initial_transform``, when set, is the outer map and is not
    optimized.  The composed map is ``moving_initial(transform(point))``.
    That is the order of the rigid-then-B-spline cascade: rigid is added
    to the composite first, and SimpleITK applies the last transform
    first, so a fixed point sees the B-spline and then the rigid.

    Masks, when set, choose the sample lattice once from the initial
    transform.  Rebuilding the lattice every iteration would make the
    objective jump as voxels cross the mask boundary.  ``None`` uses the
    stride lattice alone.

    The pyramid is intentionally not used: ``radius`` stays in input voxels.
    """

    fixed = sitk.GetArrayFromImage(fixed_image)
    moving = sitk.GetArrayFromImage(moving_image)
    if fixed.shape != moving.shape:
        raise ValueError("MIND requires fixed and moving images of the same shape")
    fixed_desc = mind_descriptor(fixed, radius=radius)
    moving_desc = mind_descriptor(moving, radius=radius)
    moving_gradient = _descriptor_gradient(moving_desc)
    points, reference = _sample_points(
        fixed_image,
        moving_image,
        fixed_desc,
        transform,
        moving_initial_transform,
        _mask_array(fixed_mask),
        _mask_array(moving_mask),
        int(radius),
        int(sample_stride),
    )
    _, du_dp = _geometry(moving_image)
    channels = moving_desc.shape[-1]
    # Line search can step to a pose whose gradient is exactly zero because
    # every sample has left the image, then stop there even though an
    # earlier pose was better.  Remember every finite evaluation.
    best_loss = np.inf
    best_theta = np.array(transform.GetParameters(), dtype=float)

    def evaluate(theta):
        nonlocal best_loss, best_theta
        theta = np.asarray(theta, dtype=float).reshape(-1)
        transform.SetParameters(_as_tuple(theta))
        # Free transform is inner.  The moving initial, when present, is
        # the outer map: moving_initial(transform(point)).
        inner = _apply_transform(transform, points)
        moved = _apply_transform(moving_initial_transform, inner)
        moving_index = _physical_to_index(moved, du_dp, moving_image)
        sampled = _sample_channels(moving_desc, moving_index)
        flat_gradient = moving_gradient.reshape(moving_desc.shape[:-1] + (-1,))
        sampled_gradient = _sample_channels(flat_gradient, moving_index).reshape(
            points.shape[0], channels, points.shape[1]
        )
        upper = np.array(moving_image.GetSize(), dtype=float) - 1.0
        outside = np.any(
            (moving_index < 0.0) | (moving_index > upper), axis=1
        )
        sampled_gradient[outside] = 0.0
        delta = sampled - reference
        # Past the border the descriptor is clamped, so its spatial gradient
        # is zero and a far pose looks stationary.  Count how far each index
        # is past the border; that term is zero for every in-image sample.
        depth = np.maximum(-moving_index, 0.0) + np.maximum(
            moving_index - upper, 0.0
        )
        signed = np.where(
            moving_index < 0.0,
            moving_index,
            np.where(moving_index > upper, moving_index - upper, 0.0),
        )
        # Small weight: a one-voxel overhang must not swamp the descriptor
        # gradient, while a pose tens of voxels outside still dominates.
        penalty = 1e-3 * float(np.mean(np.sum(np.square(depth), axis=1)))
        loss = float(np.mean(np.sum(np.square(delta), axis=1))) + penalty
        d_loss_d_index = (2.0 / delta.shape[0]) * np.einsum(
            "nc,ncd->nd", delta, sampled_gradient
        )
        d_loss_d_index += (2.0 * 1e-3 / delta.shape[0]) * signed
        d_loss_d_point = d_loss_d_index @ du_dp
        d_loss_d_inner = _pull_through_outer(
            moving_initial_transform, inner, d_loss_d_point
        )
        gradient = _parameter_gradient(transform, points, d_loss_d_inner)
        if gradient.shape != (theta.shape[0],):
            raise RuntimeError(
                f"MIND gradient shape {gradient.shape} does not match "
                f"{theta.shape[0]} parameters"
            )
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("MIND objective became non-finite")
        if loss < best_loss:
            best_loss = loss
            best_theta = np.array(theta, copy=True)
        return loss, gradient

    initial = np.array(transform.GetParameters(), dtype=float)
    if int(iterations) < 1:
        initial_loss, _ = evaluate(initial)
        transform.SetParameters(_as_tuple(initial))
        return MindOptimization(
            value=float(initial_loss),
            iterations=0,
            stop_condition="MIND skipped: iterations < 1",
        )

    # Angle and translation share one gradient.  On a masked blob the angle
    # component points into a shallow basin and L-BFGS never reaches the
    # translation that actually aligns the descriptor.  Solving the
    # translation with the linear part held fixed finds that basin; the
    # second solve is then free to add rotation or affine shear.
    notes = []
    used_iterations = 0
    theta = initial
    translation = _translation_parameter_slice(transform)
    if translation is not None:
        offset, value, used, message = _lbfgs(
            _translation_objective(evaluate, theta, translation),
            theta[translation],
            iterations,
            convergence,
        )
        theta = np.array(theta, copy=True)
        theta[translation] = offset
        used_iterations += used
        notes.append(f"translation {message}")
    theta, value, used, message = _lbfgs(
        evaluate, theta, iterations, convergence
    )
    used_iterations += used
    notes.append(message)
    if best_loss + 1e-12 < value:
        theta = best_theta
        value = best_loss
        notes.append("returned the best evaluation")
    transform.SetParameters(_as_tuple(theta))
    return MindOptimization(
        value=float(value),
        iterations=int(used_iterations),
        stop_condition="MIND L-BFGS-B: " + "; ".join(notes),
    )


def _translation_parameter_slice(transform):
    """Parameters that are a pure translation, or None for a B-spline.

    Euler and affine keep their rotation or matrix fixed during the first
    solve.  A B-spline has no separate translation block.
    """

    name = transform.GetName()
    count = int(transform.GetNumberOfParameters())
    if name == "Euler2DTransform":
        return slice(1, count)
    if name == "Euler3DTransform":
        return slice(3, count)
    if name == "AffineTransform":
        dimension = int(transform.GetDimension())
        return slice(dimension * dimension, count)
    return None


def _translation_objective(evaluate, full_parameters, translation):
    def objective(offset):
        trial = np.array(full_parameters, dtype=float, copy=True)
        trial[translation] = offset
        loss, gradient = evaluate(trial)
        return loss, np.asarray(gradient[translation], dtype=float)

    return objective


def _lbfgs(evaluate, start, iterations, convergence):
    # Copy: L-BFGS may write into the vector it was given, and a rejected
    # step must still be able to restore this start.
    start = np.array(start, dtype=float, copy=True)
    start_loss, _ = evaluate(start)
    result = minimize(
        evaluate,
        start,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(iterations),
            "ftol": float(convergence),
        },
    )
    final = np.asarray(result.x, dtype=float)
    message = result.message
    if isinstance(message, bytes):
        message = message.decode()
    message = str(message).strip()
    kept_start = (
        not np.all(np.isfinite(final))
        or not np.isfinite(result.fun)
        or float(result.fun) > float(start_loss) + 1e-8
    )
    if kept_start:
        return (
            start,
            float(start_loss),
            int(result.nit or 0),
            f"kept initial ({message})",
        )
    return final, float(result.fun), int(result.nit or 0), message


def _pull_through_outer(outer, inner_points, d_loss_d_output):
    """Push ``d(loss)/d(outer output)`` back to the inner transform's output.

    ``outer`` is the moving initial transform.  ``None`` means the free
    transform is the whole map, so the upstream gradient is unchanged.
    """

    if outer is None:
        return d_loss_d_output
    jacobian = _input_jacobian(outer, inner_points)
    return np.einsum("ni,nij->nj", d_loss_d_output, jacobian)


def _input_jacobian(transform, points):
    """``d(output)/d(input)`` with shape ``(N, dim, dim)``."""

    name = transform.GetName()
    count = points.shape[0]
    if name == "Euler2DTransform":
        angle = float(transform.GetParameters()[0])
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        matrix = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
        return np.broadcast_to(matrix, (count, 2, 2)).copy()
    if name == "Euler3DTransform":
        matrix = _euler3d_matrix(transform)
        return np.broadcast_to(matrix, (count, 3, 3)).copy()
    if name == "AffineTransform":
        dimension = points.shape[1]
        parameters = np.array(transform.GetParameters(), dtype=float)
        matrix = parameters[: dimension * dimension].reshape(dimension, dimension)
        return np.broadcast_to(matrix, (count, dimension, dimension)).copy()
    return _finite_difference_input_jacobian(transform, points)


def _euler3d_matrix(transform):
    fixed = transform.GetFixedParameters()
    if len(fixed) >= 4 and float(fixed[3]) != 0.0:
        raise ValueError(
            "MIND expects the default Euler3D angle order (computeZYX off)"
        )
    angle_x, angle_y, angle_z = (
        float(value) for value in transform.GetParameters()[:3]
    )
    cosine_x, sine_x = float(np.cos(angle_x)), float(np.sin(angle_x))
    cosine_y, sine_y = float(np.cos(angle_y)), float(np.sin(angle_y))
    cosine_z, sine_z = float(np.cos(angle_z)), float(np.sin(angle_z))
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine_x, -sine_x], [0.0, sine_x, cosine_x]]
    )
    rotation_y = np.array(
        [[cosine_y, 0.0, sine_y], [0.0, 1.0, 0.0], [-sine_y, 0.0, cosine_y]]
    )
    rotation_z = np.array(
        [[cosine_z, -sine_z, 0.0], [sine_z, cosine_z, 0.0], [0.0, 0.0, 1.0]]
    )
    return rotation_z @ rotation_x @ rotation_y


def _finite_difference_input_jacobian(transform, points, step=1e-4):
    count, dimension = points.shape
    jacobian = np.zeros((count, dimension, dimension), dtype=float)
    for axis in range(dimension):
        plus = np.array(points, dtype=float, copy=True)
        minus = np.array(points, dtype=float, copy=True)
        plus[:, axis] += step
        minus[:, axis] -= step
        jacobian[:, :, axis] = (
            _apply_transform(transform, plus) - _apply_transform(transform, minus)
        ) / (2.0 * step)
    return jacobian


def _parameter_gradient(transform, input_points, d_loss_d_output) -> np.ndarray:
    """Chain ``d(loss)/d(output point)`` into ``d(loss)/d(parameters)``.

    ``input_points`` are the points the free transform itself sees.  In the
    cascade that is the fixed point, because the free transform is the
    inner map.  B-spline support is cubic, which is the SimpleITK default.
    Euler3D uses ITK's default angle order (``computeZYX`` off): rotation
    Y, then X, then Z.
    """

    name = transform.GetName()
    if name == "BSplineTransform":
        return _bspline_parameter_gradient(
            transform, input_points, d_loss_d_output
        )
    if name == "Euler2DTransform":
        jacobian = _euler2d_jacobian(transform, input_points)
    elif name == "Euler3DTransform":
        jacobian = _euler3d_jacobian(transform, input_points)
    elif name == "AffineTransform":
        jacobian = _affine_jacobian(transform, input_points)
    else:
        raise ValueError(
            f"MIND cannot differentiate transform {name!r}. "
            "Supported kinds are rigid, affine, and bspline."
        )
    return np.einsum("nd,ndp->p", d_loss_d_output, jacobian)


def _euler2d_jacobian(transform, points) -> np.ndarray:
    angle = float(transform.GetParameters()[0])
    center = np.array(transform.GetCenter(), dtype=float)
    delta = points - center
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    jacobian = np.zeros((points.shape[0], 2, 3), dtype=float)
    jacobian[:, 0, 0] = -sine * delta[:, 0] - cosine * delta[:, 1]
    jacobian[:, 1, 0] = cosine * delta[:, 0] - sine * delta[:, 1]
    jacobian[:, 0, 1] = 1.0
    jacobian[:, 1, 2] = 1.0
    return jacobian


def _euler3d_jacobian(transform, points) -> np.ndarray:
    fixed = transform.GetFixedParameters()
    if len(fixed) >= 4 and float(fixed[3]) != 0.0:
        raise ValueError(
            "MIND expects the default Euler3D angle order (computeZYX off)"
        )
    angle_x, angle_y, angle_z = (
        float(value) for value in transform.GetParameters()[:3]
    )
    center = np.array(transform.GetCenter(), dtype=float)
    cosine_x, sine_x = float(np.cos(angle_x)), float(np.sin(angle_x))
    cosine_y, sine_y = float(np.cos(angle_y)), float(np.sin(angle_y))
    cosine_z, sine_z = float(np.cos(angle_z)), float(np.sin(angle_z))
    shifted = points - center
    px, py, pz = shifted[:, 0], shifted[:, 1], shifted[:, 2]
    jacobian = np.zeros((points.shape[0], 3, 6), dtype=float)
    # ITK Euler3DTransform::ComputeJacobianWithRespectToParameters, the
    # computeZYX == false branch.  Columns are angleX, angleY, angleZ.
    jacobian[:, 0, 0] = (
        (-sine_z * cosine_x * sine_y) * px
        + (sine_z * sine_x) * py
        + (sine_z * cosine_x * cosine_y) * pz
    )
    jacobian[:, 1, 0] = (
        (cosine_z * cosine_x * sine_y) * px
        + (-cosine_z * sine_x) * py
        + (-cosine_z * cosine_x * cosine_y) * pz
    )
    jacobian[:, 2, 0] = (
        (sine_x * sine_y) * px + cosine_x * py + (-sine_x * cosine_y) * pz
    )
    jacobian[:, 0, 1] = (
        (-cosine_z * sine_y - sine_z * sine_x * cosine_y) * px
        + (cosine_z * cosine_y - sine_z * sine_x * sine_y) * pz
    )
    jacobian[:, 1, 1] = (
        (-sine_z * sine_y + cosine_z * sine_x * cosine_y) * px
        + (sine_z * cosine_y + cosine_z * sine_x * sine_y) * pz
    )
    jacobian[:, 2, 1] = (-cosine_x * cosine_y) * px + (-cosine_x * sine_y) * pz
    jacobian[:, 0, 2] = (
        (-sine_z * cosine_y - cosine_z * sine_x * sine_y) * px
        + (-cosine_z * cosine_x) * py
        + (-sine_z * sine_y + cosine_z * sine_x * cosine_y) * pz
    )
    jacobian[:, 1, 2] = (
        (cosine_z * cosine_y - sine_z * sine_x * sine_y) * px
        + (-sine_z * cosine_x) * py
        + (cosine_z * sine_y + sine_z * sine_x * cosine_y) * pz
    )
    jacobian[:, 0, 3] = 1.0
    jacobian[:, 1, 4] = 1.0
    jacobian[:, 2, 5] = 1.0
    return jacobian


def _affine_jacobian(transform, points) -> np.ndarray:
    dimension = points.shape[1]
    center = np.array(transform.GetCenter(), dtype=float)
    delta = points - center
    n_parameters = dimension * dimension + dimension
    jacobian = np.zeros((points.shape[0], dimension, n_parameters), dtype=float)
    for row in range(dimension):
        for column in range(dimension):
            jacobian[:, row, row * dimension + column] = delta[:, column]
        jacobian[:, row, dimension * dimension + row] = 1.0
    return jacobian


def _bspline_parameter_gradient(transform, points, d_loss_d_output) -> np.ndarray:
    if int(transform.GetOrder()) != 3:
        raise ValueError("MIND only differentiates the cubic SimpleITK B-spline")
    dimension = points.shape[1]
    coefficient = transform.GetCoefficientImages()[0]
    origin = np.array(coefficient.GetOrigin(), dtype=float)
    spacing = np.array(coefficient.GetSpacing(), dtype=float)
    direction = np.array(coefficient.GetDirection(), dtype=float).reshape(
        dimension, dimension
    )
    control = np.array(coefficient.GetSize(), dtype=int)
    n_control = int(np.prod(control))
    gradient = np.zeros(dimension * n_control, dtype=float)

    domain_origin = np.array(transform.GetTransformDomainOrigin(), dtype=float)
    domain_size = np.array(
        transform.GetTransformDomainPhysicalDimensions(), dtype=float
    )
    domain_direction = np.array(
        transform.GetTransformDomainDirection(), dtype=float
    ).reshape(dimension, dimension)
    local = (points - domain_origin) @ np.linalg.inv(domain_direction).T
    inside = np.all((local >= 0.0) & (local < domain_size), axis=1)

    continuous = (points - origin) @ np.linalg.inv(direction).T
    continuous = continuous / spacing
    support = np.floor(continuous).astype(int) - 1
    offset = np.arange(4)
    index = support[:, :, None] + offset[None, None, :]
    weight = _cubic_bspline(continuous[:, :, None] - index)
    valid = (index >= 0) & (index < control[None, :, None])
    weight = np.where(valid, weight, 0.0)
    weight[~inside] = 0.0

    if dimension == 2:
        separable = weight[:, 0, :, None] * weight[:, 1, None, :]
        ix = index[:, 0, :, None]
        iy = index[:, 1, None, :]
        flat = ix + control[0] * iy
        usable = (ix >= 0) & (ix < control[0]) & (iy >= 0) & (iy < control[1])
    elif dimension == 3:
        separable = (
            weight[:, 0, :, None, None]
            * weight[:, 1, None, :, None]
            * weight[:, 2, None, None, :]
        )
        ix = index[:, 0, :, None, None]
        iy = index[:, 1, None, :, None]
        iz = index[:, 2, None, None, :]
        flat = ix + control[0] * (iy + control[1] * iz)
        usable = (
            (ix >= 0)
            & (ix < control[0])
            & (iy >= 0)
            & (iy < control[1])
            & (iz >= 0)
            & (iz < control[2])
        )
    else:
        raise ValueError(f"MIND B-spline does not support dimension {dimension}")
    flat = np.where(usable, flat, 0)
    separable = np.where(usable, separable, 0.0)
    for axis in range(dimension):
        contribution = d_loss_d_output[:, axis].reshape(
            (-1,) + (1,) * (separable.ndim - 1)
        )
        np.add.at(
            gradient,
            (axis * n_control + flat).ravel(),
            (contribution * separable).ravel(),
        )
    return gradient


def _sample_points(
    fixed_image,
    moving_image,
    fixed_desc,
    transform,
    moving_initial_transform,
    fixed_mask,
    moving_mask,
    radius,
    sample_stride,
):
    rows = _sample_rows(fixed_desc.shape[:-1], radius, sample_stride)
    if fixed_mask is not None:
        if fixed_mask.shape != fixed_desc.shape[:-1]:
            raise ValueError("fixed MIND mask shape does not match the image")
        rows = rows[fixed_mask[tuple(rows.T)] > 0]
    if rows.size == 0:
        raise ValueError("MIND sampling mask removed every voxel")
    index_xyz = rows[:, ::-1].astype(float)
    points = _index_to_physical(index_xyz, fixed_image)
    reference = fixed_desc[tuple(rows.T)]
    if moving_mask is None:
        return points, reference
    if moving_mask.shape != fixed_desc.shape[:-1]:
        raise ValueError("moving MIND mask shape does not match the image")
    initial_inner = _apply_transform(transform, points)
    initial_moved = _apply_transform(moving_initial_transform, initial_inner)
    _, du_dp = _geometry(moving_image)
    moving_index = _physical_to_index(initial_moved, du_dp, moving_image)
    nearest = np.rint(moving_index[:, ::-1]).astype(int)
    inside = np.ones(nearest.shape[0], dtype=bool)
    for axis, size in enumerate(moving_mask.shape):
        inside &= (nearest[:, axis] >= 0) & (nearest[:, axis] < size)
    keep = np.zeros(nearest.shape[0], dtype=bool)
    if np.any(inside):
        keep[inside] = moving_mask[tuple(nearest[inside].T)] > 0
    if not np.any(keep):
        raise ValueError("MIND sampling mask removed every voxel")
    return points[keep], reference[keep]


def _sample_rows(shape, radius: int, sample_stride: int) -> np.ndarray:
    """Numpy-order lattice inset by ``radius``, with at least one sample."""

    axes = []
    for size in shape:
        inset = min(int(radius), max(int(size) // 2 - 1, 0)) if size > 1 else 0
        start = inset
        stop = int(size) - inset
        if stop <= start:
            start, stop = 0, int(size)
        axis = np.arange(start, stop, int(sample_stride))
        if axis.size == 0:
            axis = np.array([int(size) // 2])
        axes.append(axis)
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([item.ravel() for item in mesh], axis=1)


def _descriptor_gradient(descriptor: np.ndarray) -> np.ndarray:
    """Central ``d(descriptor)/d(index)`` in SimpleITK axis order (x, y, z)."""

    spatial = descriptor.ndim - 1
    grads = np.gradient(descriptor, axis=tuple(range(spatial)))
    ordered = [grads[spatial - 1 - axis] for axis in range(spatial)]
    return np.stack(ordered, axis=-1)


def _sample_channels(volume: np.ndarray, index_xyz: np.ndarray) -> np.ndarray:
    """Linear sample of a ``(..., channels)`` array at SimpleITK indices."""

    channels = volume.shape[-1]
    ordered = np.moveaxis(volume, -1, 0)
    numpy_index = np.asarray(index_xyz, dtype=float)[:, ::-1]
    count = numpy_index.shape[0]
    coordinates = np.empty((1 + numpy_index.shape[1], count * channels), dtype=float)
    coordinates[0] = np.repeat(np.arange(channels), count)
    coordinates[1:] = np.tile(numpy_index.T, channels)
    sampled = map_coordinates(
        ordered, coordinates, order=1, mode="nearest", prefilter=False
    )
    return sampled.reshape(channels, count).T


def _geometry(image):
    origin, du_dp, _, _ = _frame(image)
    return origin, du_dp


def _index_to_physical(index_xyz: np.ndarray, image) -> np.ndarray:
    origin, _, direction, spacing = _frame(image)
    return origin + (np.asarray(index_xyz, dtype=float) * spacing) @ direction.T


def _physical_to_index(physical: np.ndarray, du_dp: np.ndarray, image) -> np.ndarray:
    origin = np.array(image.GetOrigin(), dtype=float)
    return (np.asarray(physical, dtype=float) - origin) @ du_dp.T


def _frame(image):
    dimension = image.GetDimension()
    origin = np.array(image.GetOrigin(), dtype=float)
    spacing = np.array(image.GetSpacing(), dtype=float)
    direction = np.array(image.GetDirection(), dtype=float).reshape(
        dimension, dimension
    )
    return origin, np.linalg.inv(direction) / spacing[:, None], direction, spacing


def _apply_transform(transform, points: np.ndarray) -> np.ndarray:
    if transform is None:
        return np.asarray(points, dtype=float)
    mapped = [
        transform.TransformPoint(tuple(float(value) for value in point))
        for point in points
    ]
    return np.asarray(mapped, dtype=float)


def _mask_array(mask):
    if mask is None:
        return None
    return sitk.GetArrayFromImage(mask)


def _as_tuple(values):
    return tuple(float(value) for value in values)


def _neighbour_offsets(ndim: int, radius: int):
    ranges = [range(-radius, radius + 1) for _ in range(ndim)]
    stacked = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, ndim)
    return [
        tuple(int(value) for value in offset)
        for offset in stacked
        if np.any(offset)
    ]


def _offset_coordinates(shape, offset) -> np.ndarray:
    axes = [
        np.arange(size, dtype=float) + shift
        for size, shift in zip(shape, offset, strict=True)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([item.ravel() for item in mesh], axis=0)


def _cubic_bspline(distance: np.ndarray) -> np.ndarray:
    """Cardinal cubic B-spline.  Nonzero only for ``abs(distance) < 2``."""

    absolute = np.abs(distance)
    weight = np.zeros_like(absolute, dtype=float)
    near = absolute < 1.0
    far = (absolute >= 1.0) & (absolute < 2.0)
    weight[near] = (2.0 / 3.0) - absolute[near] ** 2 + 0.5 * absolute[near] ** 3
    weight[far] = ((2.0 - absolute[far]) ** 3) / 6.0
    return weight
