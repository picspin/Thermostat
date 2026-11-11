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
  second voxel leaves one sample inside each interior stencil and cuts the
  2D cost by four (eight in 3D).  The lattice is inset by the radius, so the
  last row and column on the high-index side can fall outside every stencil;
  that rim is the extrapolated border the distance does not score.  A stride
  larger than ``2 * radius + 1`` can miss a stencil entirely.  The Mattes
  10% random draw is the wrong sampler here: it can miss a structure smaller
  than the draw, and it would import SimpleITK's wall-clock seed into a
  metric that is otherwise deterministic.

``MIND_SIGMA = 0.5`` is not a third knob.  It is the one-voxel Gaussian on
the squared differences: wide enough that one noisy voxel does not dominate
the stencil, and narrow enough that it does not turn radius 1 into a larger
pattern.

The moving descriptor is rebuilt after the moving image is warped onto the
fixed grid.  Neighbour directions are therefore always the fixed voxel
axes, including under rotation; resampling a precomputed moving descriptor
would compare a rotated stencil with an unrotated one.  The warp is linear
(``map_coordinates`` order 1, nearest outside the image) and the analytic
gradient is the slope of that same interpolant.  It is not a full-volume
``np.gradient`` of the descriptor: that field is both the wrong derivative
of the loss and large enough to exhaust memory on an ordinary 3D volume.
A singleton axis has slope zero because nearest lookup collapses it.

The image pyramid is not applied.  ``MIND_RADIUS`` is defined on the original
voxel grid; shrinking the image first would silently change which tissue the
stencil compares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
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
    per-voxel minimum and dividing by the mean of those centered distances
    over the whole volume makes a global map ``I -> a*I + b`` (``a != 0``)
    cancel, which is why a pure intensity offset leaves the descriptor
    unchanged.  The result sums to one along the neighbour axis.

    The divisor is that one global mean, not the per-voxel neighbour mean
    in Heinrich et al. 2012.  A bright structure anywhere rescales every
    voxel together.  This is intentional: it is the same formula as the
    INR descriptor, and this path must not fork that definition.  The INR
    optimizer itself is not used here; the formula is duplicated so the
    SimpleITK registrar does not import it.
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
        # Integer offsets make linear interpolation a nearest shift.  The
        # same shift is what the analytic gradient inverts.
        shifted = _shift_nearest(array, offset)
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
    extrapolated rim.  The inset is ``[radius, size - radius)``, so the
    high-index edge itself is outside every sample's stencil.  A stride
    greater than ``2 * radius + 1`` can leave an interior stencil with no
    sample at all; the distance then only sees the coarser lattice.
    Zero means the sampled self-similarity agrees, including the case where
    one image is a global intensity shift of the other.  A spatial
    misalignment increases it.  This compares arrays that already share a
    grid; it does not estimate a transform.
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
    stride lattice alone.  A large initial misalignment can put the true
    overlap outside the moving mask; sampling then raises ``ValueError``
    (``MIND sampling mask removed every voxel``) and the registration
    fails instead of drifting.

    ``iterations`` is the total budget for this call, not a per-phase
    budget.  Rigid and affine may spend half of it (rounded down) on a
    translation warm-up and give whatever is left to the full solve.
    The warm-up takes steps of at most two physical units so it cannot
    jump the translation basin; the full solve is L-BFGS-B.  One
    iteration is available only to the warm-up.  A B-spline has no
    translation block and spends the whole budget on its single solve.
    The reported iteration count is the sum of the phases, so it does not
    exceed ``iterations``.  ``convergence`` is both the ``ftol`` and the
    projected-gradient tolerance (``gtol``, Fortran's ``pgtol``) of L-BFGS-B.

    The pyramid is intentionally not used: ``radius`` stays in input voxels.
    """

    initial = np.array(transform.GetParameters(), dtype=float)
    objective = _mind_objective(
        fixed_image,
        moving_image,
        transform,
        moving_initial_transform,
        fixed_mask,
        moving_mask,
        int(radius),
        int(sample_stride),
    )
    # Line search can step to a pose whose gradient is exactly zero because
    # every sample has left the image, then stop there even though an
    # earlier pose was better.  Remember every finite evaluation.
    best_loss = np.inf
    best_theta = np.array(initial, copy=True)

    def evaluate(theta):
        nonlocal best_loss, best_theta
        loss, gradient = objective(theta)
        if loss < best_loss:
            best_loss = loss
            best_theta = np.array(theta, dtype=float, copy=True)
        return loss, gradient

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
    # second solve is then free to add rotation or affine shear.  The two
    # phases share ``iterations`` so a caller-configured maximum is the
    # maximum.
    notes = []
    used_iterations = 0
    theta = initial
    translation = _translation_parameter_slice(transform)
    warm_budget, _ = _iteration_budgets(int(iterations), translation is not None)
    value = None
    # The translation basin is narrow once the descriptor is built on the
    # warped image.  L-BFGS-B's first step is long enough to jump over it.
    # Walk at most two physical units per warm-up step, then give every
    # unused iteration to the full solve.  The two phases together stay
    # inside ``iterations``.
    if translation is not None and warm_budget > 0:
        offset, value, used, message = _capped_descent(
            _translation_objective(evaluate, theta, translation),
            theta[translation],
            warm_budget,
            max_step=2.0,
        )
        theta = np.array(theta, copy=True)
        theta[translation] = offset
        used_iterations += used
        notes.append(f"translation {message}")
    full_budget = int(iterations) - used_iterations
    if full_budget > 0:
        theta, value, used, message = _lbfgs(
            evaluate, theta, full_budget, convergence
        )
        used_iterations += used
        notes.append(message)
    elif value is None:
        value, _ = evaluate(theta)
        notes.append("full solve skipped: iteration budget spent on translation")
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


def _capped_descent(objective, start, steps, max_step):
    """Gradient steps whose length is at most ``max_step``.

    Backtracking shortens the step when it does not decrease the loss.
    Each accepted step counts as one iteration.  Returns the same tuple
    shape as ``_lbfgs``.
    """

    theta = np.array(start, dtype=float, copy=True)
    value, gradient = objective(theta)
    used = 0
    message = "step limit reached"
    for _ in range(int(steps)):
        grad_norm = float(np.linalg.norm(gradient))
        if not np.isfinite(grad_norm) or grad_norm < 1e-12:
            message = "gradient vanished"
            break
        direction = -np.asarray(gradient, dtype=float) / grad_norm
        length = float(max_step)
        accepted = False
        for _attempt in range(12):
            trial = theta + length * direction
            trial_value, trial_gradient = objective(trial)
            if trial_value < value - 1e-12:
                theta = np.asarray(trial, dtype=float)
                value = float(trial_value)
                gradient = np.asarray(trial_gradient, dtype=float)
                accepted = True
                used += 1
                break
            length *= 0.5
        if not accepted:
            message = "line search failed"
            break
    return theta, float(value), used, message


def _iteration_budgets(iterations, has_translation_phase):
    """Split one caller budget across the warm-up and the full solve.

    The warm-up gets half, rounded down, and at least the only iteration
    when the budget is 1.  The two ``maxiter`` values sum to ``iterations``.
    """

    iterations = int(iterations)
    if iterations < 1:
        return 0, 0
    if not has_translation_phase:
        return 0, iterations
    if iterations == 1:
        return 1, 0
    warm = iterations // 2
    return warm, iterations - warm


def _lbfgs(evaluate, start, iterations, convergence):
    # Copy: L-BFGS may write into the vector it was given, and a rejected
    # step must still be able to restore this start.
    # ``convergence`` is both the function tolerance and the projected
    # gradient tolerance.  Leaving ``pgtol`` at SciPy's default would let a
    # different threshold stop the solve than the one the caller set.
    start = np.array(start, dtype=float, copy=True)
    start_loss, _ = evaluate(start)
    result = minimize(
        evaluate,
        start,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(iterations),
            # SciPy names the projected-gradient tolerance ``gtol``.
            # The Fortran solver calls the same threshold ``pgtol``.
            "gtol": float(convergence),
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
        return points, reference, rows
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
    return points[keep], reference[keep], rows[keep]


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


# Points per objective chunk.  The descriptor itself stays a full volume;
# the gradient does not.  8192 keeps the per-chunk temporaries small on a
# clinical lattice (a 256³ stride-2 grid is a few million points).
_OBJECTIVE_CHUNK = 8192


def _mind_objective(
    fixed_image,
    moving_image,
    transform,
    moving_initial_transform,
    fixed_mask,
    moving_mask,
    radius,
    sample_stride,
):
    """Return ``(loss, d(loss)/d(parameters))`` for one pose of ``transform``.

    The callable writes ``transform``'s parameters.  The moving image is
    warped onto the fixed grid and the descriptor is computed there, so
    both stencils use the fixed voxel directions.
    """

    fixed = sitk.GetArrayFromImage(fixed_image)
    moving = sitk.GetArrayFromImage(moving_image)
    if fixed.shape != moving.shape:
        raise ValueError("MIND requires fixed and moving images of the same shape")
    fixed_desc = mind_descriptor(fixed, radius=radius)
    _, reference, rows = _sample_points(
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
    all_rows = np.stack(
        [item.ravel() for item in np.meshgrid(
            *[np.arange(size) for size in fixed.shape], indexing="ij"
        )],
        axis=1,
    )
    all_points = _index_to_physical(all_rows[:, ::-1].astype(float), fixed_image)
    sample_flat = np.ravel_multi_index(tuple(rows.T), fixed.shape)
    _, du_dp = _geometry(moving_image)
    count = rows.shape[0]
    upper = np.array(moving_image.GetSize(), dtype=float) - 1.0
    sigma = MIND_SIGMA

    def objective(theta):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        transform.SetParameters(_as_tuple(theta))
        warped, slope, inner, moving_index = _warp_moving(
            moving, moving_image, transform, moving_initial_transform, all_points, du_dp
        )
        warped = warped.reshape(fixed.shape)
        moving_desc = mind_descriptor(warped, radius=radius, sigma=sigma)
        delta = moving_desc[tuple(rows.T)] - reference
        # Nearest extrapolation clamps each axis on its own.  The depth
        # term is what keeps a pose that has left the image from looking
        # stationary once the descriptor slope has collapsed.
        sample_index = moving_index[sample_flat]
        depth = np.maximum(-sample_index, 0.0) + np.maximum(
            sample_index - upper, 0.0
        )
        signed = np.where(
            sample_index < 0.0,
            sample_index,
            np.where(sample_index > upper, sample_index - upper, 0.0),
        )
        penalty = 1e-3 * float(np.mean(np.sum(np.square(depth), axis=1)))
        loss = float(np.mean(np.sum(np.square(delta), axis=1))) + penalty
        d_descriptor = np.zeros_like(moving_desc)
        d_descriptor[tuple(rows.T)] = (2.0 / count) * delta
        d_warped = _mind_image_gradient(warped, d_descriptor, radius, sigma)
        gradient = _warp_parameter_gradient(
            d_warped.reshape(-1),
            slope,
            inner,
            all_points,
            du_dp,
            transform,
            moving_initial_transform,
        )
        d_penalty_d_index = (2.0 * 1e-3 / count) * signed
        d_penalty_d_point = d_penalty_d_index @ du_dp
        d_penalty_d_inner = _pull_through_outer(
            moving_initial_transform, inner[sample_flat], d_penalty_d_point
        )
        gradient = gradient + _parameter_gradient(
            transform, all_points[sample_flat], d_penalty_d_inner
        )
        if gradient.shape != (theta.shape[0],):
            raise RuntimeError(
                f"MIND gradient shape {gradient.shape} does not match "
                f"{theta.shape[0]} parameters"
            )
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("MIND objective became non-finite")
        return float(loss), gradient

    return objective


def _warp_moving(moving, moving_image, transform, outer, points, du_dp):
    inner = _apply_transform(transform, points)
    moved = _apply_transform(outer, inner)
    moving_index = _physical_to_index(moved, du_dp, moving_image)
    values = np.empty(points.shape[0], dtype=float)
    slope = np.empty_like(points, dtype=float)
    volume = np.ascontiguousarray(moving)[..., None]
    for start, stop in _point_chunks(points.shape[0]):
        sampled, sampled_slope = _sample_linear_channels(
            volume, moving_index[start:stop]
        )
        values[start:stop] = sampled[:, 0]
        slope[start:stop] = sampled_slope[:, 0, :]
    return values, slope, inner, moving_index


def _warp_parameter_gradient(
    d_values, slope, inner, points, du_dp, transform, outer
):
    gradient = np.zeros(int(transform.GetNumberOfParameters()), dtype=float)
    for start, stop in _point_chunks(points.shape[0]):
        d_index = d_values[start:stop, None] * slope[start:stop]
        d_point = d_index @ du_dp
        d_inner = _pull_through_outer(outer, inner[start:stop], d_point)
        gradient += _parameter_gradient(transform, points[start:stop], d_inner)
    return gradient


def _mind_image_gradient(image, d_descriptor, radius, sigma):
    """``d(loss)/d(image)`` for a descriptor loss, matching ``mind_descriptor``."""

    array = np.asarray(image, dtype=float)
    offsets = _neighbour_offsets(array.ndim, int(radius))
    kernel = _gaussian_kernel(float(sigma))
    distances = np.empty(array.shape + (len(offsets),), dtype=float)
    shifted_images = []
    for index, offset in enumerate(offsets):
        shifted = _shift_nearest(array, offset)
        shifted_images.append(shifted)
        distances[..., index] = gaussian_filter(
            np.square(array - shifted), sigma=float(sigma), mode="nearest"
        )
    centered = distances - np.min(distances, axis=-1, keepdims=True)
    scale = float(np.mean(centered)) + 1e-8
    weights = np.exp(-centered / scale)
    norm = np.sum(weights, axis=-1, keepdims=True) + 1e-12
    descriptor = weights / norm

    upstream = np.asarray(d_descriptor, dtype=float)
    dot = np.sum(upstream * descriptor, axis=-1, keepdims=True)
    d_weights = (upstream - dot) / norm
    d_scale = float(np.sum(d_weights * weights * centered / (scale**2)))
    d_centered = d_weights * (-weights / scale)
    d_centered = d_centered + (d_scale / centered.size)
    d_distances = np.array(d_centered, copy=True)
    sum_d_centered = np.sum(d_centered, axis=-1)
    argmin = np.argmin(distances, axis=-1)
    d_distances[(*np.indices(argmin.shape), argmin)] -= sum_d_centered

    d_array = np.zeros_like(array)
    for index, offset in enumerate(offsets):
        d_square = _gaussian_adjoint(d_distances[..., index], kernel)
        diff = array - shifted_images[index]
        d_diff = d_square * (2.0 * diff)
        d_array += d_diff
        d_array += _shift_nearest_adjoint(-d_diff, array.shape, offset)
    return d_array


def _gaussian_kernel(sigma, truncate=4.0):
    radius = int(truncate * float(sigma) + 0.5)
    positions = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (positions / float(sigma)) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def _gaussian_adjoint(upstream, kernel):
    """Adjoint of ``gaussian_filter(..., mode='nearest')`` for this kernel."""

    gradient = np.asarray(upstream, dtype=float)
    for axis in reversed(range(gradient.ndim)):
        gradient = _gaussian_axis_adjoint(gradient, kernel, axis)
    return gradient


def _gaussian_axis_adjoint(upstream, kernel, axis):
    radius = len(kernel) // 2
    length = upstream.shape[axis]
    positions = np.arange(length)
    gradient = np.zeros_like(upstream)
    moved = np.moveaxis(gradient, axis, 0)
    for shift, weight in zip(range(-radius, radius + 1), kernel):
        source = np.clip(positions + shift, 0, length - 1)
        np.add.at(moved, source, weight * np.moveaxis(upstream, axis, 0))
    return gradient


def _shift_nearest(array, offset):
    """``array[clip(index + offset)]``, matching integer ``map_coordinates``."""

    shifted = array
    for axis, shift in enumerate(offset):
        length = shifted.shape[axis]
        source = np.clip(np.arange(length) + int(shift), 0, length - 1)
        shifted = np.take(shifted, source, axis=axis)
    return shifted


def _shift_nearest_adjoint(upstream, shape, offset):
    gradient = np.asarray(upstream, dtype=float)
    for axis in reversed(range(len(offset))):
        length = shape[axis]
        source = np.clip(np.arange(length) + int(offset[axis]), 0, length - 1)
        scattered = np.zeros_like(gradient)
        np.add.at(np.moveaxis(scattered, axis, 0), source, np.moveaxis(gradient, axis, 0))
        gradient = scattered
    return gradient


def _point_chunks(count, size=_OBJECTIVE_CHUNK):
    start = 0
    while start < count:
        stop = min(count, start + size)
        yield start, stop
        start = stop


def _sample_linear_channels(volume, index_xyz):
    """Values and exact linear slope at SimpleITK indices.

    ``volume`` is ``(..., channels)`` in NumPy axis order.  The slope is
    ``d(value)/d(index_xyz)`` of ``map_coordinates(..., order=1,
    mode="nearest")``: along the differentiated axis it is the forward
    difference of the cell that contains the coordinate, and the other
    axes stay linear.  Nearest lookup makes an out-of-range axis (and any
    singleton axis) contribute slope zero without clearing the axes that
    are still inside the image.
    """

    spatial = volume.shape[:-1]
    coords = np.asarray(index_xyz, dtype=float)[:, ::-1]
    dimension = coords.shape[1]
    if len(spatial) != dimension:
        raise ValueError(
            f"MIND sample dimension {dimension} does not match volume {spatial}"
        )
    count = coords.shape[0]
    floors = np.floor(coords).astype(np.int64)
    fracs = coords - floors
    sizes = np.asarray(spatial, dtype=np.int64)
    channels = volume.shape[-1]

    def gather(index):
        clipped = np.clip(index, 0, sizes - 1)
        return volume[tuple(clipped.T)]

    values = np.zeros((count, channels), dtype=float)
    for mask in range(1 << dimension):
        weight = np.ones(count, dtype=float)
        index = np.empty((count, dimension), dtype=np.int64)
        for axis in range(dimension):
            bit = (mask >> axis) & 1
            fraction = fracs[:, axis]
            weight *= fraction if bit else (1.0 - fraction)
            index[:, axis] = floors[:, axis] + bit
        values += weight[:, None] * gather(index)

    gradient_numpy = np.zeros((count, channels, dimension), dtype=float)
    for axis in range(dimension):
        others = [item for item in range(dimension) if item != axis]
        diff = np.zeros((count, channels), dtype=float)
        for mask in range(1 << len(others)):
            weight = np.ones(count, dtype=float)
            index_lo = floors.copy()
            index_hi = floors.copy()
            index_hi[:, axis] = floors[:, axis] + 1
            for bit_index, other in enumerate(others):
                bit = (mask >> bit_index) & 1
                fraction = fracs[:, other]
                weight *= fraction if bit else (1.0 - fraction)
                source = floors[:, other] + bit
                index_lo[:, other] = source
                index_hi[:, other] = source
            diff += weight[:, None] * (gather(index_hi) - gather(index_lo))
        # Exactly on a voxel the two adjacent cells disagree.  Use their
        # average there: it is the slope central differences see, and it
        # is what a rigid start (every sample on an integer) needs in
        # order to move toward the translation.  Inside a cell the
        # derivative is the forward difference alone.
        on_grid = np.isclose(fracs[:, axis], 0.0)
        if np.any(on_grid):
            left = np.zeros_like(diff)
            for mask in range(1 << len(others)):
                weight = np.ones(count, dtype=float)
                index_lo = floors.copy()
                index_hi = floors.copy()
                index_lo[:, axis] = floors[:, axis] - 1
                for bit_index, other in enumerate(others):
                    bit = (mask >> bit_index) & 1
                    fraction = fracs[:, other]
                    weight *= fraction if bit else (1.0 - fraction)
                    source = floors[:, other] + bit
                    index_lo[:, other] = source
                    index_hi[:, other] = source
                left += weight[:, None] * (gather(index_hi) - gather(index_lo))
            diff = np.where(on_grid[:, None], 0.5 * (diff + left), diff)
        gradient_numpy[:, :, axis] = diff
    return values, np.ascontiguousarray(gradient_numpy[:, :, ::-1])


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
    points = np.asarray(points, dtype=float)
    if transform is None or points.shape[0] == 0:
        return points
    name = transform.GetName()
    if name == "Euler2DTransform":
        return _apply_euler2d(transform, points)
    if name == "Euler3DTransform":
        return _apply_euler3d(transform, points)
    if name == "AffineTransform":
        return _apply_affine(transform, points)
    if name == "BSplineTransform":
        return _apply_bspline(transform, points)
    mapped = [
        transform.TransformPoint(tuple(float(value) for value in point))
        for point in points
    ]
    return np.asarray(mapped, dtype=float)


def _apply_centered(matrix, center, translation, points):
    return (points - center) @ matrix.T + center + translation


def _apply_euler2d(transform, points):
    parameters = np.array(transform.GetParameters(), dtype=float)
    angle = float(parameters[0])
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    matrix = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    center = np.array(transform.GetCenter(), dtype=float)
    return _apply_centered(matrix, center, parameters[1:], points)


def _apply_euler3d(transform, points):
    matrix = _euler3d_matrix(transform)
    parameters = np.array(transform.GetParameters(), dtype=float)
    center = np.array(transform.GetCenter(), dtype=float)
    return _apply_centered(matrix, center, parameters[3:], points)


def _apply_affine(transform, points):
    dimension = points.shape[1]
    parameters = np.array(transform.GetParameters(), dtype=float)
    matrix = parameters[: dimension * dimension].reshape(dimension, dimension)
    center = np.array(transform.GetCenter(), dtype=float)
    return _apply_centered(
        matrix, center, parameters[dimension * dimension :], points
    )


def _apply_bspline(transform, points):
    if int(transform.GetOrder()) != 3:
        raise ValueError("MIND only evaluates the cubic SimpleITK B-spline")
    output = np.empty_like(points, dtype=float)
    for start, stop in _point_chunks(points.shape[0]):
        output[start:stop] = _apply_bspline_chunk(transform, points[start:stop])
    return output


def _apply_bspline_chunk(transform, points):
    dimension = points.shape[1]
    coefficient = transform.GetCoefficientImages()[0]
    origin = np.array(coefficient.GetOrigin(), dtype=float)
    spacing = np.array(coefficient.GetSpacing(), dtype=float)
    direction = np.array(coefficient.GetDirection(), dtype=float).reshape(
        dimension, dimension
    )
    control = np.array(coefficient.GetSize(), dtype=int)
    n_control = int(np.prod(control))
    parameters = np.array(transform.GetParameters(), dtype=float).reshape(
        dimension, n_control
    )
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
    index = support[:, :, None] + np.arange(4)[None, None, :]
    weight = _cubic_bspline(continuous[:, :, None] - index)
    valid = (index >= 0) & (index < control[None, :, None])
    weight = np.where(valid, weight, 0.0)
    weight[~inside] = 0.0
    if dimension == 2:
        separable = weight[:, 0, :, None] * weight[:, 1, None, :]
        ix = index[:, 0, :, None]
        iy = index[:, 1, None, :]
        usable = (ix >= 0) & (ix < control[0]) & (iy >= 0) & (iy < control[1])
        flat = np.where(usable, ix + control[0] * iy, 0)
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
        flat = np.where(usable, flat, 0)
    else:
        raise ValueError(f"MIND B-spline does not support dimension {dimension}")
    separable = np.where(usable, separable, 0.0)
    flat = np.asarray(flat, dtype=int)
    displacement = np.zeros_like(points, dtype=float)
    axes = tuple(range(1, separable.ndim))
    for axis in range(dimension):
        displacement[:, axis] = np.sum(parameters[axis][flat] * separable, axis=axes)
    return points + displacement


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


def _cubic_bspline(distance: np.ndarray) -> np.ndarray:
    """Cardinal cubic B-spline.  Nonzero only for ``abs(distance) < 2``."""

    absolute = np.abs(distance)
    weight = np.zeros_like(absolute, dtype=float)
    near = absolute < 1.0
    far = (absolute >= 1.0) & (absolute < 2.0)
    weight[near] = (2.0 / 3.0) - absolute[near] ** 2 + 0.5 * absolute[near] ** 3
    weight[far] = ((2.0 - absolute[far]) ** 3) / 6.0
    return weight
