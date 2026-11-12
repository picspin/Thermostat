"""Optional single-case INR volume registration.

This module is an explicit opt-in alternative to the default SimpleITK
B-spline registrar in ``registration.py``.  Importing it does not change
that path, its parameters, or its output, and nothing here is called by the
temperature calculation.

One case is fit from scratch.  There is no training set.  A coordinate MLP
with four hidden layers of 64 units represents ``R`` scalar potentials, and
the spatial motion basis is the analytic Jacobian of those potentials.  A
rank-2 or rank-3 coefficient vector per phase is the low-rank respiratory
model (depth plus hysteresis).  The objective uses only localizer projection
consistency and the MIND cross-phase distance.

``low_rank_sparse_decomposition`` is a separate optional analysis step.  It
is not a registration and it is not part of temperature calculation.

The only dependencies are numpy and scipy, which the project already
requires.  No GPU and no real DICOM data are required.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from mvc.models.registration import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_WARNING,
    RegistrationResult,
)

_HIDDEN_LAYERS = 4
_HIDDEN_UNITS = 64
_SUPPORTED_RANKS = (2, 3)


@dataclass(frozen=True)
class INRRegistrationConfig:
    """Parameters for the opt-in single-case INR registrar.

    The defaults are small enough for a synthetic case on CPU.  They are
    independent of ``RegistrationConfig`` and cannot change the SimpleITK
    B-spline defaults: the two configs are different types.
    """

    rank: int = 2
    hidden_layers: int = _HIDDEN_LAYERS
    hidden_units: int = _HIDDEN_UNITS
    iterations: int = 40
    learning_rate: float = 0.15
    weight_decay: float = 1e-4
    spatial_samples: int = 256
    projection_weight: float = 1.0
    mind_weight: float = 1.0
    temporal_smooth_weight: float = 0.05
    displacement_scale: float = 2.0
    mind_radius: int = 1
    mind_sigma: float = 0.5
    random_seed: int = 1234
    convergence_window: int = 8
    convergence_min_delta: float = 1e-7
    displacement_warning: float = 6.0
    gradient_clip: float = 5.0

    def __post_init__(self) -> None:
        if self.rank not in _SUPPORTED_RANKS:
            raise ValueError(f"rank must be one of {_SUPPORTED_RANKS}, got {self.rank}")
        if self.hidden_layers != _HIDDEN_LAYERS or self.hidden_units != _HIDDEN_UNITS:
            raise ValueError(
                "this registrar is specified as a 4-layer, 64-unit coordinate MLP"
            )
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.spatial_samples < 1:
            raise ValueError("spatial_samples must be >= 1")
        if self.mind_radius < 1:
            raise ValueError("mind_radius must be >= 1")
        if self.mind_sigma <= 0.0:
            raise ValueError("mind_sigma must be positive")
        if self.displacement_scale <= 0.0:
            raise ValueError("displacement_scale must be positive")


@dataclass(frozen=True)
class SpatialBasis:
    """Analytic spatial basis at a batch of coordinates.

    ``values`` has shape ``(N, R)`` and is the MLP output.  ``jacobian`` has
    shape ``(N, R, D)`` and is ``d(values)/d(coordinates)``.  The deformable
    part of the motion basis is ``jacobian``.  A separate per-phase
    translation is added outside the network, because a ``tanh`` network with
    small weights has a near-zero mean Jacobian and cannot represent a bulk
    shift on its own.
    """

    values: np.ndarray
    jacobian: np.ndarray


@dataclass
class LowRankSparseResult:
    """Low-rank background plus sparse change.  Not a registration result."""

    low_rank: np.ndarray
    sparse: np.ndarray
    rank: int
    sparse_fraction: float
    iterations: int
    converged: bool
    relative_change: float


@dataclass
class _LayerTape:
    """One dense layer, retained for the analytic backward pass."""

    weight: np.ndarray
    activation_in: np.ndarray
    jacobian_in: np.ndarray
    pre_jacobian: np.ndarray
    activation_out: np.ndarray
    derivative: np.ndarray | None


class CoordinateMLP:
    """Four-hidden-layer coordinate network with analytic derivatives.

    Architecture, fixed by the specification::

        normalized coordinates (D)
          -> Dense(64) + tanh
          -> Dense(64) + tanh
          -> Dense(64) + tanh
          -> Dense(64) + tanh
          -> Dense(R)

    ``R`` is 2 or 3.  Hidden activations are ``tanh``, whose derivative
    ``1 - tanh(pre)^2`` is exact, so both the spatial Jacobian and the
    parameter gradients are closed form.  Weights start from a small
    deterministic draw and are fit to the single case.
    """

    def __init__(
        self,
        input_dim: int,
        rank: int,
        hidden_layers: int = _HIDDEN_LAYERS,
        hidden_units: int = _HIDDEN_UNITS,
        seed: int = 1234,
    ) -> None:
        if hidden_layers != _HIDDEN_LAYERS or hidden_units != _HIDDEN_UNITS:
            raise ValueError("CoordinateMLP is fixed at 4 hidden layers of 64 units")
        if input_dim not in (2, 3):
            raise ValueError("input_dim must be 2 or 3")
        if rank not in _SUPPORTED_RANKS:
            raise ValueError(f"rank must be one of {_SUPPORTED_RANKS}, got {rank}")

        self.input_dim = int(input_dim)
        self.rank = int(rank)
        self.hidden_layers = int(hidden_layers)
        self.hidden_units = int(hidden_units)
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        fan_in = self.input_dim
        for _ in range(self.hidden_layers):
            scale = np.sqrt(1.0 / fan_in)
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, self.hidden_units)))
            self.biases.append(np.zeros(self.hidden_units, dtype=float))
            fan_in = self.hidden_units
        out_scale = np.sqrt(1.0 / fan_in)
        self.weights.append(rng.normal(0.0, out_scale, size=(fan_in, self.rank)))
        self.biases.append(np.zeros(self.rank, dtype=float))

    def evaluate(self, coordinates: np.ndarray) -> SpatialBasis:
        """Return the potentials and their analytic spatial Jacobian."""
        values, jacobian, _tape = self._forward(coordinates)
        return SpatialBasis(values=values, jacobian=jacobian)

    def jacobian_parameter_gradients(
        self,
        coordinates: np.ndarray,
        jacobian_upstream: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Analytic ``d(loss)/d(weights, biases)`` for a Jacobian upstream.

        ``jacobian_upstream`` has shape ``(N, R, D)`` and is
        ``d(loss)/d(spatial Jacobian)``.  Bias gradients are nonzero because a
        hidden bias changes ``tanh`` and therefore changes the Jacobian, even
        though the output layer's own bias does not.
        """
        _values, _jacobian, tape = self._forward(coordinates)
        return _backward_jacobian(tape, jacobian_upstream)

    def _forward(self, coordinates: np.ndarray):
        coords = np.asarray(coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != self.input_dim:
            raise ValueError(
                f"coordinates must have shape (N, {self.input_dim}), got {coords.shape}"
            )
        activation = coords
        jacobian = np.broadcast_to(
            np.eye(self.input_dim, dtype=float),
            (coords.shape[0], self.input_dim, self.input_dim),
        ).copy()
        tape: list[_LayerTape] = []
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            pre_jacobian = np.einsum("nid,io->nod", jacobian, weight)
            pre = activation @ weight + bias
            if layer < self.hidden_layers:
                activated = np.tanh(pre)
                derivative = 1.0 - np.square(activated)
                out_jacobian = derivative[:, :, None] * pre_jacobian
            else:
                activated = pre
                derivative = None
                out_jacobian = pre_jacobian
            tape.append(
                _LayerTape(
                    weight=weight,
                    activation_in=activation,
                    jacobian_in=jacobian,
                    pre_jacobian=pre_jacobian,
                    activation_out=activated,
                    derivative=derivative,
                )
            )
            activation = activated
            jacobian = out_jacobian
        if jacobian.shape != (coords.shape[0], self.rank, self.input_dim):
            raise RuntimeError(
                f"spatial Jacobian shape {jacobian.shape} != "
                f"{(coords.shape[0], self.rank, self.input_dim)}"
            )
        return activation, jacobian, tape


def _backward_jacobian(
    tape: Sequence[_LayerTape],
    upstream: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Reverse mode through the spatial-Jacobian recurrence.

    A hidden layer computes ``J = (1 - h^2) * (J_in @ W)`` with
    ``h = tanh(a_in W + b)``.  Later layers depend on both ``J`` and ``h``,
    so the backward pass carries two upstreams: ``d(loss)/dJ`` and
    ``d(loss)/dh``.  The output layer is linear, so it starts with a zero
    value upstream.  A hidden bias gets a gradient only because it moves
    ``h`` and therefore moves ``1 - h^2``.
    """
    grad_jacobian = np.asarray(upstream, dtype=float)
    # The motion basis is the Jacobian.  No loss term reads the output
    # potentials themselves, so their value upstream starts at zero.
    grad_values = np.zeros(tape[-1].activation_out.shape, dtype=float)
    weight_grads = [np.zeros_like(layer.weight) for layer in tape]
    bias_grads = [np.zeros(layer.weight.shape[1], dtype=float) for layer in tape]
    for index in range(len(tape) - 1, -1, -1):
        layer = tape[index]
        if grad_jacobian.shape[1] != layer.weight.shape[1]:
            raise RuntimeError(
                f"Jacobian upstream width {grad_jacobian.shape[1]} does not match "
                f"layer {index} output {layer.weight.shape[1]}"
            )
        if grad_values.shape != layer.activation_out.shape:
            raise RuntimeError(
                f"value upstream shape {grad_values.shape} does not match "
                f"layer {index} activation {layer.activation_out.shape}"
            )
        if layer.derivative is None:
            pre_upstream = grad_jacobian
            pre_values = grad_values
        else:
            # dJ/d(J_in W) = 1 - h^2, and d(1-h^2)/dh = -2h.
            pre_upstream = grad_jacobian * layer.derivative[:, :, None]
            through_slope = np.sum(grad_jacobian * layer.pre_jacobian, axis=2)
            hidden_values = grad_values + through_slope * (-2.0 * layer.activation_out)
            pre_values = hidden_values * layer.derivative
        weight_grads[index] = (
            layer.activation_in.T @ pre_values
            + np.einsum("nid,nod->io", layer.jacobian_in, pre_upstream)
        )
        bias_grads[index] = np.sum(pre_values, axis=0)
        grad_jacobian = np.einsum("nod,io->nid", pre_upstream, layer.weight)
        grad_values = pre_values @ layer.weight.T
    return weight_grads, bias_grads


class INRVolumeRegistrar:
    """Single-case registrar.  Opt-in, and not the default motion correction.

    For normalized coordinate ``x`` and phase ``t`` the displacement is

        u(x, t) = s * tanh(J(x) c_t + a_t)

    ``J`` is the ``(R, D)`` analytic Jacobian of the coordinate MLP and is the
    spatial basis.  ``c_t`` has length ``R`` and ``a_t`` has length ``D``.
    Together they are the low-rank respiratory model: ``a_t`` is the bulk
    depth and ``c_t`` (rank 2 or 3) lets the return path differ from the
    outbound path, which is the hysteresis.  ``s`` is ``displacement_scale``
    in voxels.  The reference phase holds both vectors at zero.

    ``J c`` is a sum of analytic gradient fields, so it is curl-free: it can
    represent a smooth expansion or a respiratory-like bend, but not a large
    rotation or a shear.  The explicit ``a_t`` is required because a
    small-weight ``tanh`` network has a near-zero mean Jacobian and cannot
    move the whole volume by itself.
    """

    def __init__(self, config: INRRegistrationConfig | None = None) -> None:
        self.config = config or INRRegistrationConfig()

    def register_complex(
        self,
        fixed_real: np.ndarray,
        fixed_imag: np.ndarray | None,
        moving_real: np.ndarray,
        moving_imag: np.ndarray | None,
        config: INRRegistrationConfig | None = None,
    ) -> RegistrationResult:
        """Register one moving phase to one fixed phase.

        The signature matches ``SimpleITKRegistrar.register_complex``.  The
        field is estimated on magnitudes when both components exist, then the
        same field resamples the real and imaginary parts.

        ``config`` must be an ``INRRegistrationConfig`` or omitted.  The
        SimpleITK ``RegistrationConfig`` is a different object and is rejected
        so the two registrars cannot silently share parameters.
        """
        if config is not None and not isinstance(config, INRRegistrationConfig):
            raise ValueError(
                "INRVolumeRegistrar requires INRRegistrationConfig; "
                "RegistrationConfig is only for the SimpleITK B-spline path"
            )
        config = config or self.config
        fixed = _as_magnitude(fixed_real, fixed_imag)
        moving = _as_magnitude(moving_real, moving_imag)
        if fixed.shape != moving.shape:
            raise ValueError("fixed and moving images must have the same shape")
        start = time.perf_counter()
        try:
            fitted = self.register_volume(
                np.stack([fixed, moving], axis=0),
                config=config,
                reference_index=0,
            )
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced through the result
            return _failed_result(config, str(exc), time.perf_counter() - start)
        if fitted.status == STATUS_FAILED or fitted.transform is None:
            return fitted

        displacement = fitted.transform["displacement"][1]
        warped_real = _resample_phase(np.asarray(moving_real, dtype=float), displacement)
        warped_imag = None
        if moving_imag is not None:
            warped_imag = _resample_phase(
                np.asarray(moving_imag, dtype=float), displacement
            )
        return RegistrationResult(
            warped_real=warped_real,
            warped_imag=warped_imag,
            transform=fitted.transform,
            config=config,
            metric_name=fitted.metric_name,
            final_metric=fitted.final_metric,
            dice=fitted.dice,
            iterations=fitted.iterations,
            stop_condition=fitted.stop_condition,
            elapsed_s=time.perf_counter() - start,
            status=fitted.status,
            message=fitted.message,
        )

    def register_volume(
        self,
        volume: np.ndarray,
        config: INRRegistrationConfig | None = None,
        reference_index: int = 0,
    ) -> RegistrationResult:
        """Register every phase of one case to ``volume[reference_index]``.

        ``volume`` has shape ``(T, H, W)`` or ``(T, D, H, W)`` with ``T >= 2``.
        The input is not modified.  ``warped_real`` is the warped comparison
        phase; the whole series, the displacement and the coefficients are
        on ``transform``.
        """
        if config is not None and not isinstance(config, INRRegistrationConfig):
            raise ValueError(
                "INRVolumeRegistrar requires INRRegistrationConfig; "
                "RegistrationConfig is only for the SimpleITK B-spline path"
            )
        config = config or self.config
        start = time.perf_counter()
        try:
            data = _validate_volume(volume)
            if not 0 <= int(reference_index) < data.shape[0]:
                raise ValueError("reference_index is outside the phase axis")
            reference_index = int(reference_index)
            spatial_shape = data.shape[1:]
            coords, _flat = _sample_coordinates(
                spatial_shape, config.spatial_samples, config.random_seed
            )
            mlp = CoordinateMLP(
                input_dim=len(spatial_shape),
                rank=config.rank,
                hidden_layers=config.hidden_layers,
                hidden_units=config.hidden_units,
                seed=config.random_seed,
            )
            coefficients = np.zeros((data.shape[0], config.rank), dtype=float)
            translations = np.zeros((data.shape[0], len(spatial_shape)), dtype=float)
            prepared = _prepare_phase_terms(data, coords, config)
            trace = _optimize(
                mlp,
                coefficients,
                translations,
                coords,
                prepared,
                reference_index,
                config,
            )
            displacement = _dense_displacement(
                mlp, coefficients, translations, spatial_shape, config
            )
            warped = _warp_volume(data, displacement)
            _check_shapes(
                data, warped, displacement, coefficients, translations, config.rank
            )
            metrics = _registration_metrics(
                data, warped, displacement, reference_index
            )
            status, message = _status_from_metrics(metrics, config)
            moving_index = _comparison_index(data.shape[0], reference_index)
            return RegistrationResult(
                warped_real=warped[moving_index],
                warped_imag=None,
                transform={
                    "kind": "inr_low_rank",
                    "displacement": displacement,
                    "coefficients": coefficients.copy(),
                    "translations": translations.copy(),
                    "warped": warped,
                    "reference_index": reference_index,
                    "loss_history": trace["loss"],
                    "projection_history": trace["projection"],
                    "mind_history": trace["mind"],
                    "metrics": metrics,
                    "rank": int(config.rank),
                },
                config=config,
                metric_name="projection+MIND",
                final_metric=float(trace["loss"][-1]),
                dice=float(metrics["dice"]),
                iterations=int(trace["iterations"]),
                stop_condition=str(trace["stop_condition"]),
                elapsed_s=time.perf_counter() - start,
                status=status,
                message=message,
            )
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced through the result
            return _failed_result(config, str(exc), time.perf_counter() - start)


def _optimize(
    mlp, coefficients, translations, coords, prepared, reference_index, config
) -> dict:
    loss_history: list[float] = []
    projection_history: list[float] = []
    mind_history: list[float] = []
    previous = None
    stable = 0
    stop = "max iterations"
    iterations = 0
    momentum = 0.8
    velocity_w = [np.zeros_like(weight) for weight in mlp.weights]
    velocity_b = [np.zeros_like(bias) for bias in mlp.biases]
    velocity_c = np.zeros_like(coefficients)
    velocity_a = np.zeros_like(translations)

    for iteration in range(config.iterations):
        iterations = iteration + 1
        terms = _objective_and_gradients(
            mlp,
            coefficients,
            translations,
            coords,
            prepared,
            reference_index,
            config,
        )
        total = (
            config.projection_weight * terms["projection"]
            + config.mind_weight * terms["mind"]
            + config.temporal_smooth_weight * terms["temporal"]
        )
        if not np.isfinite(total):
            raise FloatingPointError("INR objective became non-finite")
        loss_history.append(float(total))
        projection_history.append(float(terms["projection"]))
        mind_history.append(float(terms["mind"]))

        # The schedule only damps later iterations.  One learning rate serves
        # every parameter group because the update divides by that group's
        # own RMS; see ``_update_vector``.
        step = config.learning_rate / (1.0 + np.sqrt(iteration))
        _update_vector(
            coefficients,
            terms["coefficient_grad"],
            velocity_c,
            step,
            momentum,
            config.gradient_clip,
        )
        _update_vector(
            translations,
            terms["translation_grad"],
            velocity_a,
            step,
            momentum,
            config.gradient_clip,
        )
        coefficients[reference_index] = 0.0
        translations[reference_index] = 0.0
        for index, weight in enumerate(mlp.weights):
            gradient = terms["weight_grads"][index] + config.weight_decay * weight
            _update_vector(
                weight,
                gradient,
                velocity_w[index],
                step,
                momentum,
                config.gradient_clip,
            )
            _update_vector(
                mlp.biases[index],
                terms["bias_grads"][index],
                velocity_b[index],
                step,
                momentum,
                config.gradient_clip,
            )
        if previous is not None and abs(previous - total) < config.convergence_min_delta:
            stable += 1
        else:
            stable = 0
        previous = float(total)
        if stable >= config.convergence_window:
            stop = (
                f"convergence window {config.convergence_window} "
                f"with delta < {config.convergence_min_delta}"
            )
            break

    return {
        "loss": loss_history,
        "projection": projection_history,
        "mind": mind_history,
        "iterations": iterations,
        "stop_condition": stop,
    }


def _objective_and_gradients(
    mlp: CoordinateMLP,
    coefficients: np.ndarray,
    translations: np.ndarray,
    coords: np.ndarray,
    prepared: dict,
    reference_index: int,
    config: INRRegistrationConfig,
) -> dict:
    basis = mlp.evaluate(coords)
    projection_acc = 0.0
    mind_acc = 0.0
    comparisons = 0
    jacobian_upstream = np.zeros_like(basis.jacobian)
    coefficient_grad = np.zeros_like(coefficients)
    translation_grad = np.zeros_like(translations)
    reference = prepared[reference_index]

    for phase, coefficient in enumerate(coefficients):
        if phase == reference_index:
            continue
        field = np.einsum("nrd,r->nd", basis.jacobian, coefficient) + translations[phase]
        activated = np.tanh(field)
        displacement = config.displacement_scale * activated
        d_disp_d_field = config.displacement_scale * (1.0 - np.square(activated))
        phase_data = prepared[phase]
        position = _sample_positions(coords, phase_data["image"].shape, displacement)
        moved, image_slope = _sample_scalar(phase_data["image"], position)
        projection, d_intensity = _projection_loss_and_intensity_grad(
            moved,
            reference["samples"],
            coords,
            phase_data["image"].shape,
        )
        moved_mind, mind_slope = _sample_linear(phase_data["mind"], position)
        mind, d_mind = _mind_loss_and_grad(moved_mind, reference["mind_samples"])
        # Slope of the same linear interpolant that produced ``moved_mind``.
        # A precomputed central-difference volume is a different operator and
        # stores a full descriptor gradient that a 3D phase does not need.
        mind_gradient = np.einsum("nc,ncd->nd", d_mind, mind_slope)
        d_displacement = (
            config.projection_weight * d_intensity[:, None] * image_slope
            + config.mind_weight * mind_gradient
        )
        d_field = d_displacement * d_disp_d_field
        # field = J c + a, so d(loss)/dJ = d_field c^T, d(loss)/dc = J^T d_field
        # and d(loss)/da = sum_samples d_field.
        jacobian_upstream += d_field[:, None, :] * coefficient[None, :, None]
        coefficient_grad[phase] = np.einsum("nrd,nd->r", basis.jacobian, d_field)
        translation_grad[phase] = np.sum(d_field, axis=0)
        projection_acc += projection
        mind_acc += mind
        comparisons += 1

    if comparisons == 0:
        raise ValueError("volume must contain a phase other than the reference")
    # ``d_intensity`` and ``d_mind`` are already derivatives of means, and the
    # sums above fold in every sample.  Dividing by the sample count a second
    # time shrinks the data term by that factor relative to weight decay and
    # the temporal penalty, so a 64-wide layer only feels the decay.  Average
    # across phases only.
    jacobian_upstream /= comparisons
    coefficient_grad /= comparisons
    translation_grad /= comparisons
    weight_grads, bias_grads = mlp.jacobian_parameter_gradients(coords, jacobian_upstream)
    temporal_grad = _temporal_gradient(coefficients, translations, reference_index)
    coefficient_grad += config.temporal_smooth_weight * temporal_grad[0]
    translation_grad += config.temporal_smooth_weight * temporal_grad[1]
    coefficient_grad[reference_index] = 0.0
    translation_grad[reference_index] = 0.0
    return {
        "projection": projection_acc / comparisons,
        "mind": mind_acc / comparisons,
        "temporal": _temporal_smoothness(coefficients, translations),
        "coefficient_grad": coefficient_grad,
        "translation_grad": translation_grad,
        "weight_grads": weight_grads,
        "bias_grads": bias_grads,
    }


def _prepare_phase_terms(volume: np.ndarray, coords: np.ndarray, config) -> dict:
    """Images and MIND descriptors are constant during the fit."""
    zero = np.zeros((coords.shape[0], volume.ndim - 1), dtype=float)
    prepared = {}
    for phase in range(volume.shape[0]):
        image = volume[phase]
        descriptor = mind_descriptor(
            image, radius=config.mind_radius, sigma=config.mind_sigma
        )
        position = _sample_positions(coords, image.shape, zero)
        prepared[phase] = {
            "image": image,
            "mind": descriptor,
            "samples": _sample_scalar(image, position)[0],
            "mind_samples": _sample_linear(descriptor, position)[0],
        }
    return prepared


def _update_vector(target, gradient, velocity, step, momentum, clip):
    if gradient.shape != target.shape:
        raise RuntimeError(
            f"gradient shape {gradient.shape} does not match parameter {target.shape}"
        )
    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError("INR gradient became non-finite")
    # Each parameter group has its own scale: a translation, a 64×64 weight
    # and a bias do not share units.  Dividing by the RMS makes one learning
    # rate meaningful for all of them.  ``clip`` bounds that RMS.  It must
    # not bound the L2 norm: after this division a 64×64 matrix has L2 64,
    # and capping that at 5 multiplies the step by 5/64, which freezes the
    # spatial basis.  A finite RMS-normalized direction already has RMS 1, so
    # the default clip of 5 does not shrink it.  Non-finite gradients are
    # rejected above.
    scale = float(np.sqrt(np.mean(np.square(gradient)))) + 1e-8
    direction = gradient / scale
    direction_rms = float(np.sqrt(np.mean(np.square(direction))))
    if direction_rms > clip > 0.0:
        direction *= clip / direction_rms
    velocity *= momentum
    velocity += (1.0 - momentum) * direction
    target -= step * velocity


def _dense_displacement(
    mlp, coefficients, translations, spatial_shape, config
) -> np.ndarray:
    grid = _dense_coordinates(spatial_shape).reshape(-1, len(spatial_shape))
    jacobian = mlp.evaluate(grid).jacobian
    frames = []
    for coefficient, translation in zip(coefficients, translations):
        field = np.einsum("nrd,r->nd", jacobian, coefficient) + translation
        displacement = config.displacement_scale * np.tanh(field)
        frames.append(displacement.reshape(spatial_shape + (len(spatial_shape),)))
    stacked = np.stack(frames, axis=0)
    expected = (coefficients.shape[0],) + tuple(spatial_shape) + (len(spatial_shape),)
    if stacked.shape != expected:
        raise RuntimeError(f"dense displacement shape {stacked.shape} != {expected}")
    return stacked


def mind_descriptor(image: np.ndarray, radius: int = 1, sigma: float = 0.5) -> np.ndarray:
    """Modality-independent neighbourhood descriptor.

    Each voxel is compared with the neighbours in a Chebyshev ball (8 in 2D
    and 26 in 3D when ``radius`` is 1).  Distances are Gaussian-smoothed
    squared differences, converted with a softmax so a global intensity shift
    does not change the descriptor.  The result has shape
    ``image.shape + (neighbours,)``.
    """
    array = np.asarray(image, dtype=float)
    if array.ndim not in (2, 3):
        raise ValueError("MIND supports 2D or 3D images")
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    offsets = _neighbour_offsets(array.ndim, radius)
    distances = np.empty(array.shape + (len(offsets),), dtype=float)
    for index, offset in enumerate(offsets):
        shifted = map_coordinates(
            array,
            _offset_grid(array.shape, offset),
            order=1,
            mode="nearest",
            prefilter=False,
        ).reshape(array.shape)
        distances[..., index] = gaussian_filter(
            np.square(array - shifted), sigma=sigma, mode="nearest"
        )
    centered = distances - np.min(distances, axis=-1, keepdims=True)
    weights = np.exp(-centered / (float(np.mean(centered)) + 1e-8))
    return weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-12)


def low_rank_sparse_decomposition(
    series: np.ndarray,
    rank: int = 2,
    sparsity: float = 0.05,
    iterations: int = 30,
    tolerance: float = 1e-4,
) -> LowRankSparseResult:
    """Split a phase series into low-rank background and sparse change.

    This is robust PCA by alternating a rank truncation with a hard threshold
    on the residual, ``series = L + S``.  It is an optional analysis step.
    Temperature calculation does not call it, and it does not modify the
    default registration result.

    ``series`` has shape ``(T, *spatial)``.  ``sparsity`` is the fraction of
    entries allowed to remain in ``S``.
    """
    array = np.asarray(series, dtype=float)
    if array.ndim < 2:
        raise ValueError("series must have a phase axis and one spatial axis")
    if not np.all(np.isfinite(array)):
        raise ValueError("series contains a non-finite value")
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    matrix = np.reshape(array, (array.shape[0], -1)).T
    if rank > min(matrix.shape):
        raise ValueError(f"rank {rank} exceeds the series shape {matrix.shape}")
    low_rank = np.zeros_like(matrix)
    sparse = np.zeros_like(matrix)
    keep = max(1, round(sparsity * matrix.size))
    relative = float("inf")
    used = 0
    converged = False
    for used in range(1, iterations + 1):
        residual = matrix - sparse
        left, values, right = np.linalg.svd(residual, full_matrices=False)
        truncated = np.array(values, copy=True)
        truncated[rank:] = 0.0
        updated = (left * truncated) @ right
        candidate = matrix - updated
        magnitude = np.abs(candidate).reshape(-1)
        if keep >= magnitude.size:
            threshold = 0.0
        else:
            split = magnitude.size - keep
            threshold = float(np.partition(magnitude, split)[split])
        updated_sparse = np.where(np.abs(candidate) >= threshold, candidate, 0.0)
        delta = np.linalg.norm(updated - low_rank) + np.linalg.norm(
            updated_sparse - sparse
        )
        relative = float(delta / (np.linalg.norm(matrix) + 1e-12))
        low_rank = updated
        sparse = updated_sparse
        if relative < tolerance:
            converged = True
            break

    spatial_low = low_rank.T.reshape(array.shape)
    spatial_sparse = sparse.T.reshape(array.shape)
    if (spatial_low + spatial_sparse).shape != array.shape:
        raise RuntimeError("low-rank decomposition changed the series shape")
    nonzero = np.count_nonzero(spatial_sparse)
    return LowRankSparseResult(
        low_rank=spatial_low,
        sparse=spatial_sparse,
        rank=int(rank),
        sparse_fraction=float(nonzero / spatial_sparse.size),
        iterations=int(used),
        converged=converged,
        relative_change=relative,
    )


def _validate_volume(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume, dtype=float)
    if array.ndim not in (3, 4):
        raise ValueError("volume must have shape (phases, H, W) or (phases, D, H, W)")
    if array.shape[0] < 2:
        raise ValueError("volume must contain at least two phases")
    if not np.all(np.isfinite(array)):
        raise ValueError("volume contains a non-finite value")
    if any(size < 4 for size in array.shape[1:]):
        raise ValueError("each spatial axis must have at least four voxels")
    return array


def _as_magnitude(real: np.ndarray, imag: np.ndarray | None) -> np.ndarray:
    if real is None:
        raise ValueError("real component is required")
    real_array = np.asarray(real, dtype=float)
    if real_array.ndim not in (2, 3):
        raise ValueError("images must be 2D or 3D")
    if imag is None:
        return real_array
    imag_array = np.asarray(imag, dtype=float)
    if imag_array.shape != real_array.shape:
        raise ValueError("real and imaginary components must have the same shape")
    return np.hypot(real_array, imag_array)


def _sample_coordinates(spatial_shape, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    flat = _dense_coordinates(spatial_shape).reshape(-1, len(spatial_shape))
    interior = _interior_indices(spatial_shape)
    count = min(int(samples), int(interior.size))
    choice = np.sort(rng.choice(interior, size=count, replace=False))
    return flat[choice], choice


def _interior_indices(spatial_shape) -> np.ndarray:
    mask = np.ones(spatial_shape, dtype=bool)
    for axis in range(len(spatial_shape)):
        low = [slice(None)] * len(spatial_shape)
        high = [slice(None)] * len(spatial_shape)
        low[axis] = slice(0, 1)
        high[axis] = slice(-1, None)
        mask[tuple(low)] = False
        mask[tuple(high)] = False
    return np.flatnonzero(mask.ravel())


def _dense_coordinates(spatial_shape) -> np.ndarray:
    axes = [np.linspace(-1.0, 1.0, int(size)) for size in spatial_shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _normalized_to_index(coords: np.ndarray, spatial_shape) -> np.ndarray:
    columns = [
        (coords[:, axis] + 1.0) * 0.5 * (size - 1)
        for axis, size in enumerate(spatial_shape)
    ]
    return np.stack(columns, axis=1)


def _warp_samples(image: np.ndarray, coords: np.ndarray, offset: np.ndarray) -> np.ndarray:
    index = _normalized_to_index(coords, image.shape) + offset
    sampled = map_coordinates(image, index.T, order=1, mode="nearest", prefilter=False)
    if sampled.shape != (coords.shape[0],):
        raise RuntimeError("resampled samples changed length")
    if not np.all(np.isfinite(sampled)):
        raise FloatingPointError("resampling produced a non-finite value")
    return sampled


def _sample_positions(coords: np.ndarray, spatial_shape, offset: np.ndarray) -> np.ndarray:
    return _normalized_to_index(coords, spatial_shape) + offset


def _sample_scalar(image: np.ndarray, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, slope = _sample_linear(image, position)
    return values[:, 0], slope[:, 0, :]


def _sample_linear(volume: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Values and slope of multilinear interpolation with nearest outside.

    ``volume`` is a scalar image ``(*spatial,)`` or ``(*spatial, channels)``.
    ``index`` is ``(N, D)`` in NumPy axis order.  Returns ``values`` of shape
    ``(N, C)`` and ``slope`` of shape ``(N, C, D)`` with
    ``slope = d(value)/d(index)``.  This is the derivative of
    ``map_coordinates(..., order=1, mode="nearest", prefilter=False)``, not a
    central difference of a precomputed volume.

    Inside a cell the slope along one axis is that cell's forward difference
    and the other axes stay linear.  Exactly on a voxel the two adjacent
    cells disagree; their average is what a central difference sees, and a
    zero-displacement start lands on integer voxels, so that average is what
    lets the first step move.  An axis outside the image, and any singleton
    axis, contributes slope zero because nearest lookup is flat there.
    """
    coords = np.asarray(index, dtype=float)
    array = np.asarray(volume, dtype=float)
    if coords.ndim != 2:
        raise RuntimeError(f"sample index must have shape (N, D), got {coords.shape}")
    dim = coords.shape[1]
    if array.ndim == dim:
        array = array[..., None]
    elif array.ndim != dim + 1:
        raise RuntimeError(
            f"volume ndim {array.ndim} does not match sample dimension {dim}"
        )
    spatial = array.shape[:-1]
    count = coords.shape[0]
    floors = np.floor(coords).astype(np.int64)
    fracs = coords - floors
    sizes = np.asarray(spatial, dtype=np.int64)
    channels = array.shape[-1]

    def gather(voxel_index: np.ndarray) -> np.ndarray:
        clipped = np.clip(voxel_index, 0, sizes - 1)
        return array[tuple(clipped.T)]

    values = np.zeros((count, channels), dtype=float)
    for mask in range(1 << dim):
        weight = np.ones(count, dtype=float)
        voxel = np.empty((count, dim), dtype=np.int64)
        for axis in range(dim):
            bit = (mask >> axis) & 1
            fraction = fracs[:, axis]
            weight *= fraction if bit else (1.0 - fraction)
            voxel[:, axis] = floors[:, axis] + bit
        values += weight[:, None] * gather(voxel)

    slope = np.zeros((count, channels, dim), dtype=float)
    for axis in range(dim):
        others = [item for item in range(dim) if item != axis]
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
        # Nearest lookup is constant outside the image, so that side of an
        # out-of-range sample does not contribute.  ``gather`` already clips.
        slope[:, :, axis] = diff
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(slope)):
        raise FloatingPointError("resampling produced a non-finite value")
    return values, slope


def _projection_loss_and_intensity_grad(samples, reference_samples, coords, spatial_shape):
    """Orthogonal localizer profiles and ``d(loss)/d(sample intensity)``.

    A localizer is the mean intensity along one axis, indexed by the other
    axes.  Samples that share those coordinates form one ray.  Comparing the
    ray means is sensitive to a translation: a shift perpendicular to the ray
    moves tissue into a different bin, and the orthogonal localizer sees a
    shift parallel to this ray.  The scalar sum of a projection would not.

    Identity displacement has loss zero because both profiles use the same
    bins.  A ray whose samples are brighter and darker by the same amount has
    loss zero too.  That is the difference from a photometric residual, which
    this objective does not use.  MIND remains the cross-phase structural term.
    """
    if samples.shape != reference_samples.shape or samples.ndim != 1:
        raise RuntimeError(
            f"localizer samples {samples.shape} != reference {reference_samples.shape}"
        )
    if coords.shape[0] != samples.shape[0] or coords.shape[1] != len(spatial_shape):
        raise RuntimeError(
            f"localizer coordinates {coords.shape} do not match samples {samples.shape}"
        )
    index = np.rint(_normalized_to_index(coords, spatial_shape)).astype(np.int64)
    for axis, size in enumerate(spatial_shape):
        index[:, axis] = np.clip(index[:, axis], 0, int(size) - 1)
    dim = index.shape[1]
    loss = 0.0
    gradient = np.zeros(samples.shape[0], dtype=float)
    for axis in range(dim):
        others = [item for item in range(dim) if item != axis]
        axis_loss, axis_gradient = _profile_loss(
            samples, reference_samples, index, others, spatial_shape
        )
        loss += axis_loss
        gradient += axis_gradient
    return loss / dim, gradient / dim


def _profile_loss(samples, reference, index, key_axes, spatial_shape):
    if key_axes:
        packed = np.zeros(samples.shape[0], dtype=np.int64)
        stride = 1
        for axis in reversed(key_axes):
            packed += index[:, axis] * stride
            stride *= int(spatial_shape[axis])
    else:
        packed = np.zeros(samples.shape[0], dtype=np.int64)
    order = np.argsort(packed, kind="mergesort")
    packed = packed[order]
    ordered_samples = samples[order]
    ordered_reference = reference[order]
    change = np.flatnonzero(packed[1:] != packed[:-1]) + 1
    starts = np.concatenate((np.zeros(1, dtype=int), change))
    counts = np.diff(np.concatenate((starts, np.array([packed.shape[0]]))))
    difference = (
        np.add.reduceat(ordered_samples, starts) - np.add.reduceat(ordered_reference, starts)
    ) / counts.astype(float)
    loss = float(np.mean(np.square(difference)))
    coeff = (2.0 / float(difference.size)) * difference / counts.astype(float)
    ordered_gradient = np.repeat(coeff, counts)
    gradient = np.empty(samples.shape[0], dtype=float)
    gradient[order] = ordered_gradient
    return loss, gradient


def _mind_loss_and_grad(moved: np.ndarray, reference: np.ndarray):
    delta = moved - reference
    loss = float(np.mean(np.square(delta)))
    return loss, 2.0 * delta / delta.size


def _temporal_smoothness(coefficients: np.ndarray, translations: np.ndarray) -> float:
    if coefficients.shape[0] < 3:
        return 0.0
    coeff = float(np.mean(np.square(np.diff(coefficients, axis=0))))
    shift = float(np.mean(np.square(np.diff(translations, axis=0))))
    return 0.5 * (coeff + shift)


def _temporal_gradient(
    coefficients: np.ndarray,
    translations: np.ndarray,
    reference_index: int,
):
    return (
        _one_temporal_gradient(coefficients, reference_index),
        _one_temporal_gradient(translations, reference_index),
    )


def _one_temporal_gradient(values: np.ndarray, reference_index: int) -> np.ndarray:
    gradient = np.zeros_like(values)
    if values.shape[0] < 3:
        return gradient
    delta = np.diff(values, axis=0)
    # ``_temporal_smoothness`` contributes ``0.5 * mean(delta**2)`` for this
    # array.  ``d(0.5 * mean(delta**2))/d(delta) = delta / delta.size``.
    # Dividing only by ``T - 1`` leaves an extra factor of the column count,
    # so rank 3 would be pulled three times too hard.
    gradient[:-1] -= delta
    gradient[1:] += delta
    gradient /= float(delta.size)
    gradient[reference_index] = 0.0
    return gradient


def _warp_volume(volume: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    warped = np.empty_like(volume)
    spatial_shape = volume.shape[1:]
    grid = _dense_coordinates(spatial_shape).reshape(-1, len(spatial_shape))
    for phase in range(volume.shape[0]):
        offset = displacement[phase].reshape(-1, len(spatial_shape))
        warped[phase] = _warp_samples(volume[phase], grid, offset).reshape(spatial_shape)
    return warped


def _resample_phase(image: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    if displacement.shape != image.shape + (image.ndim,):
        raise ValueError(
            f"displacement shape {displacement.shape} does not match image {image.shape}"
        )
    grid = _dense_coordinates(image.shape).reshape(-1, image.ndim)
    samples = _warp_samples(image, grid, displacement.reshape(-1, image.ndim))
    return samples.reshape(image.shape)


def _registration_metrics(original, warped, displacement, reference_index: int) -> dict:
    moving_index = _comparison_index(original.shape[0], reference_index)
    reference = original[reference_index]
    magnitude = np.linalg.norm(displacement, axis=-1)
    return {
        "ncc_before": _ncc(reference, original[moving_index]),
        "ncc_after": _ncc(reference, warped[moving_index]),
        "dice": _soft_dice(reference, warped[moving_index]),
        "finite": bool(
            np.all(np.isfinite(warped)) and np.all(np.isfinite(displacement))
        ),
        "displacement_rms": float(np.sqrt(np.mean(np.square(magnitude)))),
        "displacement_max": float(np.max(magnitude)),
    }


def _comparison_index(phases: int, reference_index: int) -> int:
    if reference_index == 0 and phases > 1:
        return 1
    return 0


def _ncc(left: np.ndarray, right: np.ndarray) -> float:
    centered_left = left.ravel() - float(np.mean(left))
    centered_right = right.ravel() - float(np.mean(right))
    denom = float(np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
    if denom == 0.0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(centered_left, centered_right) / denom)


def _soft_dice(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = left >= float(np.mean(left))
    right_mask = right >= float(np.mean(right))
    intersection = np.count_nonzero(left_mask & right_mask)
    total = int(np.count_nonzero(left_mask) + np.count_nonzero(right_mask))
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def _status_from_metrics(metrics: dict, config: INRRegistrationConfig):
    if not metrics["finite"]:
        return STATUS_FAILED, "registration produced a non-finite value"
    if metrics["displacement_max"] > config.displacement_warning:
        message = (
            "maximum displacement "
            f"{metrics['displacement_max']:.2f} px exceeds "
            f"{config.displacement_warning:.2f}"
        )
        return STATUS_WARNING, message
    return STATUS_OK, ""


def _failed_result(config, message: str, elapsed: float) -> RegistrationResult:
    return RegistrationResult(
        warped_real=None,
        warped_imag=None,
        transform=None,
        config=config,
        metric_name="projection+MIND",
        final_metric=float("nan"),
        dice=float("nan"),
        iterations=0,
        stop_condition="",
        elapsed_s=elapsed,
        status=STATUS_FAILED,
        message=message,
    )


def _check_shapes(
    volume, warped, displacement, coefficients, translations, rank: int
) -> None:
    spatial = volume.shape[1:]
    if warped.shape != volume.shape:
        raise RuntimeError(f"warped shape {warped.shape} != volume shape {volume.shape}")
    expected = (volume.shape[0],) + spatial + (len(spatial),)
    if displacement.shape != expected:
        raise RuntimeError(f"displacement shape {displacement.shape} != {expected}")
    if coefficients.shape != (volume.shape[0], rank):
        raise RuntimeError(
            f"coefficient shape {coefficients.shape} != {(volume.shape[0], rank)}"
        )
    if translations.shape != (volume.shape[0], len(volume.shape) - 1):
        raise RuntimeError(
            f"translation shape {translations.shape} does not match the volume"
        )
    if not np.all(np.isfinite(warped)) or not np.all(np.isfinite(displacement)):
        raise FloatingPointError("registration produced a non-finite value")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(translations)):
        raise FloatingPointError("INR coefficients became non-finite")


def _neighbour_offsets(ndim: int, radius: int) -> list[tuple[int, ...]]:
    ranges = [range(-radius, radius + 1) for _ in range(ndim)]
    stacked = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, ndim)
    return [tuple(int(value) for value in offset) for offset in stacked if np.any(offset)]


def _offset_grid(spatial_shape, offset: Sequence[int]) -> np.ndarray:
    axes = [np.arange(size, dtype=float) + shift for size, shift in zip(spatial_shape, offset)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([item.ravel() for item in mesh], axis=0)


__all__ = [
    "CoordinateMLP",
    "INRRegistrationConfig",
    "INRVolumeRegistrar",
    "LowRankSparseResult",
    "SpatialBasis",
    "low_rank_sparse_decomposition",
    "mind_descriptor",
]
