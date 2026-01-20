"""Ulam flow environment for Perron-Frobenius operator spectral analysis.

This environment focuses on the analysis of residual block flows through
Ulam discretization and Perron-Frobenius operator spectral properties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency at runtime
    import verifiers as vf
    try:
        from verifiers import Messages, Parser  # type: ignore[attr-defined]
    except Exception:
        Parser = getattr(vf, "Parser", None)  # type: ignore[assignment]
        Messages = getattr(vf, "Messages", list)  # type: ignore[assignment]
    try:
        from verifiers.core import Env as _VFEnv  # type: ignore[import-not-found]
        from verifiers.core import Step as _VFStep  # type: ignore[import-not-found]
    except Exception:
        _VFEnv = object  # type: ignore[assignment]

        @dataclass
        class _VFStep:  # type: ignore[override]
            obs: Dict[str, Any]
            reward: float
            done: bool
            info: Dict[str, Any]
except Exception:  # pragma: no cover - graceful fallback
    vf = None
    Parser = None
    _VFEnv = object
    Messages = list  # type: ignore[assignment]

    @dataclass
    class _VFStep:  # type: ignore[override]
        obs: Dict[str, Any]
        reward: float
        done: bool
        info: Dict[str, Any]


@dataclass
class UlamObservation:
    """Structured observation emitted by the Ulam flow environment."""

    layer_index: int
    spectral_gap: float
    leading_eigenvalue: float
    second_eigenvalue: float
    mixing_time: float | None
    jacobian_determinant_mean: float
    jacobian_determinant_std: float
    expansion_factor: float
    contraction_factor: float
    ergodicity_measure: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "spectral_gap": self.spectral_gap,
            "leading_eigenvalue": self.leading_eigenvalue,
            "second_eigenvalue": self.second_eigenvalue,
            "mixing_time": self.mixing_time,
            "jacobian_determinant_mean": self.jacobian_determinant_mean,
            "jacobian_determinant_std": self.jacobian_determinant_std,
            "expansion_factor": self.expansion_factor,
            "contraction_factor": self.contraction_factor,
            "ergodicity_measure": self.ergodicity_measure,
        }


class UlamFlowEnv(_VFEnv):
    """Environment focused on Ulam discretization and PF operator analysis."""

    def __init__(
        self,
        model: Any,
        dataset: np.ndarray,
        *,
        ulam_bins: int = 32,
        ulam_samples_per_cell: int = 4,
        spectral_gap_weight: float = 1.0,
        mixing_weight: float = 0.5,
        jacobian_weight: float = 0.3,
        ergodicity_weight: float = 0.7,
    ) -> None:
        self._dataset = np.asarray(dataset, dtype=np.float32)
        if self._dataset.ndim != 2:
            raise ValueError("dataset must be 2D (N, d)")

        self._model = model
        self._ulam_bins = ulam_bins
        self._ulam_samples_per_cell = ulam_samples_per_cell
        self._spectral_gap_weight = float(spectral_gap_weight)
        self._mixing_weight = float(mixing_weight)
        self._jacobian_weight = float(jacobian_weight)
        self._ergodicity_weight = float(ergodicity_weight)

        self._cursor = 0
        self._history: list[Dict[str, Any]] = []

        # Extract flow information from residual-like blocks
        self._flow_observations = self._analyze_flows()

    def _analyze_flows(self) -> list[UlamObservation]:
        """Analyze flow dynamics in each layer using Ulam discretization."""
        import torch

        observations = []

        # Convert model to evaluation mode
        self._model.eval()

        # Collect residual-like layers (those that could represent flows)
        flow_layers = []
        for i, module in enumerate(self._model.modules()):
            if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
                # Treat each linear layer as a potential flow
                flow_layers.append((i, module))

        # Analyze each potential flow layer
        for layer_idx, layer in flow_layers:
            try:
                # Create input domain grid for Ulam discretization
                data_min = self._dataset.min(axis=0)
                data_max = self._dataset.max(axis=0)
                data_range = data_max - data_min
                data_min -= 0.1 * data_range  # Expand slightly
                data_max += 0.1 * data_range

                # Generate Ulam grid
                grid_points = self._generate_ulam_grid(data_min, data_max)

                # Compute forward flow through network up to this layer
                with torch.no_grad():
                    flow_output = self._compute_layer_flow(grid_points, layer_idx)

                # Build Ulam transition matrix
                P = self._build_ulam_operator(grid_points, flow_output, data_min, data_max)

                # Analyze spectral properties
                eigenvals, eigenvecs = np.linalg.eig(P)
                eigenvals = np.real(eigenvals)  # Take real part
                sorted_indices = np.argsort(eigenvals)[::-1]  # Sort descending
                eigenvals = eigenvals[sorted_indices]

                leading_eigenval = eigenvals[0] if len(eigenvals) > 0 else 0.0
                second_eigenval = eigenvals[1] if len(eigenvals) > 1 else 0.0
                spectral_gap = abs(leading_eigenval - second_eigenval)

                # Estimate mixing time
                mixing_time = self._estimate_mixing_time(spectral_gap)

                # Compute Jacobian statistics
                jacobian_stats = self._compute_jacobian_statistics(grid_points, layer_idx)

                # Compute expansion/contraction factors
                expansion_factor, contraction_factor = self._compute_expansion_contraction(eigenvals)

                # Compute ergodicity measure
                ergodicity_measure = self._compute_ergodicity_measure(P)

                obs = UlamObservation(
                    layer_index=layer_idx,
                    spectral_gap=spectral_gap,
                    leading_eigenvalue=leading_eigenval,
                    second_eigenvalue=second_eigenval,
                    mixing_time=mixing_time,
                    jacobian_determinant_mean=jacobian_stats["mean"],
                    jacobian_determinant_std=jacobian_stats["std"],
                    expansion_factor=expansion_factor,
                    contraction_factor=contraction_factor,
                    ergodicity_measure=ergodicity_measure,
                )

                observations.append(obs)

            except Exception:
                # Fallback for problematic layers
                obs = UlamObservation(
                    layer_index=layer_idx,
                    spectral_gap=0.0,
                    leading_eigenvalue=1.0,
                    second_eigenvalue=1.0,
                    mixing_time=None,
                    jacobian_determinant_mean=1.0,
                    jacobian_determinant_std=0.0,
                    expansion_factor=1.0,
                    contraction_factor=1.0,
                    ergodicity_measure=0.0,
                )
                observations.append(obs)

        return observations

    def _generate_ulam_grid(self, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
        """Generate a regular grid for Ulam discretization."""
        # For now, create a simple regular grid
        d = len(data_min)
        if d == 2:
            x = np.linspace(data_min[0], data_max[0], self._ulam_bins)
            y = np.linspace(data_min[1], data_max[1], self._ulam_bins)
            xx, yy = np.meshgrid(x, y)
            return np.column_stack([xx.ravel(), yy.ravel()])
        elif d == 3:
            x = np.linspace(data_min[0], data_max[0], int(self._ulam_bins**(1/3)))
            y = np.linspace(data_min[1], data_max[1], int(self._ulam_bins**(1/3)))
            z = np.linspace(data_min[2], data_max[2], int(self._ulam_bins**(1/3)))
            xx, yy, zz = np.meshgrid(x, y, z)
            return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        else:
            # For higher dimensions, use random sampling within bounds
            n_points = self._ulam_bins * self._ulam_samples_per_cell
            return np.random.uniform(data_min, data_max, (n_points, d))

    def _compute_layer_flow(self, grid_points: np.ndarray, layer_idx: int) -> np.ndarray:
        """Compute forward flow through network up to specified layer."""
        import torch

        # Simple implementation: just pass through the network
        # In a more sophisticated version, we'd track intermediate activations
        with torch.no_grad():
            x = torch.from_numpy(grid_points.astype(np.float32))
            y = self._model(x)
            return y.numpy()

    def _build_ulam_operator(
        self,
        grid_points: np.ndarray,
        flow_output: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray
    ) -> np.ndarray:
        """Build the Ulam discretization of the Perron-Frobenius operator."""
        n_cells = int(math.sqrt(len(grid_points))) if len(data_min) == 2 else int(len(grid_points)**(1/3))
        P = np.zeros((n_cells * n_cells, n_cells * n_cells))

        # This is a simplified implementation
        # In practice, you'd need more sophisticated cell assignment and mass transfer
        try:
            for i in range(len(grid_points)):
                # Map input and output to cell indices
                input_cell = self._point_to_cell(grid_points[i], data_min, data_max, n_cells)
                if flow_output.shape[1] >= len(data_min):
                    output_cell = self._point_to_cell(flow_output[i, :len(data_min)], data_min, data_max, n_cells)
                else:
                    # Handle dimension mismatch
                    continue

                if 0 <= input_cell < P.shape[0] and 0 <= output_cell < P.shape[1]:
                    P[output_cell, input_cell] += 1.0

            # Normalize columns to make it stochastic
            col_sums = P.sum(axis=0)
            col_sums[col_sums == 0] = 1.0  # Avoid division by zero
            P = P / col_sums[None, :]

        except Exception:
            # Fallback to identity matrix
            P = np.eye(n_cells * n_cells)

        return P

    def _point_to_cell(self, point: np.ndarray, data_min: np.ndarray, data_max: np.ndarray, n_cells: int) -> int:
        """Map a point to its cell index in the Ulam grid."""
        normalized = (point - data_min) / (data_max - data_min + 1e-10)
        normalized = np.clip(normalized, 0, 1 - 1e-10)

        if len(point) == 2:
            i = int(normalized[0] * n_cells)
            j = int(normalized[1] * n_cells)
            return i * n_cells + j
        else:
            # For higher dimensions, use a simpler mapping
            return int(np.sum(normalized) * n_cells * n_cells) % (n_cells * n_cells)

    def _estimate_mixing_time(self, spectral_gap: float) -> float | None:
        """Estimate mixing time from spectral gap."""
        if spectral_gap > 1e-10:
            return float(-1.0 / math.log(abs(1.0 - spectral_gap)))
        return None

    def _compute_jacobian_statistics(self, grid_points: np.ndarray, layer_idx: int) -> dict[str, float]:
        """Compute Jacobian determinant statistics."""
        # Simplified implementation - in practice you'd compute actual Jacobians
        n_samples = min(100, len(grid_points))
        _sample_indices = np.random.choice(len(grid_points), n_samples, replace=False)

        # Mock Jacobian determinants (replace with actual computation)
        jacobian_dets = np.random.lognormal(0, 0.3, n_samples)

        return {
            "mean": float(np.mean(jacobian_dets)),
            "std": float(np.std(jacobian_dets))
        }

    def _compute_expansion_contraction(self, eigenvals: np.ndarray) -> tuple[float, float]:
        """Compute expansion and contraction factors."""
        if len(eigenvals) == 0:
            return 1.0, 1.0

        # Expansion factor: largest eigenvalue magnitude
        expansion = float(np.max(np.abs(eigenvals)))

        # Contraction factor: smallest positive eigenvalue
        positive_eigenvals = eigenvals[eigenvals > 1e-10]
        contraction = float(np.min(positive_eigenvals)) if len(positive_eigenvals) > 0 else 1.0

        return expansion, contraction

    def _compute_ergodicity_measure(self, P: np.ndarray) -> float:
        """Compute ergodicity measure based on second-largest eigenvalue."""
        try:
            eigenvals = np.linalg.eigvals(P)
            eigenvals = np.real(eigenvals)
            eigenvals = np.sort(np.abs(eigenvals))[::-1]

            if len(eigenvals) > 1:
                return float(1.0 - eigenvals[1])  # 1 - second largest eigenvalue
            return 0.0
        except Exception:
            return 0.0

    def reset(self, *, seed: Optional[int] = None) -> _VFStep:  # type: ignore[override]
        del seed
        self._cursor = 0
        self._history.clear()

        if not self._flow_observations:
            # No flows to analyze
            return _VFStep(obs={}, reward=0.0, done=True, info={"message": "no flows found"})

        obs = self._flow_observations[self._cursor]
        reward = self._reward(obs)
        done = len(self._flow_observations) == 1
        self._cursor += 1
        return _VFStep(obs=obs.to_dict(), reward=reward, done=done, info={"layer": obs.layer_index})

    def step(self, action: Optional[Dict[str, Any]] = None) -> _VFStep:  # type: ignore[override]
        self._history.append(action or {})
        if self._cursor >= len(self._flow_observations):
            return _VFStep(obs={}, reward=0.0, done=True, info={"message": "complete"})

        obs = self._flow_observations[self._cursor]
        reward = self._reward(obs)
        done = self._cursor == len(self._flow_observations) - 1
        self._cursor += 1
        info = {"layer": obs.layer_index}
        if action is not None:
            info["action"] = action
        return _VFStep(obs=obs.to_dict(), reward=reward, done=done, info=info)

    def _reward(self, obs: UlamObservation) -> float:
        """Compute reward based on flow quality."""
        reward = 0.0

        # Reward larger spectral gaps (better mixing)
        reward += self._spectral_gap_weight * obs.spectral_gap

        # Reward finite mixing times
        if obs.mixing_time is not None and obs.mixing_time > 0:
            reward += self._mixing_weight / (1.0 + obs.mixing_time)

        # Penalize extreme Jacobian behavior
        jacobian_penalty = abs(obs.jacobian_determinant_mean - 1.0) + obs.jacobian_determinant_std
        reward -= self._jacobian_weight * jacobian_penalty

        # Reward ergodicity
        reward += self._ergodicity_weight * obs.ergodicity_measure

        return float(reward)

    @property
    def flow_observations(self) -> Sequence[UlamObservation]:
        return tuple(self._flow_observations)

    @property
    def history(self) -> Sequence[Dict[str, Any]]:
        return tuple(self._history)


def load_environment(
    *,
    system_prompt: Optional[str] = None,
    use_think: bool = False,
    seed: int = 1234,
    max_episodes: Optional[int] = None,
    **env_kwargs: Any,
):
    """Load Ulam flow environment (placeholder for verifiers integration)."""

    # This is a skeleton - full integration would require dataset generation
    # and prompt formatting similar to af_partition

    raise NotImplementedError(
        "Ulam flow environment integration with verifiers is not yet implemented. "
        "Use UlamFlowEnv directly for now."
    )


__all__ = ["UlamFlowEnv", "UlamObservation", "load_environment"]