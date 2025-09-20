from __future__ import annotations

"""CP dilation environment for completely positive map diagnostics.

This environment focuses on the CP map health and Stinespring dilations derived
from neural network partition structures.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
from helix import build_V_from_incidence, sanity_check_ucp

try:  # pragma: no cover - optional dependency
    from datasets import Dataset
except Exception:  # pragma: no cover
    Dataset = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency at runtime
    import verifiers as vf
    # Modern verifiers exposes Parser and Messages at the top-level
    try:
        from verifiers import Parser, Messages  # type: ignore[attr-defined]
    except Exception:
        Parser = getattr(vf, "Parser", None)  # type: ignore[assignment]
        Messages = getattr(vf, "Messages", list)  # type: ignore[assignment]
    try:  # type: ignore[unused-ignore]
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
except Exception:  # pragma: no cover - graceful fallback for tests / dev shells
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

from helix.env_api import AFLevelMetrics, AFMetrics, extract_af_metrics


@dataclass
class CPObservation:
    """Structured observation emitted by the CP dilation environment."""

    depth: int
    unital_error: float
    coisometry_error: float
    psd_violation: float
    condition_number: float
    choi_min_eigenvalue: float
    stinespring_rank: int | None = None
    channel_capacity: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "unital_error": self.unital_error,
            "coisometry_error": self.coisometry_error,
            "psd_violation": self.psd_violation,
            "condition_number": self.condition_number,
            "choi_min_eigenvalue": self.choi_min_eigenvalue,
            "stinespring_rank": self.stinespring_rank,
            "channel_capacity": self.channel_capacity,
        }


class CPDilationEnv(_VFEnv):
    """Environment focused on CP map diagnostics and Stinespring dilations."""

    def __init__(
        self,
        model: Any,
        dataset: np.ndarray,
        *,
        sample_weights: Optional[np.ndarray] = None,
        max_depth: Optional[int] = None,
        mass_tol: float = 1e-10,
        unital_weight: float = 1.0,
        coisometry_weight: float = 1.0,
        psd_weight: float = 2.0,
        condition_weight: float = 0.1,
    ) -> None:
        self._dataset = np.asarray(dataset, dtype=np.float32)
        if self._dataset.ndim != 2:
            raise ValueError("dataset must be 2D (N, d)")

        # Handle sample weights
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=np.float64)
            if weights.shape != (self._dataset.shape[0],):
                raise ValueError("sample_weights shape must match number of samples")
            if np.any(weights < 0):
                raise ValueError("sample_weights must be non-negative")
            total = float(weights.sum())
            if total <= 0:
                raise ValueError("sample_weights must sum to a positive value")
            if not np.isclose(total, 1.0):
                weights = weights / total
        else:
            weights = None

        # Extract AF metrics and levels
        metrics = extract_af_metrics(model, self._dataset, sample_weights=weights, mass_tol=mass_tol)
        levels = list(metrics.levels)
        if max_depth is not None:
            levels = levels[:max_depth]
        if not levels:
            raise ValueError("Model produced no ReLU partitions; at least one depth is required.")

        self._metrics = metrics
        self._levels = levels
        self._unital_weight = float(unital_weight)
        self._coisometry_weight = float(coisometry_weight)
        self._psd_weight = float(psd_weight)
        self._condition_weight = float(condition_weight)
        self._cursor = 0
        self._history: list[Dict[str, Any]] = []

        # Precompute CP diagnostics per depth
        extraction = metrics.extraction
        self._cp_observations: list[CPObservation] = []
        for idx, B in enumerate(extraction.B_list, start=1):
            tau_prev = np.array([1.0]) if idx == 1 else np.array(extraction.tau_list[idx - 2])
            tau_cur = np.array(extraction.tau_list[idx - 1])

            try:
                V = build_V_from_incidence(np.array(B), tau_prev, tau_cur)
                diag = sanity_check_ucp(V, trials=8)

                # Compute additional CP-specific metrics
                choi_matrix = self._compute_choi_matrix(V)
                choi_eigenvalues = np.linalg.eigvals(choi_matrix)
                choi_min_eig = float(np.min(np.real(choi_eigenvalues)))

                condition_num = float(np.linalg.cond(V))

                # Estimate Stinespring rank (environmental dimension)
                stinespring_rank = self._estimate_stinespring_rank(V)

                # Estimate channel capacity (optional advanced metric)
                channel_capacity = self._estimate_channel_capacity(choi_matrix)

                obs = CPObservation(
                    depth=idx,
                    unital_error=float(diag.get("unital_err_fro", 0.0)),
                    coisometry_error=float(diag.get("coisometry_err_fro", 0.0)),
                    psd_violation=float(diag.get("psd_min_eig_violation", 0.0)),
                    condition_number=condition_num,
                    choi_min_eigenvalue=choi_min_eig,
                    stinespring_rank=stinespring_rank,
                    channel_capacity=channel_capacity,
                )

            except Exception:
                # Fallback for degenerate cases
                obs = CPObservation(
                    depth=idx,
                    unital_error=float("nan"),
                    coisometry_error=float("nan"),
                    psd_violation=float("nan"),
                    condition_number=float("inf"),
                    choi_min_eigenvalue=float("-inf"),
                    stinespring_rank=None,
                    channel_capacity=None,
                )

            self._cp_observations.append(obs)

    def reset(self, *, seed: Optional[int] = None) -> _VFStep:  # type: ignore[override]
        del seed  # deterministic environment
        self._cursor = 0
        self._history.clear()
        obs = self._cp_observations[self._cursor]
        reward = self._reward(obs)
        done = len(self._cp_observations) == 1
        self._cursor += 1
        return _VFStep(obs=obs.to_dict(), reward=reward, done=done, info={"depth": obs.depth})

    def step(self, action: Optional[Dict[str, Any]] = None) -> _VFStep:  # type: ignore[override]
        self._history.append(action or {})
        if self._cursor >= len(self._cp_observations):
            return _VFStep(obs={}, reward=0.0, done=True, info={"message": "complete"})

        obs = self._cp_observations[self._cursor]
        reward = self._reward(obs)
        done = self._cursor == len(self._cp_observations) - 1
        self._cursor += 1
        info = {"depth": obs.depth}
        if action is not None:
            info["action"] = action
        return _VFStep(obs=obs.to_dict(), reward=reward, done=done, info=info)

    def _reward(self, obs: CPObservation) -> float:
        """Compute reward based on CP map health."""
        reward = 0.0

        # Penalize violations of CP properties
        if not np.isnan(obs.unital_error):
            reward -= self._unital_weight * obs.unital_error
        if not np.isnan(obs.coisometry_error):
            reward -= self._coisometry_weight * obs.coisometry_error
        if not np.isnan(obs.psd_violation):
            reward -= self._psd_weight * max(0, -obs.psd_violation)  # Penalize negative eigenvalues

        # Penalize poor conditioning
        if np.isfinite(obs.condition_number):
            reward -= self._condition_weight * np.log(max(1.0, obs.condition_number))

        # Bonus for positive semidefinite Choi matrix
        if obs.choi_min_eigenvalue > 0:
            reward += 0.5

        return float(reward)

    def _compute_choi_matrix(self, V: np.ndarray) -> np.ndarray:
        """Compute the Choi matrix representation of the CP map."""
        # For a CP map Φ(X) = V^* X V, the Choi matrix is
        # J_Φ = Σ_ij |i⟩⟨j| ⊗ Φ(|i⟩⟨j|)
        # In our case, this simplifies to V V^*
        return V @ V.T

    def _estimate_stinespring_rank(self, V: np.ndarray) -> int:
        """Estimate the environmental dimension in Stinespring dilation."""
        # The environmental dimension is the rank of V
        s = np.linalg.svd(V, compute_uv=False)
        tol = 1e-10 * s[0] if len(s) > 0 else 1e-10
        return int(np.sum(s > tol))

    def _estimate_channel_capacity(self, choi_matrix: np.ndarray) -> float | None:
        """Estimate the quantum channel capacity (simplified)."""
        try:
            # Simplified capacity estimate based on Choi matrix eigenvalues
            eigenvals = np.linalg.eigvals(choi_matrix)
            eigenvals = np.real(eigenvals[eigenvals > 1e-12])
            if len(eigenvals) == 0:
                return 0.0

            # Normalize
            eigenvals = eigenvals / np.sum(eigenvals)

            # Compute entropy-based capacity estimate
            entropy = -np.sum(eigenvals * np.log(eigenvals + 1e-15))
            return float(entropy)
        except Exception:
            return None

    @property
    def levels(self) -> Sequence[AFLevelMetrics]:
        return self._levels

    @property
    def metrics(self) -> AFMetrics:
        return self._metrics

    @property
    def cp_observations(self) -> Sequence[CPObservation]:
        return tuple(self._cp_observations)

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
    """Load CP dilation environment (placeholder for verifiers integration)."""

    # This is a skeleton - full integration would require dataset generation
    # and prompt formatting similar to af_partition

    raise NotImplementedError(
        "CP dilation environment integration with verifiers is not yet implemented. "
        "Use CPDilationEnv directly for now."
    )


__all__ = ["CPDilationEnv", "CPObservation", "load_environment"]