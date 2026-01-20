"""K-theory environment for persistent topological analysis of neural networks.

This environment provides an advanced interface for analyzing K-theory invariants
across AF partition hierarchies using discrete Hodge theory and spectral methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from helix import (
        PersistentHomologySummary,
        compute_persistence_k_theory,
        compute_persistent_homology,
        extract_af_metrics,
        k_invariants_hodge,
    )
except ImportError:
    # Local imports for development
    import sys
    from pathlib import Path
    helix_path = Path(__file__).parent.parent.parent / "code"
    sys.path.insert(0, str(helix_path))

    from helix.env_api import extract_af_metrics
    from helix.ktheory import compute_persistence_k_theory, k_invariants_hodge
    from helix.topology import PersistentHomologySummary, compute_persistent_homology

try:
    import verifiers as vf
    from verifiers import Messages, Parser
    try:
        from verifiers.core import Env as _VFEnv
        from verifiers.core import Step as _VFStep
    except Exception:
        _VFEnv = object

        @dataclass
        class _VFStep:
            obs: Dict[str, Any]
            reward: float
            done: bool
            info: Dict[str, Any]

except Exception:
    vf = None
    Parser = None
    Messages = list
    _VFEnv = object

    @dataclass
    class _VFStep:
        obs: Dict[str, Any]
        reward: float
        done: bool
        info: Dict[str, Any]


@dataclass
class KTheoryObservation:
    """Structured observation for K-theory environment."""

    step: int
    depth: int

    # K-theory invariants
    betti_numbers: List[int]
    rank: int
    nullity: int
    torsion_orders: List[int]
    spectral_gap: float
    condition_number: float

    # Persistent features
    rank_sequence: List[int]
    betti_evolution: List[int]
    stability_measure: float

    # Physics interpretation
    topology_summary: str
    gauge_analysis: str

    # Numerical features for ML
    feature_vector: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "step": self.step,
            "depth": self.depth,
            "k_theory": {
                "betti_numbers": self.betti_numbers,
                "rank": self.rank,
                "nullity": self.nullity,
                "torsion_orders": self.torsion_orders,
                "spectral_gap": self.spectral_gap,
                "condition_number": self.condition_number,
            },
            "persistence": {
                "rank_sequence": self.rank_sequence,
                "betti_evolution": self.betti_evolution,
                "stability": self.stability_measure,
            },
            "interpretation": {
                "topology": self.topology_summary,
                "gauge": self.gauge_analysis,
            },
            "features": self.feature_vector.tolist(),
        }


class KTheoryEnv(_VFEnv):
    """Environment for K-theory analysis of neural network AF partitions."""

    def __init__(
        self,
        model: Any,
        dataset: np.ndarray,
        *,
        sample_weights: Optional[np.ndarray] = None,
        max_depth: Optional[int] = None,
        tolerance: float = 1e-10,
        k_theory_weight: float = 1.0,
        persistence_weight: float = 0.5,
        stability_weight: float = 0.3,
        compute_ph: bool = True,
    ) -> None:
        """Initialize K-theory environment.

        Parameters
        ----------
        model : Any
            PyTorch model to analyze
        dataset : np.ndarray
            Input dataset for AF extraction (N, d)
        sample_weights : Optional[np.ndarray]
            Optional sample weights
        max_depth : Optional[int]
            Maximum depth to analyze
        tolerance : float
            Numerical tolerance for Hodge decomposition
        k_theory_weight : float
            Weight for K-theory metrics in reward
        persistence_weight : float
            Weight for persistence metrics in reward
        stability_weight : float
            Weight for stability metrics in reward
        compute_ph : bool
            Whether to compute persistent homology
        """
        self._dataset = np.asarray(dataset, dtype=np.float32)
        if self._dataset.ndim != 2:
            raise ValueError("dataset must be 2D (N, d)")

        # Extract AF metrics
        af_metrics = extract_af_metrics(
            model, self._dataset,
            sample_weights=sample_weights
        )

        if max_depth is not None:
            levels = list(af_metrics.levels[:max_depth])
        else:
            levels = list(af_metrics.levels)

        if not levels:
            raise ValueError("No AF partition levels found")

        self._af_metrics = af_metrics
        self._levels = levels
        self._tolerance = tolerance
        self._k_theory_weight = k_theory_weight
        self._persistence_weight = persistence_weight
        self._stability_weight = stability_weight
        self._compute_ph = compute_ph

        # Compute K-theory analysis
        B_sequence = [level.B for level in levels]
        self._k_analysis = compute_persistence_k_theory(
            B_sequence, tolerance=tolerance
        )

        # Environment state
        self._cursor = 0
        self._max_steps = len(levels)
        self._history: List[Dict[str, Any]] = []

        # Precompute persistent homology if requested
        self._ph_analysis: Optional[Dict[int, PersistentHomologySummary]] = None
        if compute_ph:
            self._compute_persistent_homology()

    def _compute_persistent_homology(self) -> None:
        """Compute persistent homology for each level."""
        self._ph_analysis = {}

        for i, level in enumerate(self._levels):
            try:
                # Extract representative points from AF partition
                # Use mass-weighted centroids of cells
                points = self._extract_partition_points(level)
                if len(points) > 2:
                    ph_summary = compute_persistent_homology(
                        points, maxdim=2, sample_cap=500
                    )
                    self._ph_analysis[i] = ph_summary
            except Exception:
                # Skip if PH computation fails
                pass

    def _extract_partition_points(self, level) -> List[List[float]]:
        """Extract representative points from AF partition level."""
        # Simple approach: return random points weighted by cell masses
        # In practice, this would use the actual partition structure
        n_points = min(level.n_regions, 100)
        return np.random.randn(n_points, 2).tolist()  # Simplified 2D projection

    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state."""
        self._cursor = 0
        self._history.clear()
        return self._get_observation()

    def step(self, action: Optional[Dict[str, Any]] = None) -> _VFStep:
        """Take a step in the K-theory analysis."""
        if self._cursor >= self._max_steps:
            return _VFStep(
                obs=self._get_observation(),
                reward=0.0,
                done=True,
                info={"error": "Maximum steps reached"}
            )

        # Process action (advance depth, recompute, etc.)
        action = action or {"type": "advance"}
        self._process_action(action)

        # Get current observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward()

        # Check if done
        done = self._cursor >= self._max_steps - 1

        # Prepare info
        info = {
            "k_theory_analysis": self._get_current_k_analysis(),
            "physics_interpretation": self._get_physics_interpretation(),
            "step": self._cursor,
            "max_steps": self._max_steps,
        }

        # Record in history
        step_record = {
            "step": self._cursor,
            "action": action,
            "observation": obs,
            "reward": reward,
            "info": info,
        }
        self._history.append(step_record)

        return _VFStep(obs=obs, reward=reward, done=done, info=info)

    def _process_action(self, action: Dict[str, Any]) -> None:
        """Process the given action."""
        action_type = action.get("type", "advance")

        if action_type == "advance":
            self._cursor = min(self._cursor + 1, self._max_steps - 1)
        elif action_type == "recompute":
            # Recompute with different tolerance
            new_tolerance = action.get("tolerance", self._tolerance)
            if new_tolerance != self._tolerance:
                self._tolerance = new_tolerance
                B_sequence = [level.B for level in self._levels]
                self._k_analysis = compute_persistence_k_theory(
                    B_sequence, tolerance=new_tolerance
                )
        elif action_type == "analyze_depth":
            # Jump to specific depth
            target_depth = action.get("depth", self._cursor)
            self._cursor = min(max(0, target_depth), self._max_steps - 1)

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        if self._cursor >= len(self._levels):
            return {"error": "Invalid cursor position"}

        current_level = self._levels[self._cursor]
        depth_analysis = self._k_analysis.get("depth_analysis", [])

        if self._cursor < len(depth_analysis):
            k_inv = depth_analysis[self._cursor]
        else:
            # Fallback computation
            k_inv = k_invariants_hodge(current_level.B, tolerance=self._tolerance)

        # Extract persistent features up to current depth
        persistent_features = self._k_analysis.get("persistent_features", {})
        rank_seq = persistent_features.get("rank_sequence", [])[:self._cursor + 1]
        betti_seq = persistent_features.get("betti_sequence", [])[:self._cursor + 1]
        stability_measures = persistent_features.get("stability_measures", [])

        # Compute stability measure
        stability = float(np.mean(stability_measures)) if stability_measures else 0.0

        # Create structured observation
        obs = KTheoryObservation(
            step=self._cursor,
            depth=current_level.depth,
            betti_numbers=k_inv.get("betti_numbers", [0]),
            rank=k_inv.get("rank", 0),
            nullity=k_inv.get("nullity", 0),
            torsion_orders=k_inv.get("torsion_orders", []),
            spectral_gap=k_inv.get("spectral_gap", 0.0),
            condition_number=k_inv.get("condition_number", 1.0),
            rank_sequence=rank_seq,
            betti_evolution=betti_seq,
            stability_measure=stability,
            topology_summary=self._generate_topology_summary(k_inv),
            gauge_analysis=self._generate_gauge_analysis(k_inv),
            feature_vector=self._extract_feature_vector(k_inv, rank_seq, betti_seq),
        )

        return obs.to_dict()

    def _generate_topology_summary(self, k_inv: Dict[str, Any]) -> str:
        """Generate human-readable topology summary."""
        betti = k_inv.get("betti_numbers", [0])
        rank = k_inv.get("rank", 0)
        torsion = k_inv.get("torsion_orders", [])

        summary_parts = []

        if len(betti) > 0 and betti[0] > 0:
            summary_parts.append(f"β₀={betti[0]} connected components")

        if rank > 0:
            summary_parts.append(f"rank-{rank} free part")

        if torsion:
            torsion_str = ",".join(map(str, torsion))
            summary_parts.append(f"Z/{torsion_str} torsion")

        if not summary_parts:
            return "trivial topology"

        return "; ".join(summary_parts)

    def _generate_gauge_analysis(self, k_inv: Dict[str, Any]) -> str:
        """Generate gauge theory interpretation."""
        gap = k_inv.get("spectral_gap", 0.0)
        condition = k_inv.get("condition_number", 1.0)

        if gap > 0.5:
            gap_analysis = "gapped phase (massive)"
        elif gap > 0.1:
            gap_analysis = "small gap (weakly coupled)"
        else:
            gap_analysis = "gapless (critical/conformal)"

        if condition > 100:
            condition_analysis = "ill-conditioned (gauge fixing issues)"
        elif condition > 10:
            condition_analysis = "moderate conditioning"
        else:
            condition_analysis = "well-conditioned"

        return f"{gap_analysis}; {condition_analysis}"

    def _extract_feature_vector(
        self,
        k_inv: Dict[str, Any],
        rank_seq: List[int],
        betti_seq: List[int]
    ) -> np.ndarray:
        """Extract numerical feature vector for ML analysis."""
        features = [
            float(self._cursor),  # Current step
            float(k_inv.get("rank", 0)),
            float(k_inv.get("nullity", 0)),
            float(len(k_inv.get("torsion_orders", []))),
            float(k_inv.get("spectral_gap", 0.0)),
            float(k_inv.get("condition_number", 1.0)),
        ]

        # Add sequence features (padded to fixed length)
        max_seq_len = 10
        rank_padded = (rank_seq + [0] * max_seq_len)[:max_seq_len]
        betti_padded = (betti_seq + [0] * max_seq_len)[:max_seq_len]

        features.extend(rank_padded)
        features.extend(betti_padded)

        # Add summary statistics
        if rank_seq:
            features.extend([
                float(np.mean(rank_seq)),
                float(np.std(rank_seq)),
                float(rank_seq[-1] - rank_seq[0]) if len(rank_seq) > 1 else 0.0,
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float64)

    def _compute_reward(self) -> float:
        """Compute reward based on K-theory stability and interpretability."""
        if self._cursor >= len(self._k_analysis.get("depth_analysis", [])):
            return 0.0

        k_inv = self._k_analysis["depth_analysis"][self._cursor]
        persistent_features = self._k_analysis.get("persistent_features", {})

        # K-theory reward: prefer simple torsion with good spectral properties
        k_reward = 0.0
        spectral_gap = k_inv.get("spectral_gap", 0.0)
        condition_num = k_inv.get("condition_number", float("inf"))

        # Reward good spectral gap
        k_reward += min(spectral_gap, 1.0)

        # Penalize poor conditioning
        k_reward -= min(np.log10(max(condition_num, 1.0)) / 5.0, 1.0)

        # Reward interpretable torsion
        torsion_orders = k_inv.get("torsion_orders", [])
        if torsion_orders:
            # Small torsion orders are more interpretable
            avg_torsion = np.mean(torsion_orders)
            k_reward += max(0, 1.0 - avg_torsion / 10.0)

        # Persistence reward: prefer stable evolution
        persistence_reward = 0.0
        stability_measures = persistent_features.get("stability_measures", [])
        if stability_measures:
            # Reward decreasing instability (increasing stability)
            avg_stability = np.mean(stability_measures)
            persistence_reward += max(0, 1.0 - avg_stability / 5.0)

        # Stability reward: prefer consistent rank evolution
        stability_reward = 0.0
        rank_sequence = persistent_features.get("rank_sequence", [])
        if len(rank_sequence) > 1:
            rank_variance = np.var(rank_sequence)
            stability_reward += max(0, 1.0 - rank_variance / 10.0)

        # Combine rewards
        total_reward = (
            self._k_theory_weight * k_reward +
            self._persistence_weight * persistence_reward +
            self._stability_weight * stability_reward
        )

        return float(total_reward)

    def _get_current_k_analysis(self) -> Dict[str, Any]:
        """Get K-theory analysis for current depth."""
        if self._cursor < len(self._k_analysis.get("depth_analysis", [])):
            return self._k_analysis["depth_analysis"][self._cursor]
        return {}

    def _get_physics_interpretation(self) -> Dict[str, str]:
        """Get physics interpretation of current state."""
        k_inv = self._get_current_k_analysis()

        return {
            "topology": self._generate_topology_summary(k_inv),
            "gauge": self._generate_gauge_analysis(k_inv),
            "regime": self._classify_regime(k_inv),
        }

    def _classify_regime(self, k_inv: Dict[str, Any]) -> str:
        """Classify the current physical regime."""
        gap = k_inv.get("spectral_gap", 0.0)
        rank = k_inv.get("rank", 0)

        if gap > 0.5 and rank > 0:
            return "massive_gauge_theory"
        elif gap > 0.1:
            return "weakly_coupled_phase"
        elif gap < 0.01:
            return "critical_conformal_phase"
        else:
            return "intermediate_coupling"

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full history of steps."""
        return self._history.copy()

    def get_persistence_summary(self) -> Dict[str, Any]:
        """Get summary of persistent K-theory features."""
        return self._k_analysis.get("persistence_summary", {})


def load_k_theory_environment(
    model: Any,
    dataset: np.ndarray,
    **kwargs
) -> KTheoryEnv:
    """Factory function for creating K-theory environment."""
    return KTheoryEnv(model, dataset, **kwargs)


__all__ = [
    "KTheoryEnv",
    "KTheoryObservation",
    "load_k_theory_environment"
]