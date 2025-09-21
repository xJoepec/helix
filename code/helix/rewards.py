"""Physics-informed reward computation for neural network analysis environments.

This module implements reward functions based on fundamental physics principles:
- Conservation laws (mass, energy, information)
- Topological stability (persistent homology)
- Wave coherence and gauge invariance
- Renormalization group flow principles
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .env_api import AFLevelMetrics, AFMetrics


@dataclass(frozen=True)
class PhysicsRewardComponents:
    """Individual components of physics-based reward."""

    # Conservation-based rewards
    mass_conservation: float
    energy_conservation: float
    information_preservation: float

    # Topological stability rewards
    homological_persistence: float
    topological_robustness: float
    structural_coherence: float

    # Wave mechanics rewards
    phase_coherence: float
    gauge_invariance: float
    unitarity_preservation: float

    # Dynamical systems rewards
    ergodic_mixing: float
    spectral_stability: float
    thermalization_rate: float

    # Combined scores
    conservation_score: float
    topology_score: float
    wave_score: float
    dynamics_score: float
    total_score: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "conservation": {
                "mass": self.mass_conservation,
                "energy": self.energy_conservation,
                "information": self.information_preservation,
                "total": self.conservation_score,
            },
            "topology": {
                "persistence": self.homological_persistence,
                "robustness": self.topological_robustness,
                "coherence": self.structural_coherence,
                "total": self.topology_score,
            },
            "wave_mechanics": {
                "phase_coherence": self.phase_coherence,
                "gauge_invariance": self.gauge_invariance,
                "unitarity": self.unitarity_preservation,
                "total": self.wave_score,
            },
            "dynamics": {
                "ergodic_mixing": self.ergodic_mixing,
                "spectral_stability": self.spectral_stability,
                "thermalization": self.thermalization_rate,
                "total": self.dynamics_score,
            },
            "total": self.total_score,
        }


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for physics-informed reward computation."""

    # Weight factors for different components
    conservation_weight: float = 1.0
    topology_weight: float = 0.7
    wave_weight: float = 0.8
    dynamics_weight: float = 0.6

    # Individual component weights
    mass_weight: float = 1.0
    cp_weight: float = 0.8
    ph_weight: float = 0.6
    gap_weight: float = 0.5

    # Tolerance parameters
    mass_tolerance: float = 1e-10
    cp_tolerance: float = 1e-8
    gap_threshold: float = 0.01

    # Physics regime preferences
    prefer_gapped_phase: bool = True
    prefer_topological_order: bool = True
    prefer_wave_coherence: bool = True

    @classmethod
    def conservative(cls) -> RewardConfig:
        """Conservative reward config emphasizing stability."""
        return cls(
            conservation_weight=1.5,
            topology_weight=1.0,
            wave_weight=0.5,
            dynamics_weight=0.3,
        )

    @classmethod
    def exploratory(cls) -> RewardConfig:
        """Exploratory config encouraging complex dynamics."""
        return cls(
            conservation_weight=0.8,
            topology_weight=1.2,
            wave_weight=1.0,
            dynamics_weight=1.0,
            prefer_gapped_phase=False,
        )

    @classmethod
    def wave_focused(cls) -> RewardConfig:
        """Config emphasizing wave mechanics and coherence."""
        return cls(
            conservation_weight=1.0,
            topology_weight=0.5,
            wave_weight=1.5,
            dynamics_weight=0.7,
        )


class PhysicsRewardComputer:
    """Computes physics-informed rewards from AF metrics."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

        # Track reward history for temporal analysis
        self._reward_history: List[PhysicsRewardComponents] = []

    def compute_reward(
        self,
        af_metrics: AFMetrics,
        level_index: int = -1,
        previous_metrics: Optional[AFMetrics] = None
    ) -> PhysicsRewardComponents:
        """Compute comprehensive physics-informed reward.

        Parameters
        ----------
        af_metrics : AFMetrics
            Current AF extraction metrics
        level_index : int
            Index of level to analyze (-1 for deepest)
        previous_metrics : Optional[AFMetrics]
            Previous metrics for temporal comparison

        Returns
        -------
        PhysicsRewardComponents
            Detailed breakdown of reward components
        """
        if not af_metrics.levels:
            return self._zero_reward()

        # Select level
        if level_index == -1:
            level = af_metrics.levels[-1]
        else:
            level = af_metrics.levels[min(level_index, len(af_metrics.levels) - 1)]

        # Compute individual components
        conservation_rewards = self._compute_conservation_rewards(level, af_metrics)
        topology_rewards = self._compute_topology_rewards(level, af_metrics)
        wave_rewards = self._compute_wave_rewards(level, af_metrics)
        dynamics_rewards = self._compute_dynamics_rewards(level, af_metrics)

        # Temporal comparison if available
        temporal_bonus = 0.0
        if previous_metrics and self._reward_history:
            temporal_bonus = self._compute_temporal_bonus(
                af_metrics, previous_metrics, level_index
            )

        # Combine into overall scores
        conservation_score = np.mean(list(conservation_rewards.values()))
        topology_score = np.mean(list(topology_rewards.values()))
        wave_score = np.mean(list(wave_rewards.values()))
        dynamics_score = np.mean(list(dynamics_rewards.values()))

        # Weighted total
        total_score = (
            self.config.conservation_weight * conservation_score +
            self.config.topology_weight * topology_score +
            self.config.wave_weight * wave_score +
            self.config.dynamics_weight * dynamics_score +
            temporal_bonus
        ) / (
            self.config.conservation_weight +
            self.config.topology_weight +
            self.config.wave_weight +
            self.config.dynamics_weight
        )

        reward = PhysicsRewardComponents(
            # Conservation components
            mass_conservation=conservation_rewards["mass"],
            energy_conservation=conservation_rewards["energy"],
            information_preservation=conservation_rewards["information"],

            # Topology components
            homological_persistence=topology_rewards["persistence"],
            topological_robustness=topology_rewards["robustness"],
            structural_coherence=topology_rewards["coherence"],

            # Wave mechanics components
            phase_coherence=wave_rewards["phase"],
            gauge_invariance=wave_rewards["gauge"],
            unitarity_preservation=wave_rewards["unitarity"],

            # Dynamics components
            ergodic_mixing=dynamics_rewards["ergodic"],
            spectral_stability=dynamics_rewards["spectral"],
            thermalization_rate=dynamics_rewards["thermalization"],

            # Combined scores
            conservation_score=conservation_score,
            topology_score=topology_score,
            wave_score=wave_score,
            dynamics_score=dynamics_score,
            total_score=total_score,
        )

        self._reward_history.append(reward)
        return reward

    def _compute_conservation_rewards(
        self,
        level: AFLevelMetrics,
        af_metrics: AFMetrics
    ) -> Dict[str, float]:
        """Compute rewards based on conservation laws."""

        # Mass conservation (Noether's theorem for translation symmetry)
        mass_error = level.mass_error
        mass_reward = np.exp(-mass_error / self.config.mass_tolerance)

        # Energy conservation (from Hamiltonian structure)
        # Approximate via spectral properties and CP map unitality
        energy_error = 0.0
        if level.cp_diagnostics:
            energy_error = level.cp_diagnostics.unital_err_fro
        energy_reward = np.exp(-energy_error / self.config.cp_tolerance)

        # Information preservation (quantum unitarity)
        info_error = 0.0
        if level.cp_diagnostics:
            # Combine unitality and coisometry errors
            info_error = (
                level.cp_diagnostics.unital_err_fro +
                level.cp_diagnostics.coisometry_err_fro
            ) / 2.0
        info_reward = np.exp(-info_error / self.config.cp_tolerance)

        return {
            "mass": float(mass_reward),
            "energy": float(energy_reward),
            "information": float(info_reward),
        }

    def _compute_topology_rewards(
        self,
        level: AFLevelMetrics,
        af_metrics: AFMetrics
    ) -> Dict[str, float]:
        """Compute rewards based on topological stability."""

        # Homological persistence (lifetime of topological features)
        persistence_reward = 0.0
        if level.persistent_homology and level.persistent_homology.computed:
            ph = level.persistent_homology
            if len(ph.average_lifetimes) > 1:
                # Reward long-lived β₁ features
                avg_lifetime = ph.average_lifetimes[1]
                persistence_reward = min(avg_lifetime / level.depth, 1.0)

        # Topological robustness (consistency across scales)
        robustness_reward = 0.0
        if len(af_metrics.levels) > 1:
            # Compare Betti numbers across scales
            betti_sequence = []
            for l in af_metrics.levels:
                if l.persistent_homology and l.persistent_homology.computed:
                    betti = l.persistent_homology.betti_numbers
                    betti_1 = betti[1] if len(betti) > 1 else 0
                    betti_sequence.append(betti_1)

            if len(betti_sequence) > 1:
                # Reward decreasing variance (stability)
                betti_variance = np.var(betti_sequence)
                robustness_reward = np.exp(-betti_variance)

        # Structural coherence (K-theory invariants)
        coherence_reward = 0.0
        if level.k_theory_invariants:
            # Reward simple torsion structure
            torsion = level.k_theory_invariants.get('torsion', [])
            if torsion:
                # Prefer small torsion orders
                avg_torsion = np.mean(torsion)
                coherence_reward = np.exp(-avg_torsion / 10.0)
            else:
                # Free abelian groups are coherent
                coherence_reward = 1.0

        return {
            "persistence": float(persistence_reward),
            "robustness": float(robustness_reward),
            "coherence": float(coherence_reward),
        }

    def _compute_wave_rewards(
        self,
        level: AFLevelMetrics,
        af_metrics: AFMetrics
    ) -> Dict[str, float]:
        """Compute rewards based on wave mechanics principles."""

        # Phase coherence (unitarity of evolution)
        phase_coherence = 0.0
        if level.cp_diagnostics:
            # Perfect coisometry indicates unitary evolution
            coiso_error = level.cp_diagnostics.coisometry_err_fro
            phase_coherence = np.exp(-coiso_error / self.config.cp_tolerance)

        # Gauge invariance (unitality of CP maps)
        gauge_invariance = 0.0
        if level.cp_diagnostics:
            unital_error = level.cp_diagnostics.unital_err_fro
            gauge_invariance = np.exp(-unital_error / self.config.cp_tolerance)

        # Unitarity preservation (positive semidefiniteness)
        unitarity = 0.0
        if level.cp_diagnostics:
            psd_violation = abs(level.cp_diagnostics.psd_min_eig_violation)
            unitarity = np.exp(-psd_violation / self.config.cp_tolerance)

        return {
            "phase": float(phase_coherence),
            "gauge": float(gauge_invariance),
            "unitarity": float(unitarity),
        }

    def _compute_dynamics_rewards(
        self,
        level: AFLevelMetrics,
        af_metrics: AFMetrics
    ) -> Dict[str, float]:
        """Compute rewards based on dynamical systems theory."""

        # Ergodic mixing (spectral gap of transfer operator)
        ergodic_reward = 0.0
        if level.spectral_gap is not None:
            gap = level.spectral_gap
            if self.config.prefer_gapped_phase:
                # Reward large gaps (fast mixing)
                ergodic_reward = min(gap / 0.5, 1.0)
            else:
                # Reward critical dynamics (small gaps)
                ergodic_reward = np.exp(-gap / self.config.gap_threshold)

        # Spectral stability (condition numbers, etc.)
        spectral_stability = 0.0
        if level.k_theory_invariants:
            # Use condition number from K-theory computation
            condition = level.k_theory_invariants.get('condition_number', 1.0)
            if condition < float('inf'):
                spectral_stability = 1.0 / (1.0 + np.log10(max(condition, 1.0)))

        # Thermalization rate (approach to equilibrium)
        thermalization = 0.0
        if level.spectral_gap is not None and level.spectral_gap > 0:
            # Larger gaps mean faster thermalization
            thermalization = min(level.spectral_gap, 1.0)

        return {
            "ergodic": float(ergodic_reward),
            "spectral": float(spectral_stability),
            "thermalization": float(thermalization),
        }

    def _compute_temporal_bonus(
        self,
        current: AFMetrics,
        previous: AFMetrics,
        level_index: int
    ) -> float:
        """Compute bonus based on temporal evolution."""
        if len(self._reward_history) < 2:
            return 0.0

        current_reward = self._reward_history[-1]
        previous_reward = self._reward_history[-2]

        # Reward consistent improvement
        improvement_bonus = 0.0
        components = [
            (current_reward.conservation_score, previous_reward.conservation_score),
            (current_reward.topology_score, previous_reward.topology_score),
            (current_reward.wave_score, previous_reward.wave_score),
            (current_reward.dynamics_score, previous_reward.dynamics_score),
        ]

        improvements = [curr > prev for curr, prev in components]
        improvement_bonus = sum(improvements) / len(improvements) * 0.1

        # Reward stability (small changes)
        stability_bonus = 0.0
        total_change = abs(current_reward.total_score - previous_reward.total_score)
        if total_change < 0.05:  # Stable evolution
            stability_bonus = 0.05

        return improvement_bonus + stability_bonus

    def _zero_reward(self) -> PhysicsRewardComponents:
        """Return zero reward for error cases."""
        return PhysicsRewardComponents(
            mass_conservation=0.0,
            energy_conservation=0.0,
            information_preservation=0.0,
            homological_persistence=0.0,
            topological_robustness=0.0,
            structural_coherence=0.0,
            phase_coherence=0.0,
            gauge_invariance=0.0,
            unitarity_preservation=0.0,
            ergodic_mixing=0.0,
            spectral_stability=0.0,
            thermalization_rate=0.0,
            conservation_score=0.0,
            topology_score=0.0,
            wave_score=0.0,
            dynamics_score=0.0,
            total_score=0.0,
        )

    def get_reward_history(self) -> List[PhysicsRewardComponents]:
        """Get complete reward history."""
        return self._reward_history.copy()

    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward evolution."""
        if not self._reward_history:
            return {"error": "No reward history available"}

        total_scores = [r.total_score for r in self._reward_history]
        conservation_scores = [r.conservation_score for r in self._reward_history]
        topology_scores = [r.topology_score for r in self._reward_history]

        return {
            "total_reward": {
                "mean": float(np.mean(total_scores)),
                "std": float(np.std(total_scores)),
                "min": float(np.min(total_scores)),
                "max": float(np.max(total_scores)),
                "trend": "improving" if total_scores[-1] > total_scores[0] else "declining",
            },
            "conservation": {
                "mean": float(np.mean(conservation_scores)),
                "stability": float(1.0 - np.std(conservation_scores)),
            },
            "topology": {
                "mean": float(np.mean(topology_scores)),
                "complexity": float(np.mean([r.homological_persistence for r in self._reward_history])),
            },
            "episodes": len(self._reward_history),
        }


# Factory functions for common reward computers
def create_stable_reward_computer() -> PhysicsRewardComputer:
    """Create reward computer optimized for stable network analysis."""
    return PhysicsRewardComputer(RewardConfig.conservative())


def create_exploratory_reward_computer() -> PhysicsRewardComputer:
    """Create reward computer encouraging topological exploration."""
    return PhysicsRewardComputer(RewardConfig.exploratory())


def create_wave_reward_computer() -> PhysicsRewardComputer:
    """Create reward computer focused on wave mechanics."""
    return PhysicsRewardComputer(RewardConfig.wave_focused())


__all__ = [
    "PhysicsRewardComponents",
    "RewardConfig",
    "PhysicsRewardComputer",
    "create_stable_reward_computer",
    "create_exploratory_reward_computer",
    "create_wave_reward_computer",
]