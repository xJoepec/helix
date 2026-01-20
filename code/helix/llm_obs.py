"""LLM-first observation schema with dual surfaces for neural network analysis.

This module provides observations optimized for Large Language Model consumption,
featuring both structured JSON data (for tool callers) and natural language
summaries (for instruction-tuned models).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .env_api import AFLevelMetrics, AFMetrics


@dataclass(frozen=True)
class StructuredMetrics:
    """Structured numerical metrics for tool calling and ML analysis."""

    # Core AF metrics
    n_regions: int
    mass_error_l1: float
    mass_error_linf: float
    wasted_regions: int
    combinatorial_entropy: float

    # CP map diagnostics
    cp_unital_error: float
    cp_coisometry_error: float
    cp_psd_violation: float

    # Topological features
    betti_0: int
    betti_1: int
    betti_2: int
    avg_lifetime_1: float
    max_lifetime_1: float

    # Capacity and efficiency
    capacity_mean_loss: float
    capacity_max_loss: float

    # Spectral properties
    spectral_gap: float

    # K-theory invariants
    k_rank: int
    k_nullity: int
    k_torsion_count: int

    # Physics parameters
    wave_coherence_score: float
    gauge_invariance_score: float
    conservation_score: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_feature_vector(self) -> np.ndarray:
        """Convert to compact numerical vector for ML."""
        return np.array([
            float(self.n_regions),
            float(self.mass_error_l1),
            float(self.wasted_regions),
            float(self.cp_unital_error),
            float(self.betti_1),
            float(self.avg_lifetime_1),
            float(self.capacity_mean_loss),
            float(self.spectral_gap),
            float(self.k_rank),
            float(self.wave_coherence_score),
        ], dtype=np.float64)


@dataclass(frozen=True)
class NaturalLanguageContext:
    """Human-readable natural language context for LLM understanding."""

    # Core summary (< 100 chars)
    headline: str

    # Detailed analysis (< 400 chars total)
    topology_summary: str
    dynamics_summary: str
    physics_interpretation: str

    # Actionable insights (< 100 chars)
    recommendations: str

    # Delta from previous state (< 150 chars)
    change_summary: str

    def to_compact_prompt(self) -> str:
        """Generate compact prompt for LLM context (<500 chars)."""
        parts = [
            f"Current: {self.headline}",
            f"Topology: {self.topology_summary}",
            f"Dynamics: {self.dynamics_summary}",
            f"Physics: {self.physics_interpretation}",
        ]

        if self.change_summary.strip():
            parts.append(f"Change: {self.change_summary}")

        if self.recommendations.strip():
            parts.append(f"Suggest: {self.recommendations}")

        return " | ".join(parts)


@dataclass(frozen=True)
class DualSurfaceObservation:
    """Complete dual-surface observation for LLM-first environments."""

    # Metadata
    step: int
    depth: int
    timestamp: float

    # Dual surfaces
    structured: StructuredMetrics
    natural_language: NaturalLanguageContext

    # Optional trajectory context
    trajectory_window: Optional[List[Dict[str, Any]]] = None

    # Schema version for compatibility
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to complete JSON-serializable dictionary."""
        return {
            "metadata": {
                "step": self.step,
                "depth": self.depth,
                "timestamp": self.timestamp,
                "schema_version": self.schema_version,
            },
            "structured": self.structured.to_dict(),
            "natural_language": {
                "headline": self.natural_language.headline,
                "topology": self.natural_language.topology_summary,
                "dynamics": self.natural_language.dynamics_summary,
                "physics": self.natural_language.physics_interpretation,
                "recommendations": self.natural_language.recommendations,
                "changes": self.natural_language.change_summary,
                "compact_prompt": self.natural_language.to_compact_prompt(),
            },
            "trajectory": self.trajectory_window,
        }

    def get_llm_context(self) -> str:
        """Get optimized context string for LLM consumption."""
        return self.natural_language.to_compact_prompt()

    def get_tool_data(self) -> Dict[str, Union[int, float]]:
        """Get structured data for tool calling."""
        return self.structured.to_dict()


class ObservationGenerator:
    """Factory for generating dual-surface observations from AF metrics."""

    def __init__(
        self,
        *,
        enable_physics_interpretation: bool = True,
        enable_trajectory_context: bool = True,
        max_trajectory_window: int = 5,
        interpretation_style: str = "concise"  # "concise", "detailed", "technical"
    ):
        self.enable_physics = enable_physics_interpretation
        self.enable_trajectory = enable_trajectory_context
        self.max_window = max_trajectory_window
        self.style = interpretation_style

        # Track previous states for delta computation
        self._previous_metrics: Optional[StructuredMetrics] = None
        self._trajectory_history: List[Dict[str, Any]] = []

    def generate_observation(
        self,
        af_metrics: AFMetrics,
        step: int,
        timestamp: float,
        level_index: int = -1,  # -1 for deepest level
        metadata: Optional[Dict[str, Any]] = None
    ) -> DualSurfaceObservation:
        """Generate dual-surface observation from AF metrics."""

        if not af_metrics.levels:
            raise ValueError("No AF levels available")

        # Select level to analyze
        if level_index == -1:
            level = af_metrics.levels[-1]  # Deepest level
        else:
            level = af_metrics.levels[min(level_index, len(af_metrics.levels) - 1)]

        # Extract structured metrics
        structured = self._extract_structured_metrics(level, af_metrics)

        # Generate natural language context
        natural_language = self._generate_natural_language(
            structured, level, metadata
        )

        # Prepare trajectory window
        trajectory_window = None
        if self.enable_trajectory and self._trajectory_history:
            trajectory_window = self._trajectory_history[-self.max_window:]

        # Create observation
        observation = DualSurfaceObservation(
            step=step,
            depth=level.depth,
            timestamp=timestamp,
            structured=structured,
            natural_language=natural_language,
            trajectory_window=trajectory_window,
        )

        # Update internal state
        self._previous_metrics = structured
        if self.enable_trajectory:
            self._trajectory_history.append({
                "step": step,
                "timestamp": timestamp,
                "structured": structured.to_dict(),
                "headline": natural_language.headline,
            })

        return observation

    def _extract_structured_metrics(
        self,
        level: AFLevelMetrics,
        af_metrics: AFMetrics
    ) -> StructuredMetrics:
        """Extract structured numerical metrics."""

        # Core AF metrics
        n_regions = level.n_regions
        mass_error_l1 = level.mass_error
        mass_error_linf = level.trace_residual_linf
        wasted_regions = level.wasted_regions
        combinatorial_entropy = level.combinatorial_entropy

        # CP map diagnostics
        cp_unital_error = 0.0
        cp_coisometry_error = 0.0
        cp_psd_violation = 0.0
        if level.cp_diagnostics:
            cp_unital_error = level.cp_diagnostics.unital_err_fro
            cp_coisometry_error = level.cp_diagnostics.coisometry_err_fro
            cp_psd_violation = level.cp_diagnostics.psd_min_eig_violation

        # Topological features
        betti_0 = 0
        betti_1 = 0
        betti_2 = 0
        avg_lifetime_1 = 0.0
        max_lifetime_1 = 0.0
        if level.persistent_homology and level.persistent_homology.computed:
            ph = level.persistent_homology
            betti_numbers = ph.betti_numbers
            if len(betti_numbers) > 0:
                betti_0 = betti_numbers[0]
            if len(betti_numbers) > 1:
                betti_1 = betti_numbers[1]
            if len(betti_numbers) > 2:
                betti_2 = betti_numbers[2]

            avg_lifetimes = ph.average_lifetimes
            max_lifetimes = ph.max_lifetimes
            if len(avg_lifetimes) > 1:
                avg_lifetime_1 = avg_lifetimes[1]
            if len(max_lifetimes) > 1:
                max_lifetime_1 = max_lifetimes[1]

        # Capacity metrics
        capacity_mean_loss = 0.0
        capacity_max_loss = 0.0
        if level.capacity_metrics and level.capacity_metrics.computed:
            capacity_mean_loss = level.capacity_metrics.mean_loss
            capacity_max_loss = level.capacity_metrics.max_loss

        # Spectral properties
        spectral_gap = level.spectral_gap or 0.0

        # K-theory invariants
        k_rank = 0
        k_nullity = 0
        k_torsion_count = 0
        if level.k_theory_invariants:
            k_rank = level.k_theory_invariants.get('rank', 0)
            k_nullity = level.k_theory_invariants.get('nullity', 0)
            k_torsion_count = len(level.k_theory_invariants.get('torsion', []))

        # Physics-derived scores
        wave_coherence_score = self._compute_wave_coherence(level)
        gauge_invariance_score = self._compute_gauge_invariance(level)
        conservation_score = self._compute_conservation_score(level)

        return StructuredMetrics(
            n_regions=n_regions,
            mass_error_l1=mass_error_l1,
            mass_error_linf=mass_error_linf,
            wasted_regions=wasted_regions,
            combinatorial_entropy=combinatorial_entropy,
            cp_unital_error=cp_unital_error,
            cp_coisometry_error=cp_coisometry_error,
            cp_psd_violation=cp_psd_violation,
            betti_0=betti_0,
            betti_1=betti_1,
            betti_2=betti_2,
            avg_lifetime_1=avg_lifetime_1,
            max_lifetime_1=max_lifetime_1,
            capacity_mean_loss=capacity_mean_loss,
            capacity_max_loss=capacity_max_loss,
            spectral_gap=spectral_gap,
            k_rank=k_rank,
            k_nullity=k_nullity,
            k_torsion_count=k_torsion_count,
            wave_coherence_score=wave_coherence_score,
            gauge_invariance_score=gauge_invariance_score,
            conservation_score=conservation_score,
        )

    def _generate_natural_language(
        self,
        structured: StructuredMetrics,
        level: AFLevelMetrics,
        metadata: Optional[Dict[str, Any]]
    ) -> NaturalLanguageContext:
        """Generate natural language interpretation."""

        # Generate headline
        headline = self._generate_headline(structured)

        # Topology summary
        topology_summary = self._generate_topology_summary(structured)

        # Dynamics summary
        dynamics_summary = self._generate_dynamics_summary(structured)

        # Physics interpretation
        physics_interpretation = ""
        if self.enable_physics:
            physics_interpretation = self._generate_physics_interpretation(structured)

        # Recommendations
        recommendations = self._generate_recommendations(structured)

        # Change summary
        change_summary = ""
        if self._previous_metrics:
            change_summary = self._generate_change_summary(
                self._previous_metrics, structured
            )

        return NaturalLanguageContext(
            headline=headline,
            topology_summary=topology_summary,
            dynamics_summary=dynamics_summary,
            physics_interpretation=physics_interpretation,
            recommendations=recommendations,
            change_summary=change_summary,
        )

    def _generate_headline(self, metrics: StructuredMetrics) -> str:
        """Generate concise headline summary."""
        # Classify regime
        if metrics.mass_error_l1 < 0.01 and metrics.spectral_gap > 0.5:
            regime = "stable"
        elif metrics.mass_error_l1 > 0.1:
            regime = "unstable"
        elif metrics.spectral_gap < 0.1:
            regime = "critical"
        else:
            regime = "transitional"

        # Add topology info
        if metrics.betti_1 > 0:
            topology = f"β₁={metrics.betti_1}"
        else:
            topology = "trivial"

        return f"D{metrics.k_rank} {regime} ({metrics.n_regions} regions, {topology})"

    def _generate_topology_summary(self, metrics: StructuredMetrics) -> str:
        """Generate topology summary."""
        parts = []

        if metrics.betti_1 > 0:
            parts.append(f"β₁={metrics.betti_1} loops")
            if metrics.avg_lifetime_1 > 0.5:
                parts.append("persistent")
            else:
                parts.append("transient")

        if metrics.k_torsion_count > 0:
            parts.append(f"Z/{metrics.k_torsion_count} torsion")

        if not parts:
            return "topologically trivial"

        return "; ".join(parts)

    def _generate_dynamics_summary(self, metrics: StructuredMetrics) -> str:
        """Generate dynamics summary."""
        gap_desc = "gapped" if metrics.spectral_gap > 0.3 else "gapless"
        mass_desc = "conserved" if metrics.mass_error_l1 < 0.01 else "violated"

        capacity_desc = ""
        if metrics.capacity_mean_loss > 0.2:
            capacity_desc = ", lossy"
        elif metrics.capacity_mean_loss > 0.05:
            capacity_desc = ", compressed"

        return f"{gap_desc}, mass {mass_desc}{capacity_desc}"

    def _generate_physics_interpretation(self, metrics: StructuredMetrics) -> str:
        """Generate physics interpretation."""
        interpretations = []

        # Wave mechanics
        if metrics.wave_coherence_score > 0.7:
            interpretations.append("coherent wave")
        elif metrics.wave_coherence_score > 0.3:
            interpretations.append("partially coherent")
        else:
            interpretations.append("decoherent")

        # Gauge theory
        if metrics.gauge_invariance_score > 0.8:
            interpretations.append("gauge invariant")
        elif metrics.cp_unital_error > 0.1:
            interpretations.append("gauge anomaly")

        # Phase classification
        if metrics.spectral_gap > 0.5:
            interpretations.append("massive phase")
        elif metrics.spectral_gap < 0.1:
            interpretations.append("critical phase")

        return "; ".join(interpretations)

    def _generate_recommendations(self, metrics: StructuredMetrics) -> str:
        """Generate actionable recommendations."""
        if metrics.mass_error_l1 > 0.1:
            return "improve mass conservation"
        elif metrics.wasted_regions > metrics.n_regions * 0.3:
            return "reduce wasted regions"
        elif metrics.capacity_mean_loss > 0.3:
            return "prevent capacity collapse"
        elif metrics.spectral_gap < 0.05:
            return "increase spectral gap"
        else:
            return "maintain current regime"

    def _generate_change_summary(
        self,
        prev: StructuredMetrics,
        curr: StructuredMetrics
    ) -> str:
        """Generate summary of changes."""
        changes = []

        # Mass conservation
        mass_delta = curr.mass_error_l1 - prev.mass_error_l1
        if abs(mass_delta) > 0.01:
            trend = "improving" if mass_delta < 0 else "degrading"
            changes.append(f"mass {trend}")

        # Topology
        betti_delta = curr.betti_1 - prev.betti_1
        if betti_delta != 0:
            trend = "complexifying" if betti_delta > 0 else "simplifying"
            changes.append(f"topology {trend}")

        # Spectral gap
        gap_delta = curr.spectral_gap - prev.spectral_gap
        if abs(gap_delta) > 0.05:
            trend = "opening" if gap_delta > 0 else "closing"
            changes.append(f"gap {trend}")

        if not changes:
            return "stable evolution"

        return "; ".join(changes)

    def _compute_wave_coherence(self, level: AFLevelMetrics) -> float:
        """Compute wave coherence score from CP diagnostics."""
        if not level.cp_diagnostics:
            return 0.0

        # Perfect unitality and coisometry indicate wave coherence
        unital_score = max(0, 1.0 - level.cp_diagnostics.unital_err_fro)
        coiso_score = max(0, 1.0 - level.cp_diagnostics.coisometry_err_fro)

        return float((unital_score + coiso_score) / 2.0)

    def _compute_gauge_invariance(self, level: AFLevelMetrics) -> float:
        """Compute gauge invariance score."""
        if not level.cp_diagnostics:
            return 0.0

        # Gauge invariance requires unitality and positive semidefiniteness
        unital_score = max(0, 1.0 - level.cp_diagnostics.unital_err_fro)
        psd_score = max(0, 1.0 - abs(level.cp_diagnostics.psd_min_eig_violation))

        return float((unital_score + psd_score) / 2.0)

    def _compute_conservation_score(self, level: AFLevelMetrics) -> float:
        """Compute conservation score from mass errors."""
        # Perfect conservation gives score 1.0
        l1_score = max(0, 1.0 - level.mass_error * 10)  # Scale for visibility
        linf_score = max(0, 1.0 - level.trace_residual_linf * 100)

        return float((l1_score + linf_score) / 2.0)


# Pre-configured observation generators for different use cases
def create_concise_generator() -> ObservationGenerator:
    """Create generator optimized for concise LLM interactions."""
    return ObservationGenerator(
        enable_physics_interpretation=True,
        enable_trajectory_context=True,
        max_trajectory_window=3,
        interpretation_style="concise"
    )


def create_detailed_generator() -> ObservationGenerator:
    """Create generator for detailed technical analysis."""
    return ObservationGenerator(
        enable_physics_interpretation=True,
        enable_trajectory_context=True,
        max_trajectory_window=10,
        interpretation_style="detailed"
    )


def create_minimal_generator() -> ObservationGenerator:
    """Create minimal generator for fast responses."""
    return ObservationGenerator(
        enable_physics_interpretation=False,
        enable_trajectory_context=False,
        interpretation_style="concise"
    )


__all__ = [
    "StructuredMetrics",
    "NaturalLanguageContext",
    "DualSurfaceObservation",
    "ObservationGenerator",
    "create_concise_generator",
    "create_detailed_generator",
    "create_minimal_generator",
]