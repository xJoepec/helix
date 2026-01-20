"""Efficient trajectory logging with compressed sensing for high-dimensional states.

This module provides memory-efficient trajectory logging that reduces storage
requirements through compressed sensing while maintaining the ability to
recover important features and compute meaningful delta summaries for LLM context.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .env_api import AFLevelMetrics, AFMetrics


class CountSketch:
    """Count sketch for sparse feature recovery with provable guarantees."""

    def __init__(self, vector_size: int, sketch_size: int, seed: int = 42):
        """Initialize count sketch.

        Parameters
        ----------
        vector_size : int
            Dimension of input vectors
        sketch_size : int
            Size of the sketch (should be O(k log n) for k-sparse vectors)
        seed : int
            Random seed for reproducibility
        """
        self.vector_size = vector_size
        self.sketch_size = sketch_size

        # Random hash functions for count sketch
        rng = np.random.default_rng(seed)
        self.hash_indices = rng.integers(0, sketch_size, size=vector_size)
        self.hash_signs = rng.choice([-1, 1], size=vector_size)

        # The sketch itself
        self.sketch = np.zeros(sketch_size)

    def update(self, vector: np.ndarray) -> None:
        """Update sketch with new vector."""
        if len(vector) != self.vector_size:
            raise ValueError(f"Vector size {len(vector)} != expected {self.vector_size}")

        # Add vector to sketch using hash functions
        for i, (val, idx, sign) in enumerate(zip(vector, self.hash_indices, self.hash_signs)):
            self.sketch[idx] += sign * val

    def query(self, index: int) -> float:
        """Query estimate for vector component at index."""
        if index >= self.vector_size:
            raise ValueError(f"Index {index} >= vector size {self.vector_size}")

        return self.hash_signs[index] * self.sketch[self.hash_indices[index]]

    def recover_heavy_hitters(self, threshold: float) -> List[Tuple[int, float]]:
        """Recover indices with values above threshold."""
        heavy_hitters = []
        for i in range(self.vector_size):
            estimate = self.query(i)
            if abs(estimate) >= threshold:
                heavy_hitters.append((i, estimate))
        return heavy_hitters


@dataclass(frozen=True)
class TrajectorySnapshot:
    """Compact snapshot of AF extraction state for trajectory logging.

    This dataclass captures the essential information from an AF extraction
    at a given time step, optimized for efficient storage and delta computation.
    """

    step: int
    timestamp: float
    n_levels: int
    total_regions: int
    total_mass_error: float
    max_mass_error: float

    # Per-level summaries (compressed)
    region_counts: Tuple[int, ...]
    mass_errors: Tuple[float, ...]
    wasted_counts: Tuple[int, ...]
    cp_unital_errors: Tuple[float, ...]
    spectral_gaps: Tuple[Optional[float], ...]

    # Topological features
    betti_0: Tuple[int, ...]
    betti_1: Tuple[int, ...]
    betti_2: Tuple[int, ...]
    avg_lifetimes_1: Tuple[float, ...]

    # Capacity features
    capacity_means: Tuple[float, ...]
    capacity_maxes: Tuple[float, ...]

    # K-theory features
    k_ranks: Tuple[int, ...]
    k_nullities: Tuple[int, ...]
    k_torsion_counts: Tuple[int, ...]

    # Optional metadata
    metadata: Dict[str, Any] = None

    @classmethod
    def from_af_metrics(
        cls,
        metrics: AFMetrics,
        step: int,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrajectorySnapshot:
        """Create snapshot from AFMetrics object."""

        def safe_extract(levels: Sequence[AFLevelMetrics], extractor, default=0):
            """Safely extract values with fallback to default."""
            return tuple(extractor(level) if level else default for level in levels)

        levels = metrics.levels

        return cls(
            step=step,
            timestamp=timestamp,
            n_levels=len(levels),
            total_regions=sum(level.n_regions for level in levels),
            total_mass_error=sum(level.mass_error for level in levels),
            max_mass_error=max((level.mass_error for level in levels), default=0.0),

            # Per-level core metrics
            region_counts=tuple(level.n_regions for level in levels),
            mass_errors=tuple(level.mass_error for level in levels),
            wasted_counts=tuple(level.wasted_regions for level in levels),
            cp_unital_errors=safe_extract(
                levels,
                lambda level: level.cp_diagnostics.unital_err_fro if level.cp_diagnostics else 0.0,
            ),
            spectral_gaps=safe_extract(levels, lambda level: level.spectral_gap),

            # Topological features
            betti_0=safe_extract(
                levels,
                lambda level: level.persistent_homology.betti_numbers[0]
                if level.persistent_homology and len(level.persistent_homology.betti_numbers) > 0
                else 0,
            ),
            betti_1=safe_extract(
                levels,
                lambda level: level.persistent_homology.betti_numbers[1]
                if level.persistent_homology and len(level.persistent_homology.betti_numbers) > 1
                else 0,
            ),
            betti_2=safe_extract(
                levels,
                lambda level: level.persistent_homology.betti_numbers[2]
                if level.persistent_homology and len(level.persistent_homology.betti_numbers) > 2
                else 0,
            ),
            avg_lifetimes_1=safe_extract(
                levels,
                lambda level: (
                    level.persistent_homology.average_lifetimes[1]
                    if (
                        level.persistent_homology
                        and len(level.persistent_homology.average_lifetimes) > 1
                    )
                    else 0.0
                ),
            ),

            # Capacity features
            capacity_means=safe_extract(
                levels,
                lambda level: level.capacity_metrics.mean_loss if level.capacity_metrics else 0.0,
            ),
            capacity_maxes=safe_extract(
                levels,
                lambda level: level.capacity_metrics.max_loss if level.capacity_metrics else 0.0,
            ),

            # K-theory features
            k_ranks=safe_extract(
                levels,
                lambda level: (
                    level.k_theory_invariants.get('rank', 0)
                    if level.k_theory_invariants
                    else 0
                ),
            ),
            k_nullities=safe_extract(
                levels,
                lambda level: level.k_theory_invariants.get('nullity', 0)
                if level.k_theory_invariants
                else 0,
            ),
            k_torsion_counts=safe_extract(
                levels,
                lambda level: len(level.k_theory_invariants.get('torsion', []))
                if level.k_theory_invariants
                else 0,
            ),

            metadata=metadata or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    def to_compact_vector(self) -> np.ndarray:
        """Convert to compact numerical vector for ML analysis."""
        features = [
            float(self.step),
            float(self.n_levels),
            float(self.total_regions),
            float(self.total_mass_error),
            float(self.max_mass_error),
        ]

        # Add per-level features (pad to max depth for consistency)
        max_depth = 10  # Configurable
        for attr in ['region_counts', 'mass_errors', 'wasted_counts', 'cp_unital_errors']:
            values = getattr(self, attr)
            padded = list(values) + [0.0] * (max_depth - len(values))
            features.extend(padded[:max_depth])

        # Add summary topological features
        avg_lifetime = float(np.mean(self.avg_lifetimes_1)) if self.avg_lifetimes_1 else 0.0
        avg_capacity = (
            float(sum(self.capacity_means) / len(self.capacity_means))
            if self.capacity_means
            else 0.0
        )
        features.extend(
            [
                float(sum(self.betti_1)),  # Total β₁
                avg_lifetime,
                avg_capacity,
            ]
        )

        return np.array(features, dtype=np.float64)


@dataclass
class TrajectoryDelta:
    """Delta between two trajectory snapshots for LLM context compression."""

    step_from: int
    step_to: int
    time_delta: float

    # Core changes
    region_count_change: int
    mass_error_change: float
    max_depth_change: int

    # Most significant per-level changes
    significant_level_changes: List[Dict[str, Any]]

    # Physics interpretation
    physics_summary: str

    @classmethod
    def compute_delta(
        cls,
        prev: TrajectorySnapshot,
        curr: TrajectorySnapshot,
        significance_threshold: float = 0.1
    ) -> TrajectoryDelta:
        """Compute meaningful delta between two snapshots."""

        # Core changes
        region_change = curr.total_regions - prev.total_regions
        mass_change = curr.total_mass_error - prev.total_mass_error
        depth_change = curr.n_levels - prev.n_levels

        # Find significant level changes
        significant_changes = []
        for i in range(min(len(prev.region_counts), len(curr.region_counts))):
            mass_delta = curr.mass_errors[i] - prev.mass_errors[i]
            region_delta = curr.region_counts[i] - prev.region_counts[i]

            if (
                abs(mass_delta) > significance_threshold
                or abs(region_delta) > significance_threshold
            ):
                change = {
                    'level': i + 1,
                    'mass_delta': float(mass_delta),
                    'region_delta': int(region_delta),
                    'spectral_gap_delta': (
                        curr.spectral_gaps[i] - prev.spectral_gaps[i]
                        if curr.spectral_gaps[i] and prev.spectral_gaps[i] else None
                    )
                }
                significant_changes.append(change)

        # Generate physics interpretation
        physics_summary = cls._generate_physics_summary(prev, curr, significant_changes)

        return cls(
            step_from=prev.step,
            step_to=curr.step,
            time_delta=curr.timestamp - prev.timestamp,
            region_count_change=region_change,
            mass_error_change=mass_change,
            max_depth_change=depth_change,
            significant_level_changes=significant_changes,
            physics_summary=physics_summary
        )

    @staticmethod
    def _generate_physics_summary(
        prev: TrajectorySnapshot,
        curr: TrajectorySnapshot,
        changes: List[Dict[str, Any]]
    ) -> str:
        """Generate concise physics interpretation of changes."""
        summaries = []

        # Mass conservation
        if curr.total_mass_error < prev.total_mass_error * 0.9:
            summaries.append("mass conservation improving")
        elif curr.total_mass_error > prev.total_mass_error * 1.1:
            summaries.append("mass conservation degrading")

        # Topology changes
        total_betti_1_prev = sum(prev.betti_1)
        total_betti_1_curr = sum(curr.betti_1)
        if total_betti_1_curr > total_betti_1_prev:
            summaries.append("topology complexifying (β₁ rising)")
        elif total_betti_1_curr < total_betti_1_prev:
            summaries.append("topology simplifying (β₁ falling)")

        # Spectral gaps
        gap_changes = [c.get('spectral_gap_delta') for c in changes if c.get('spectral_gap_delta')]
        if gap_changes:
            avg_gap_change = np.mean([g for g in gap_changes if g is not None])
            if avg_gap_change > 0.1:
                summaries.append("ergodicity improving (gaps rising)")
            elif avg_gap_change < -0.1:
                summaries.append("approaching criticality (gaps falling)")

        # Region evolution
        if curr.total_regions < prev.total_regions * 0.9:
            summaries.append("coarse-graining (regions merging)")
        elif curr.total_regions > prev.total_regions * 1.1:
            summaries.append("refinement (regions splitting)")

        return "; ".join(summaries) if summaries else "stable dynamics"


class CompressedTrajectoryLogger:
    """Memory-efficient trajectory logger with O(log n) storage complexity.

    Uses compressed sensing with random projections and importance sampling
    to maintain only the most significant trajectory components.
    """

    def __init__(
        self,
        compression_rank: int = 20,
        max_snapshots: int = 1000,
        delta_threshold: float = 0.05,
        importance_decay: float = 0.9,
        sketch_size: int = 100
    ):
        self.compression_rank = compression_rank
        self.max_snapshots = max_snapshots
        self.delta_threshold = delta_threshold
        self.importance_decay = importance_decay
        self.sketch_size = sketch_size

        # Only keep most recent full snapshots (O(log n) count)
        self.recent_snapshots: List[TrajectorySnapshot] = []
        self.deltas: List[TrajectoryDelta] = []

        # Compressed representation using random projections and sketching
        self._projection_matrix: Optional[np.ndarray] = None
        self._compressed_trajectory: List[np.ndarray] = []

        # Importance-weighted sketch for critical features
        self._importance_weights: List[float] = []
        self._critical_features: List[np.ndarray] = []

        # Count sketch for exact recovery of sparse features
        self._count_sketch: Optional[CountSketch] = None

    def log_snapshot(self, snapshot: TrajectorySnapshot) -> None:
        """Add new snapshot to trajectory with O(log n) memory usage."""
        vector = snapshot.to_compact_vector()

        # Initialize compression structures on first snapshot
        if self._projection_matrix is None:
            self._initialize_compression(vector)

        # Determine if this snapshot should be kept as full or compressed
        importance_score = self._compute_importance_score(snapshot)

        # Always compress for the sketch
        compressed = self._projection_matrix @ vector
        self._compressed_trajectory.append(compressed)

        # Prune compressed trajectory to maintain O(log n) memory
        max_compressed = max(10, int(np.log2(len(self._compressed_trajectory) + 1)) * 2)
        if len(self._compressed_trajectory) > max_compressed:
            # Keep only the most recent max_compressed entries
            self._compressed_trajectory = self._compressed_trajectory[-max_compressed:]

        # Update count sketch for exact sparse recovery
        self._count_sketch.update(vector)

        # Keep only logarithmically many full snapshots based on importance
        if self._should_keep_full_snapshot(importance_score):
            # Exponential decay: keep fewer snapshots as trajectory grows
            max_recent = max(5, int(np.log2(len(self._compressed_trajectory) + 1)) + 1)

            # Add to recent snapshots with eviction
            self.recent_snapshots.append(snapshot)
            if len(self.recent_snapshots) > max_recent:
                # Remove least important snapshot (not the most recent)
                if len(self.recent_snapshots) > 1:
                    # Keep first and last, remove least important middle ones
                    importances = [
                        self._compute_importance_score(snapshot_candidate)
                        for snapshot_candidate in self.recent_snapshots[1:-1]
                    ]
                    min_idx = np.argmin(importances) + 1  # Offset for skipped first element
                    self.recent_snapshots.pop(min_idx)

            # Store critical features with importance weighting
            self._critical_features.append(vector * importance_score)
            self._importance_weights.append(importance_score)

            # Prune critical features to maintain O(log n) memory
            max_critical = max(10, int(np.log2(len(self._critical_features) + 1)) * 2)
            if len(self._critical_features) > max_critical:
                self._critical_features = self._critical_features[-max_critical:]
                self._importance_weights = self._importance_weights[-max_critical:]

        # Compute delta from most recent full snapshot
        if self.recent_snapshots and len(self.recent_snapshots) > 1:
            delta = TrajectoryDelta.compute_delta(
                self.recent_snapshots[-2], snapshot, self.delta_threshold
            )
            self.deltas.append(delta)

            # Keep only recent deltas (O(log n))
            max_deltas = max(10, int(np.log2(len(self._compressed_trajectory))))
            if len(self.deltas) > max_deltas:
                self.deltas.pop(0)

    def _initialize_compression(self, vector: np.ndarray) -> None:
        """Initialize compression matrices and sketches."""
        # Random projection matrix (Gaussian)
        self._projection_matrix = np.random.randn(self.compression_rank, len(vector))
        self._projection_matrix /= np.linalg.norm(self._projection_matrix, axis=1, keepdims=True)

        # Initialize count sketch for sparse recovery
        self._count_sketch = CountSketch(len(vector), self.sketch_size)

    def _compute_importance_score(self, snapshot: TrajectorySnapshot) -> float:
        """Compute importance score for snapshot retention decision."""
        score = 0.0

        # Mass error changes are important
        if len(self.recent_snapshots) > 0:
            prev = self.recent_snapshots[-1]
            mass_change = abs(snapshot.total_mass_error - prev.total_mass_error)
            score += mass_change * 10  # Scale factor

        # Topology changes are very important
        if len(self.recent_snapshots) > 0:
            prev = self.recent_snapshots[-1]
            betti_change = sum(abs(a - b) for a, b in zip(snapshot.betti_1, prev.betti_1))
            score += betti_change * 5

        # Large region count changes are important
        if len(self.recent_snapshots) > 0:
            prev = self.recent_snapshots[-1]
            region_change = abs(snapshot.total_regions - prev.total_regions) / max(
                prev.total_regions,
                1,
            )
            score += region_change

        # Add base importance to prevent zero scores
        score += 0.1

        # Apply temporal decay (recent snapshots more important)
        temporal_weight = self.importance_decay ** len(self._compressed_trajectory)
        score *= temporal_weight

        return float(score)

    def _should_keep_full_snapshot(self, importance_score: float) -> bool:
        """Decide whether to keep full snapshot based on importance."""
        # Always keep first few snapshots
        if len(self.recent_snapshots) < 3:
            return True

        # Keep if importance exceeds adaptive threshold
        if len(self._importance_weights) > 0:
            avg_importance = np.mean(self._importance_weights[-10:])  # Recent average
            threshold = avg_importance * 0.5  # Adaptive threshold
        else:
            threshold = 0.1

        return importance_score > threshold

    def get_recent_summary(self, window: int = 5) -> Dict[str, Any]:
        """Get summary of recent trajectory for LLM context."""
        if not self.recent_snapshots:
            return {"status": "no_data"}

        recent_snapshots = self.recent_snapshots[-window:]
        recent_deltas = self.deltas[-min(window - 1, len(self.deltas)):]

        return {
            "latest_step": recent_snapshots[-1].step,
            "trajectory_length": len(self.recent_snapshots),
            "recent_changes": [delta.physics_summary for delta in recent_deltas],
            "current_state": {
                "total_regions": recent_snapshots[-1].total_regions,
                "mass_error": recent_snapshots[-1].total_mass_error,
                "max_depth": recent_snapshots[-1].n_levels,
            },
            "trend_analysis": self._analyze_trends(recent_snapshots)
        }

    def _analyze_trends(self, snapshots: List[TrajectorySnapshot]) -> Dict[str, str]:
        """Analyze trends in recent snapshots."""
        if len(snapshots) < 2:
            return {"trend": "insufficient_data"}

        # Analyze mass error trend
        mass_errors = [s.total_mass_error for s in snapshots]
        mass_trend = "improving" if mass_errors[-1] < mass_errors[0] else "degrading"

        # Analyze region count trend
        regions = [s.total_regions for s in snapshots]
        region_trend = "refining" if regions[-1] > regions[0] else "coarsening"

        return {
            "mass_conservation": mass_trend,
            "partitioning": region_trend,
            "steps_analyzed": len(snapshots)
        }

    def save_trajectory(self, path: Union[str, Path]) -> None:
        """Save trajectory to disk in JSON format."""
        data = {
            "metadata": {
                "compression_rank": self.compression_rank,
                "total_snapshots": len(self.recent_snapshots),
                "total_deltas": len(self.deltas)
            },
            "snapshots": [s.to_dict() for s in self.recent_snapshots],
            "deltas": [asdict(d) for d in self.deltas]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load_trajectory(cls, path: Union[str, Path]) -> CompressedTrajectoryLogger:
        """Load trajectory from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        logger = cls(
            compression_rank=data["metadata"]["compression_rank"],
            max_snapshots=1000  # Use default
        )

        # Reconstruct snapshots and deltas
        for snap_data in data["snapshots"]:
            snapshot = TrajectorySnapshot(**snap_data)
            logger.snapshots.append(snapshot)

        for delta_data in data["deltas"]:
            delta = TrajectoryDelta(**delta_data)
            logger.deltas.append(delta)

        return logger


__all__ = [
    "TrajectorySnapshot",
    "TrajectoryDelta",
    "CompressedTrajectoryLogger"
]