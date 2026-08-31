import numpy as np
import pytest
from helix.temporal import FeatureObservation, TemporalPhaseDetector


def observation(step: int, x: float, y: float) -> FeatureObservation:
    return FeatureObservation("r", f"c{step}", step, "p", {"x": x, "y": y})


def test_detector_handles_singular_baseline_and_scores_drift() -> None:
    detector = TemporalPhaseDetector(shrinkage=0.2, ridge=1e-6, threshold=4.0)
    detector.fit([observation(i, float(i), float(i)) for i in range(5)])
    normal = detector.score(observation(5, 2.0, 2.0))
    drift = detector.score(observation(6, 20.0, -20.0))
    assert np.isfinite(normal.distance_squared)
    assert drift.distance_squared > normal.distance_squared
    assert drift.status == "candidate_transition"


def test_detector_requires_two_baseline_observations() -> None:
    detector = TemporalPhaseDetector()
    with pytest.raises(ValueError, match="at least two"):
        detector.fit([observation(0, 0.0, 0.0)])
