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


def test_detector_rejects_inconsistent_baseline_schema() -> None:
    detector = TemporalPhaseDetector()
    first = observation(0, 0.0, 0.0)
    second = FeatureObservation("r", "c1", 1, "p", first.features, "2.0")

    with pytest.raises(ValueError, match="schema"):
        detector.fit([first, second])


def test_detector_rejects_nonfinite_baseline() -> None:
    detector = TemporalPhaseDetector()

    with pytest.raises(ValueError, match="finite"):
        detector.fit([observation(0, 0.0, 0.0), observation(1, np.nan, 1.0)])


@pytest.mark.parametrize(
    "invalid_observation",
    [
        FeatureObservation("r", "missing", 5, "p", {"x": 1.0}),
        FeatureObservation("r", "schema", 5, "p", {"x": 1.0, "y": 1.0}, "2.0"),
        FeatureObservation("r", "nan", 5, "p", {"x": np.nan, "y": 1.0}),
        FeatureObservation("r", "inf", 5, "p", {"x": 1.0, "y": np.inf}),
    ],
    ids=["missing-feature", "schema-mismatch", "nan", "infinity"],
)
def test_detector_returns_insufficient_data_for_invalid_observation(
    invalid_observation: FeatureObservation,
) -> None:
    detector = TemporalPhaseDetector()
    detector.fit([observation(0, 0.0, 0.0), observation(1, 1.0, 1.0)])

    score = detector.score(invalid_observation)

    assert score.status == "insufficient_data"
    assert np.isnan(score.distance_squared)
    assert score.features == ("x", "y")
