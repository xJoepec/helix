from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from helix import temporal
from helix.temporal import FeatureObservation, TemporalPhaseDetector

_DEFAULT_PROVENANCE = object()


def provenance(
    *,
    source: str = "studio",
    run_id: str = "r",
    checkpoint_id: str = "c0",
    checkpoint_fingerprint: str = "sha256:0",
    probe_set_id: str = "p",
    schema_version: str = "1.0",
) -> "temporal.ObservationProvenance":
    return temporal.ObservationProvenance(
        source,
        run_id,
        checkpoint_id,
        checkpoint_fingerprint,
        probe_set_id,
        schema_version,
    )


def observation(
    step: int,
    x: float,
    y: float,
    *,
    source: str = "studio",
    run_id: str = "r",
    probe_set_id: str = "p",
    schema_version: str = "1.0",
    checkpoint_fingerprint: str | None = None,
    observation_provenance: object = _DEFAULT_PROVENANCE,
) -> FeatureObservation:
    checkpoint_id = f"c{step}"
    return FeatureObservation(
        run_id,
        checkpoint_id,
        step,
        probe_set_id,
        {"x": x, "y": y},
        provenance(
            source=source,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            checkpoint_fingerprint=(
                f"sha256:{step}" if checkpoint_fingerprint is None else checkpoint_fingerprint
            ),
            probe_set_id=probe_set_id,
            schema_version=schema_version,
        )
        if observation_provenance is _DEFAULT_PROVENANCE
        else observation_provenance,
        schema_version,
    )


def assert_insufficient_data(score: object) -> None:
    assert score.status == "insufficient_data"
    assert np.isnan(score.distance_squared)
    assert score.features == ("x", "y")


def test_observation_provenance_interface_is_available() -> None:
    assert hasattr(temporal, "ObservationProvenance")


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
    second = observation(1, 1.0, 1.0, schema_version="2.0")

    with pytest.raises(ValueError, match="schema"):
        detector.fit([first, second])


def test_detector_rejects_nonfinite_baseline() -> None:
    detector = TemporalPhaseDetector()

    with pytest.raises(ValueError, match="finite"):
        detector.fit([observation(0, 0.0, 0.0), observation(1, np.nan, 1.0)])


@pytest.mark.parametrize(
    "invalid_observation_factory",
    [
        lambda: replace(observation(5, 1.0, 1.0), features={"x": 1.0}),
        lambda: observation(5, 1.0, 1.0, schema_version="2.0"),
        lambda: observation(5, np.nan, 1.0),
        lambda: observation(5, 1.0, np.inf),
    ],
    ids=["missing-feature", "schema-mismatch", "nan", "infinity"],
)
def test_detector_returns_insufficient_data_for_invalid_observation(
    invalid_observation_factory: Callable[[], FeatureObservation],
) -> None:
    detector = TemporalPhaseDetector()
    detector.fit([observation(0, 0.0, 0.0), observation(1, 1.0, 1.0)])

    score = detector.score(invalid_observation_factory())

    assert_insufficient_data(score)


def test_observation_provenance_is_immutable() -> None:
    item = provenance()

    with pytest.raises(FrozenInstanceError):
        item.run_id = "other"


def test_observation_rejects_missing_provenance() -> None:
    with pytest.raises(ValueError, match="provenance"):
        observation(0, 0.0, 0.0, observation_provenance=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "other-run"),
        ("checkpoint_id", "other-checkpoint"),
        ("probe_set_id", "other-probes"),
        ("schema_version", "2.0"),
    ],
)
def test_observation_rejects_duplicated_identity_inconsistent_with_provenance(
    field: str, value: str
) -> None:
    item = observation(0, 0.0, 0.0)

    with pytest.raises(ValueError, match="provenance"):
        replace(item, **{field: value})


@pytest.mark.parametrize(
    "baseline_factory",
    [
        lambda: [observation(0, 0.0, 0.0), observation(1, 1.0, 1.0, run_id="other-run")],
        lambda: [
            observation(0, 0.0, 0.0),
            observation(1, 1.0, 1.0, probe_set_id="other-probes"),
        ],
        lambda: [
            observation(0, 0.0, 0.0),
            observation(1, 1.0, 1.0, source="other-source"),
        ],
        lambda: [observation(0, 0.0, 0.0), observation(1, 1.0, 1.0, schema_version="2.0")],
    ],
    ids=["cross-run", "cross-probe-set", "cross-source", "cross-schema"],
)
def test_detector_rejects_baselines_without_one_provenance_identity(
    baseline_factory: Callable[[], list[FeatureObservation]],
) -> None:
    detector = TemporalPhaseDetector()

    with pytest.raises(ValueError, match="provenance identity|schema version"):
        detector.fit(baseline_factory())


def test_detector_scores_a_future_checkpoint_from_the_same_run() -> None:
    detector = TemporalPhaseDetector()
    detector.fit([observation(0, 0.0, 0.0), observation(1, 1.0, 1.0)])

    score = detector.score(observation(2, 2.0, 2.0, checkpoint_fingerprint="sha256:new"))

    assert np.isfinite(score.distance_squared)
    assert score.status == "normal"


def observation_with_missing_provenance() -> FeatureObservation:
    item = observation(2, 1.0, 1.0)
    object.__setattr__(item, "provenance", None)
    return item


def observation_with_internally_inconsistent_provenance() -> FeatureObservation:
    item = observation(2, 1.0, 1.0)
    object.__setattr__(item, "run_id", "other-run")
    return item


@pytest.mark.parametrize(
    "invalid_observation_factory",
    [
        observation_with_missing_provenance,
        observation_with_internally_inconsistent_provenance,
        lambda: observation(2, 1.0, 1.0, checkpoint_fingerprint=""),
        lambda: observation(2, 1.0, 1.0, run_id="other-run"),
        lambda: observation(2, 1.0, 1.0, probe_set_id="other-probes"),
        lambda: observation(2, 1.0, 1.0, source="other-source"),
        lambda: observation(2, 1.0, 1.0, schema_version="2.0"),
    ],
    ids=[
        "missing-provenance",
        "internally-inconsistent",
        "empty-checkpoint-fingerprint",
        "cross-run",
        "cross-probe-set",
        "cross-source",
        "cross-schema",
    ],
)
def test_detector_returns_insufficient_data_for_invalid_provenance(
    invalid_observation_factory: Callable[[], FeatureObservation],
) -> None:
    detector = TemporalPhaseDetector()
    detector.fit([observation(0, 0.0, 0.0), observation(1, 1.0, 1.0)])

    assert_insufficient_data(detector.score(invalid_observation_factory()))
