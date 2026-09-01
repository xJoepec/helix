"""Robust multivariate phase-distance scoring across checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ObservationProvenance:
    source: str
    run_id: str
    checkpoint_id: str
    checkpoint_fingerprint: str
    probe_set_id: str
    schema_version: str


@dataclass(frozen=True)
class FeatureObservation:
    run_id: str
    checkpoint_id: str
    step: int
    probe_set_id: str
    features: Mapping[str, float]
    provenance: ObservationProvenance
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not _matches_provenance(self):
            raise ValueError("observation requires matching provenance")


@dataclass(frozen=True)
class PhaseScore:
    distance_squared: float
    status: str
    features: tuple[str, ...]


class TemporalPhaseDetector:
    def __init__(
        self, *, shrinkage: float = 0.2, ridge: float = 1e-8, threshold: float = 9.0
    ) -> None:
        self.shrinkage, self.ridge, self.threshold = shrinkage, ridge, threshold
        self._names: tuple[str, ...] = ()
        self._schema_version: Optional[str] = None
        self._provenance_identity: Optional[tuple[str, str, str, str]] = None

    def fit(self, observations: Sequence[FeatureObservation]) -> None:
        if len(observations) < 2:
            raise ValueError("at least two baseline observations are required")
        if not all(_has_complete_matching_provenance(item) for item in observations):
            raise ValueError("baseline observations require complete matching provenance")
        schema_versions = {item.schema_version for item in observations}
        if len(schema_versions) != 1:
            raise ValueError("baseline observations must share one schema version")
        provenance_identities = {_provenance_identity(item) for item in observations}
        if len(provenance_identities) != 1:
            raise ValueError("baseline observations must share one provenance identity")
        names = set(observations[0].features)
        for item in observations[1:]:
            names &= set(item.features)
        fitted_names = tuple(sorted(names))
        if not fitted_names:
            raise ValueError("baseline observations share no features")
        matrix = np.array(
            [[item.features[name] for name in fitted_names] for item in observations], dtype=float
        )
        if not np.isfinite(matrix).all():
            raise ValueError("baseline feature values must be finite")
        mean = matrix.mean(axis=0)
        covariance = np.atleast_2d(np.cov(matrix, rowvar=False))
        diagonal = np.diag(np.diag(covariance))
        regularized = (1 - self.shrinkage) * covariance + self.shrinkage * diagonal
        regularized += self.ridge * np.eye(len(fitted_names))
        inverse = np.linalg.pinv(regularized)

        self._names = fitted_names
        self._schema_version = observations[0].schema_version
        self._provenance_identity = _provenance_identity(observations[0])
        self._mean = mean
        self._inverse = inverse

    def score(self, observation: FeatureObservation) -> PhaseScore:
        if not self._names:
            raise RuntimeError("detector has not been fitted")
        has_matching_provenance = (
            _has_complete_matching_provenance(observation)
            and _provenance_identity(observation) == self._provenance_identity
        )
        has_required_features = all(name in observation.features for name in self._names)
        if not has_matching_provenance or not has_required_features:
            return PhaseScore(float("nan"), "insufficient_data", self._names)
        vector = np.array([observation.features[name] for name in self._names], dtype=float)
        if not np.isfinite(vector).all():
            return PhaseScore(float("nan"), "insufficient_data", self._names)
        delta = vector - self._mean
        distance = float(delta @ self._inverse @ delta)
        status = "candidate_transition" if distance >= self.threshold else "normal"
        return PhaseScore(distance, status, self._names)


def _matches_provenance(observation: FeatureObservation) -> bool:
    provenance = observation.provenance
    return isinstance(provenance, ObservationProvenance) and (
        observation.run_id,
        observation.checkpoint_id,
        observation.probe_set_id,
        observation.schema_version,
    ) == (
        provenance.run_id,
        provenance.checkpoint_id,
        provenance.probe_set_id,
        provenance.schema_version,
    )


def _has_complete_matching_provenance(observation: FeatureObservation) -> bool:
    if not _matches_provenance(observation):
        return False
    provenance = observation.provenance
    assert isinstance(provenance, ObservationProvenance)
    return all(
        isinstance(value, str) and value.strip()
        for value in (
            provenance.source,
            provenance.run_id,
            provenance.checkpoint_id,
            provenance.checkpoint_fingerprint,
            provenance.probe_set_id,
            provenance.schema_version,
        )
    )


def _provenance_identity(observation: FeatureObservation) -> tuple[str, str, str, str]:
    provenance = observation.provenance
    assert isinstance(provenance, ObservationProvenance)
    return (
        provenance.source,
        provenance.run_id,
        provenance.probe_set_id,
        provenance.schema_version,
    )
