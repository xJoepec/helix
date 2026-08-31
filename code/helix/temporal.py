"""Robust multivariate phase-distance scoring across checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureObservation:
    run_id: str
    checkpoint_id: str
    step: int
    probe_set_id: str
    features: Mapping[str, float]
    schema_version: str = "1.0"


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

    def fit(self, observations: Sequence[FeatureObservation]) -> None:
        if len(observations) < 2:
            raise ValueError("at least two baseline observations are required")
        names = set(observations[0].features)
        for item in observations[1:]:
            names &= set(item.features)
        self._names = tuple(sorted(names))
        if not self._names:
            raise ValueError("baseline observations share no features")
        matrix = np.array(
            [[item.features[name] for name in self._names] for item in observations], dtype=float
        )
        self._mean = matrix.mean(axis=0)
        covariance = np.atleast_2d(np.cov(matrix, rowvar=False))
        diagonal = np.diag(np.diag(covariance))
        regularized = (1 - self.shrinkage) * covariance + self.shrinkage * diagonal
        regularized += self.ridge * np.eye(len(self._names))
        self._inverse = np.linalg.pinv(regularized)

    def score(self, observation: FeatureObservation) -> PhaseScore:
        if not self._names:
            raise RuntimeError("detector has not been fitted")
        vector = np.array([observation.features[name] for name in self._names], dtype=float)
        delta = vector - self._mean
        distance = float(delta @ self._inverse @ delta)
        status = "candidate_transition" if distance >= self.threshold else "normal"
        return PhaseScore(distance, status, self._names)
