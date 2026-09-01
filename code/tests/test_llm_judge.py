"""Tests for the structured AF-metrics LLM judge API."""

from __future__ import annotations

import json

import numpy as np
import pytest
from helix.env_api import AFCPDiagnostics, AFLevelMetrics, AFMetrics
from helix.partitions import AFExtraction
from helix.topology import PersistentHomologySummary

from environments.helixenv.llm_judge import LLMJudge, PhysicsRubric

RUBRIC_CATEGORIES = {
    "wave_coherence",
    "gauge_invariance",
    "ergodic_mixing",
    "topological_robustness",
    "information_preservation",
}


def make_level(
    *,
    depth: int = 1,
    mass_error: float = 1e-12,
    unital_error: float | None = 0.005,
    coisometry_error: float = 0.003,
    spectral_gap: float | None = 0.8,
    average_lifetime_1: float | None = 0.9,
) -> AFLevelMetrics:
    """Build a complete AF level with independently chosen diagnostic values."""
    cp_diagnostics = None
    if unital_error is not None:
        cp_diagnostics = AFCPDiagnostics(
            unital_err_fro=unital_error,
            coisometry_err_fro=coisometry_error,
            psd_min_eig_violation=0.0,
        )

    persistent_homology = None
    if average_lifetime_1 is not None:
        persistent_homology = PersistentHomologySummary(
            betti_numbers=(1, 2),
            average_lifetimes=(0.0, average_lifetime_1),
            max_lifetimes=(0.0, average_lifetime_1),
            finite_pairs=(0, 2),
            computed=True,
            backend="test-fixture",
        )

    return AFLevelMetrics(
        depth=depth,
        B=np.ones((1, 1), dtype=np.float64),
        tau_prev=np.array([1.0]),
        tau=np.array([1.0]),
        mass_error=mass_error,
        trace_residual_linf=mass_error,
        wasted_regions=0,
        n_regions=1,
        combinatorial_entropy=0.0,
        cp_diagnostics=cp_diagnostics,
        persistent_homology=persistent_homology,
        spectral_gap=spectral_gap,
        k_theory_invariants={"rank": 1, "nullity": 0, "torsion": []},
    )


def make_metrics(*levels: AFLevelMetrics) -> AFMetrics:
    """Build AFMetrics without relying on a machine-specific repository path."""
    extraction = AFExtraction(
        B_list=[level.B for level in levels],
        tau_list=[level.tau for level in levels],
        parts=[],
        n_list=[level.n_regions for level in levels],
        parent_of_list=[],
    )
    return AFMetrics(levels=levels, extraction=extraction)


@pytest.fixture
def offline_judge(monkeypatch: pytest.MonkeyPatch) -> LLMJudge:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return LLMJudge(api_key=None)


def test_default_rubric_has_complete_normalized_weighting() -> None:
    rubric = PhysicsRubric.default_rubric()
    criteria = vars(rubric)

    assert set(criteria) == RUBRIC_CATEGORIES
    assert sum(category["weight"] for category in criteria.values()) == pytest.approx(1.0)
    assert rubric.wave_coherence["excellent"] == "CP coisometry error < 0.01"
    assert rubric.ergodic_mixing["poor"] == "Spectral gap ≤ 0.1"


def test_calibration_examples_anchor_stable_and_unstable_regimes(
    offline_judge: LLMJudge,
) -> None:
    examples = offline_judge.calibration_examples

    assert [example.expected_label for example in examples] == ["A", "C"]
    assert set(examples[0].expected_scores) == RUBRIC_CATEGORIES
    assert set(examples[0].expected_scores.values()) == {"excellent"}
    assert set(examples[1].expected_scores.values()) == {"poor"}


def test_heuristic_evaluation_scores_an_excellent_af_fixture(
    offline_judge: LLMJudge,
) -> None:
    result = offline_judge.evaluate_af_metrics(
        make_metrics(make_level()),
        step=7,
        timestamp=1234.5,
        context={"phase": "validation"},
    )

    assert result.overall_label == "A"
    assert result.rubric_scores == dict.fromkeys(RUBRIC_CATEGORIES, "excellent")
    assert result.timestamp == 1234.5
    assert result.model_used == "heuristic_fallback"
    assert "weighted score 1.000" in result.explanation


def test_heuristic_evaluation_uses_the_deepest_unstable_af_level(
    offline_judge: LLMJudge,
) -> None:
    unstable = make_level(
        depth=2,
        mass_error=0.1,
        unital_error=0.2,
        coisometry_error=0.15,
        spectral_gap=0.05,
        average_lifetime_1=None,
    )

    result = offline_judge.evaluate_af_metrics(
        make_metrics(make_level(depth=1), unstable),
        timestamp=10.0,
    )

    assert result.overall_label == "C"
    assert result.rubric_scores == dict.fromkeys(RUBRIC_CATEGORIES, "poor")
    assert "Mass error: 1.00e-01" in result.explanation


def test_heuristic_evaluation_reports_empty_partitions(offline_judge: LLMJudge) -> None:
    result = offline_judge.evaluate_af_metrics(make_metrics(), timestamp=9.0)

    assert result.overall_label == "C"
    assert result.confidence == 1.0
    assert result.rubric_scores == {}
    assert result.consistency_flags == ["empty_partitions"]
    assert result.explanation == "No partition levels found"


def test_evaluation_without_key_or_fallback_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = LLMJudge(api_key=None, fallback_to_heuristic=False)

    with pytest.raises(ValueError, match="No API key provided"):
        judge.evaluate_af_metrics(make_metrics(make_level()), timestamp=1.0)


def test_prompt_contains_observation_rubric_calibration_and_json_contract(
    offline_judge: LLMJudge,
) -> None:
    observation = offline_judge.obs_generator.generate_observation(
        make_metrics(make_level()),
        step=3,
        timestamp=42.0,
        metadata={"run": "fixture"},
    )

    prompt = offline_judge._build_evaluation_prompt(observation)

    assert "STRUCTURED METRICS:" in prompt
    assert '"mass_error_l1": 1e-12' in prompt
    assert "NATURAL LANGUAGE SUMMARY:" in prompt
    assert "D1 stable" in prompt
    assert "PHYSICS RUBRIC:" in prompt
    assert "WAVE_COHERENCE:" in prompt
    assert "CALIBRATION EXAMPLES:" in prompt
    assert "Expected Label: A" in prompt
    assert '"rubric_scores"' in prompt
    assert "overall stability label: A (stable), B (transitional), or C (unstable)" in prompt


def test_prompt_omits_calibration_examples_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = LLMJudge(api_key=None, enable_calibration=False)
    observation = judge.obs_generator.generate_observation(
        make_metrics(make_level()), step=0, timestamp=1.0
    )

    assert "CALIBRATION EXAMPLES:" not in judge._build_evaluation_prompt(observation)


def valid_response_payload(*, confidence: float = 0.93) -> dict[str, object]:
    return {
        "overall_label": "A",
        "confidence": confidence,
        "rubric_scores": dict.fromkeys(RUBRIC_CATEGORIES, "excellent"),
        "explanation": "All rubric thresholds are satisfied.",
        "consistency_flags": [],
    }


def test_response_parser_accepts_fenced_nested_json(offline_judge: LLMJudge) -> None:
    response = "Analysis follows.\n```json\n" + json.dumps(valid_response_payload()) + "\n```"

    result = offline_judge._parse_llm_response(response, timestamp=77.0)

    assert result.overall_label == "A"
    assert result.confidence == 0.93
    assert result.rubric_scores == dict.fromkeys(RUBRIC_CATEGORIES, "excellent")
    assert result.timestamp == 77.0
    assert result.model_used == "gpt-4"


@pytest.mark.parametrize(("supplied", "expected"), [(-0.2, 0.0), (1.4, 1.0)])
def test_response_parser_clamps_confidence(
    offline_judge: LLMJudge,
    supplied: float,
    expected: float,
) -> None:
    result = offline_judge._parse_llm_response(
        json.dumps(valid_response_payload(confidence=supplied)), timestamp=1.0
    )

    assert result.confidence == expected


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        json.dumps({"overall_label": "A"}),
        json.dumps({**valid_response_payload(), "overall_label": "stable"}),
    ],
)
def test_response_parser_rejects_malformed_or_incomplete_results(
    offline_judge: LLMJudge,
    response: str,
) -> None:
    with pytest.raises(RuntimeError, match="Failed to parse LLM response"):
        offline_judge._parse_llm_response(response, timestamp=1.0)
