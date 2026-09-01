"""LLM-based judge for AF partition stability with rubric-based scoring.

This module implements a sophisticated LLM judge that can evaluate the stability
and interpretability of AF partition structures using calibrated prompts and
physics-informed rubrics.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from helix.env_api import AFMetrics
from helix.llm_obs import DualSurfaceObservation, ObservationGenerator


@dataclass(frozen=True)
class PhysicsRubric:
    """Physics-informed rubric for LLM evaluation."""

    wave_coherence: Dict[str, Any]
    gauge_invariance: Dict[str, Any]
    ergodic_mixing: Dict[str, Any]
    topological_robustness: Dict[str, Any]
    information_preservation: Dict[str, Any]

    @classmethod
    def default_rubric(cls) -> PhysicsRubric:
        """Create default physics rubric."""
        return cls(
            wave_coherence={
                "description": "Maintains phase relationships across layers",
                "excellent": "CP coisometry error < 0.01",
                "good": "CP coisometry error < 0.05",
                "fair": "CP coisometry error < 0.1",
                "poor": "CP coisometry error ≥ 0.1",
                "weight": 0.25
            },
            gauge_invariance={
                "description": "Predictions unchanged under gauge transformations",
                "excellent": "CP unitality error < 0.01",
                "good": "CP unitality error < 0.05",
                "fair": "CP unitality error < 0.1",
                "poor": "CP unitality error ≥ 0.1",
                "weight": 0.25
            },
            ergodic_mixing={
                "description": "Efficient exploration of state space",
                "excellent": "Spectral gap > 0.7",
                "good": "Spectral gap > 0.4",
                "fair": "Spectral gap > 0.1",
                "poor": "Spectral gap ≤ 0.1",
                "weight": 0.2
            },
            topological_robustness={
                "description": "Persistent features across scales",
                "excellent": "β₁ lifetime / total_depth > 0.8",
                "good": "β₁ lifetime / total_depth > 0.5",
                "fair": "β₁ lifetime / total_depth > 0.2",
                "poor": "β₁ lifetime / total_depth ≤ 0.2",
                "weight": 0.15
            },
            information_preservation={
                "description": "No information destroyed (unitarity)",
                "excellent": "Mass error < 1e-10",
                "good": "Mass error < 1e-6",
                "fair": "Mass error < 1e-3",
                "poor": "Mass error ≥ 1e-3",
                "weight": 0.15
            }
        )


@dataclass(frozen=True)
class LLMJudgmentResult:
    """Result of LLM-based evaluation."""

    overall_label: str  # "A" (stable), "B" (transitional), "C" (unstable)
    confidence: float  # 0.0 to 1.0
    rubric_scores: Dict[str, str]  # Category -> rating (excellent/good/fair/poor)
    explanation: str  # LLM's reasoning
    consistency_flags: List[str]  # Any consistency issues detected
    timestamp: float
    model_used: str


@dataclass
class CalibrationExample:
    """Calibration example for few-shot prompting."""

    structured_metrics: Dict[str, Any]
    natural_language: str
    expected_label: str
    expected_scores: Dict[str, str]
    explanation: str


class LLMJudge:
    """LLM-based judge for AF partition evaluation.

    This judge uses calibrated prompts with physics-informed rubrics to
    evaluate the stability and interpretability of neural network partitions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 500,
        rubric: Optional[PhysicsRubric] = None,
        enable_calibration: bool = True,
        fallback_to_heuristic: bool = True
    ):
        """Initialize LLM judge.

        Parameters
        ----------
        api_key : Optional[str]
            OpenAI API key (will look for OPENAI_API_KEY env var if None)
        model_name : str
            LLM model to use
        temperature : float
            Temperature for LLM generation (clamped for consistency)
        max_tokens : int
            Maximum tokens in response
        rubric : Optional[PhysicsRubric]
            Physics rubric for evaluation
        enable_calibration : bool
            Whether to use few-shot calibration examples
        fallback_to_heuristic : bool
            Whether to fall back to heuristic scoring when API unavailable
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = max(0.0, min(temperature, 0.3))  # Clamp for consistency
        self.max_tokens = max_tokens
        self.rubric = rubric or PhysicsRubric.default_rubric()
        self.enable_calibration = enable_calibration
        self.fallback_to_heuristic = fallback_to_heuristic

        # Initialize observation generator for creating LLM-friendly summaries
        self.obs_generator = ObservationGenerator(
            enable_physics_interpretation=True,
            interpretation_style="concise"
        )

        # Load calibration examples
        self.calibration_examples = self._create_calibration_examples()

        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 1.0  # Minimum seconds between requests

    def evaluate_af_metrics(
        self,
        af_metrics: AFMetrics,
        step: int = 0,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMJudgmentResult:
        """Evaluate AF metrics using LLM judge.

        Parameters
        ----------
        af_metrics : AFMetrics
            AF extraction metrics to evaluate
        step : int
            Current step number
        timestamp : Optional[float]
            Timestamp for the evaluation
        context : Optional[Dict[str, Any]]
            Additional context for evaluation

        Returns
        -------
        LLMJudgmentResult
            LLM judgment with scores and explanation
        """
        if timestamp is None:
            timestamp = time.time()

        # Rate limiting
        self._enforce_rate_limit()

        # Generate dual-surface observation
        try:
            observation = self.obs_generator.generate_observation(
                af_metrics, step, timestamp, metadata=context
            )
        except Exception as e:
            # Fallback to heuristic if observation generation fails
            if self.fallback_to_heuristic:
                return self._heuristic_evaluation(af_metrics, timestamp)
            else:
                raise e

        # Try LLM evaluation
        if self.api_key:
            try:
                return self._llm_evaluation(observation, timestamp)
            except Exception as e:
                if self.fallback_to_heuristic:
                    print(f"LLM evaluation failed ({e}), falling back to heuristic")
                    return self._heuristic_evaluation(af_metrics, timestamp)
                else:
                    raise e
        else:
            if self.fallback_to_heuristic:
                return self._heuristic_evaluation(af_metrics, timestamp)
            else:
                raise ValueError("No API key provided and heuristic fallback disabled")

    def _llm_evaluation(
        self,
        observation: DualSurfaceObservation,
        timestamp: float
    ) -> LLMJudgmentResult:
        """Perform LLM-based evaluation."""
        # Construct prompt
        prompt = self._build_evaluation_prompt(observation)

        # Make API call
        try:
            import openai
            openai.api_key = self.api_key

            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            response_text = response.choices[0].message.content
            self._last_request_time = time.time()

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

        # Parse response
        try:
            return self._parse_llm_response(response_text, timestamp)
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM response: {e}")

    def _heuristic_evaluation(
        self,
        af_metrics: AFMetrics,
        timestamp: float
    ) -> LLMJudgmentResult:
        """Fallback heuristic evaluation when LLM is unavailable."""
        if not af_metrics.levels:
            return LLMJudgmentResult(
                overall_label="C",
                confidence=1.0,
                rubric_scores={},
                explanation="No partition levels found",
                consistency_flags=["empty_partitions"],
                timestamp=timestamp,
                model_used="heuristic_fallback"
            )

        # Use deepest level for evaluation
        level = af_metrics.levels[-1]

        # Evaluate each rubric category
        scores = {}

        # Wave coherence (CP coisometry)
        if level.cp_diagnostics:
            coiso_err = level.cp_diagnostics.coisometry_err_fro
            if coiso_err < 0.01:
                scores["wave_coherence"] = "excellent"
            elif coiso_err < 0.05:
                scores["wave_coherence"] = "good"
            elif coiso_err < 0.1:
                scores["wave_coherence"] = "fair"
            else:
                scores["wave_coherence"] = "poor"
        else:
            scores["wave_coherence"] = "poor"

        # Gauge invariance (CP unitality)
        if level.cp_diagnostics:
            unital_err = level.cp_diagnostics.unital_err_fro
            if unital_err < 0.01:
                scores["gauge_invariance"] = "excellent"
            elif unital_err < 0.05:
                scores["gauge_invariance"] = "good"
            elif unital_err < 0.1:
                scores["gauge_invariance"] = "fair"
            else:
                scores["gauge_invariance"] = "poor"
        else:
            scores["gauge_invariance"] = "poor"

        # Ergodic mixing (spectral gap)
        if level.spectral_gap is not None:
            gap = level.spectral_gap
            if gap > 0.7:
                scores["ergodic_mixing"] = "excellent"
            elif gap > 0.4:
                scores["ergodic_mixing"] = "good"
            elif gap > 0.1:
                scores["ergodic_mixing"] = "fair"
            else:
                scores["ergodic_mixing"] = "poor"
        else:
            scores["ergodic_mixing"] = "poor"

        # Information preservation (mass conservation)
        mass_err = level.mass_error
        if mass_err < 1e-10:
            scores["information_preservation"] = "excellent"
        elif mass_err < 1e-6:
            scores["information_preservation"] = "good"
        elif mass_err < 1e-3:
            scores["information_preservation"] = "fair"
        else:
            scores["information_preservation"] = "poor"

        # Topological robustness (simplified)
        if level.persistent_homology and level.persistent_homology.computed:
            # Use average lifetime as proxy
            avg_lifetimes = level.persistent_homology.average_lifetimes
            if len(avg_lifetimes) > 1:
                lifetime_ratio = avg_lifetimes[1] / level.depth
                if lifetime_ratio > 0.8:
                    scores["topological_robustness"] = "excellent"
                elif lifetime_ratio > 0.5:
                    scores["topological_robustness"] = "good"
                elif lifetime_ratio > 0.2:
                    scores["topological_robustness"] = "fair"
                else:
                    scores["topological_robustness"] = "poor"
            else:
                scores["topological_robustness"] = "poor"
        else:
            scores["topological_robustness"] = "poor"

        # Overall label based on weighted score
        weighted_score = 0.0
        total_weight = 0.0

        score_map = {"excellent": 1.0, "good": 0.75, "fair": 0.5, "poor": 0.25}

        for category, rating in scores.items():
            if hasattr(self.rubric, category):
                weight = getattr(self.rubric, category).get("weight", 0.2)
                weighted_score += score_map[rating] * weight
                total_weight += weight

        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0.0

        # Convert to A/B/C label
        if normalized_score > 0.8:
            overall_label = "A"  # Stable
        elif normalized_score > 0.5:
            overall_label = "B"  # Transitional
        else:
            overall_label = "C"  # Unstable

        # Generate explanation
        explanation = f"Heuristic evaluation: weighted score {normalized_score:.3f}. "
        explanation += f"Mass error: {mass_err:.2e}, "

        if level.cp_diagnostics:
            explanation += f"CP errors: {level.cp_diagnostics.unital_err_fro:.3f} (unital), "
            explanation += f"{level.cp_diagnostics.coisometry_err_fro:.3f} (coisometry). "

        if level.spectral_gap is not None:
            explanation += f"Spectral gap: {level.spectral_gap:.3f}."

        return LLMJudgmentResult(
            overall_label=overall_label,
            confidence=0.8,  # High confidence in heuristic
            rubric_scores=scores,
            explanation=explanation,
            consistency_flags=[],
            timestamp=timestamp,
            model_used="heuristic_fallback"
        )

    def _build_evaluation_prompt(self, observation: DualSurfaceObservation) -> str:
        """Build evaluation prompt for LLM."""
        structured = observation.structured.to_dict()
        natural_lang = observation.natural_language.to_compact_prompt()

        prompt = (
            f"Evaluate this neural network AF partition analysis using the physics rubric below.\n"
            f"\nSTRUCTURED METRICS:\n{json.dumps(structured, indent=2)}\n"
            f"\nNATURAL LANGUAGE SUMMARY:\n{natural_lang}\n"
            f"\nPHYSICS RUBRIC:\n{self._format_rubric()}\n"
            f"\n{self._get_calibration_examples()}\n"
            f"\nINSTRUCTIONS:\n"
            f"1. Evaluate each rubric category (wave_coherence, gauge_invariance, ergodic_mixing,\n"
            f"   topological_robustness, information_preservation)\n"
            f"2. Assign ratings: excellent, good, fair, or poor\n"
            f"3. Provide overall stability label: A (stable), B (transitional), or C (unstable)\n"
            f"4. Include confidence (0.0-1.0) and brief explanation\n"
            f"\nRESPONSE FORMAT (JSON):\n"
            f"{{\n"
            f'    "overall_label": "A/B/C",\n'
            f'    "confidence": 0.0-1.0,\n'
            f'    "rubric_scores": {{\n'
            f'        "wave_coherence": "excellent/good/fair/poor",\n'
            f'        "gauge_invariance": "excellent/good/fair/poor",\n'
            f'        "ergodic_mixing": "excellent/good/fair/poor",\n'
            f'        "topological_robustness": "excellent/good/fair/poor",\n'
            f'        "information_preservation": "excellent/good/fair/poor"\n'
            f'    }},\n'
            f'    "explanation": "Brief reasoning",\n'
            f'    "consistency_flags": ["any issues detected"]\n'
            f'}}'
        )

        return prompt

    def _format_rubric(self) -> str:
        """Format rubric for prompt."""
        rubric_text = ""
        for category, criteria in asdict(self.rubric).items():
            rubric_text += f"\n{category.upper()}:\n"
            rubric_text += f"  Description: {criteria['description']}\n"
            rubric_text += f"  Excellent: {criteria['excellent']}\n"
            rubric_text += f"  Good: {criteria['good']}\n"
            rubric_text += f"  Fair: {criteria['fair']}\n"
            rubric_text += f"  Poor: {criteria['poor']}\n"

        return rubric_text

    def _get_calibration_examples(self) -> str:
        """Get few-shot calibration examples."""
        if not self.enable_calibration or not self.calibration_examples:
            return ""

        examples_text = "\nCALIBRATION EXAMPLES:\n"
        for i, example in enumerate(self.calibration_examples[:2]):  # Limit to 2 examples
            examples_text += f"\nExample {i+1}:\n"
            examples_text += f"Context: {example.natural_language}\n"
            examples_text += f"Expected Label: {example.expected_label}\n"
            examples_text += f"Expected Scores: {json.dumps(example.expected_scores)}\n"
            examples_text += f"Reasoning: {example.explanation}\n"

        return examples_text

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return (
            "You are a physicist specializing in neural network analysis through "
            "operator algebras.\n"
            "\nEvaluate AF (Approximately Finite) partition structures using principles from:\n"
            "- Quantum mechanics (unitarity, gauge invariance)\n"
            "- Statistical mechanics (ergodicity, thermalization)\n"
            "- Topology (persistent homology, robustness)\n"
            "- Information theory (conservation laws)\n"
            "\nBe precise, consistent, and base judgments strictly on the provided rubric.\n"
            "Temperature is low to ensure consistency across evaluations."
        )

    def _parse_llm_response(self, response_text: str, timestamp: float) -> LLMJudgmentResult:
        """Parse LLM response into structured result."""
        try:
            # Decode the first complete JSON object, allowing nested objects and
            # common LLM wrappers such as prose or fenced code blocks.
            decoder = json.JSONDecoder()
            parsed = None
            for index, character in enumerate(response_text):
                if character != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(response_text, index)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    parsed = candidate
                    break

            if parsed is None:
                parsed = json.loads(response_text)

            # Validate required fields
            required_fields = ["overall_label", "confidence", "rubric_scores", "explanation"]
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")

            # Validate label
            if parsed["overall_label"] not in ["A", "B", "C"]:
                raise ValueError(f"Invalid label: {parsed['overall_label']}")

            # Validate confidence
            confidence = float(parsed["confidence"])
            if not 0.0 <= confidence <= 1.0:
                confidence = max(0.0, min(1.0, confidence))

            return LLMJudgmentResult(
                overall_label=parsed["overall_label"],
                confidence=confidence,
                rubric_scores=parsed["rubric_scores"],
                explanation=parsed["explanation"],
                consistency_flags=parsed.get("consistency_flags", []),
                timestamp=timestamp,
                model_used=self.model_name
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise RuntimeError(f"Failed to parse LLM response: {e}\nResponse: {response_text}")

    def _create_calibration_examples(self) -> List[CalibrationExample]:
        """Create calibration examples for few-shot prompting."""
        examples = []

        # Example 1: Excellent case
        examples.append(CalibrationExample(
            structured_metrics={
                "mass_error_l1": 1e-12,
                "cp_unital_error": 0.005,
                "cp_coisometry_error": 0.003,
                "spectral_gap": 0.8,
                "betti_1": 2,
                "avg_lifetime_1": 0.9
            },
            natural_language=(
                "D3 stable (50 regions, β₁=2) | Topology: β₁=2 loops; persistent | "
                "Dynamics: gapped, mass conserved | Physics: coherent wave; gauge invariant; "
                "massive phase"
            ),
            expected_label="A",
            expected_scores={
                "wave_coherence": "excellent",
                "gauge_invariance": "excellent",
                "ergodic_mixing": "excellent",
                "topological_robustness": "excellent",
                "information_preservation": "excellent"
            },
            explanation=(
                "All metrics excellent: perfect mass conservation, low CP errors, large "
                "spectral gap, persistent topology"
            )
        ))

        # Example 2: Poor case
        examples.append(CalibrationExample(
            structured_metrics={
                "mass_error_l1": 0.1,
                "cp_unital_error": 0.2,
                "cp_coisometry_error": 0.15,
                "spectral_gap": 0.05,
                "betti_1": 0,
                "avg_lifetime_1": 0.1
            },
            natural_language=(
                "D1 unstable (10 regions, trivial) | Topology: topologically trivial | "
                "Dynamics: gapless, mass violated | Physics: decoherent; gauge anomaly; "
                "critical phase"
            ),
            expected_label="C",
            expected_scores={
                "wave_coherence": "poor",
                "gauge_invariance": "poor",
                "ergodic_mixing": "poor",
                "topological_robustness": "poor",
                "information_preservation": "poor"
            },
            explanation=(
                "Multiple failures: mass conservation violated, large CP errors, small spectral "
                "gap, no persistent topology"
            )
        ))

        return examples

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            time.sleep(sleep_time)


def create_production_judge() -> LLMJudge:
    """Create LLM judge with production settings."""
    return LLMJudge(
        model_name="gpt-4",
        temperature=0.1,
        max_tokens=500,
        enable_calibration=True,
        fallback_to_heuristic=True
    )


def create_development_judge() -> LLMJudge:
    """Create LLM judge for development/testing."""
    return LLMJudge(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=300,
        enable_calibration=False,
        fallback_to_heuristic=True
    )


__all__ = [
    "PhysicsRubric",
    "LLMJudgmentResult",
    "CalibrationExample",
    "LLMJudge",
    "create_production_judge",
    "create_development_judge",
]
