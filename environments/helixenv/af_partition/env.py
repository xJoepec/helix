from __future__ import annotations

"""Operator-algebra AF environments for verifiers and direct CLI usage."""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from datasets import Dataset
except Exception:  # pragma: no cover
    Dataset = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency at runtime
    import verifiers as vf
    from verifiers import Parser
    from verifiers.core import Env as _VFEnv
    from verifiers.core import Step as _VFStep
    from verifiers.types import Messages
except Exception:  # pragma: no cover - graceful fallback for tests / dev shells
    vf = None
    Parser = None
    _VFEnv = object
    Messages = list  # type: ignore[assignment]

    @dataclass
    class _VFStep:  # type: ignore[override]
        obs: Dict[str, Any]
        reward: float
        done: bool
        info: Dict[str, Any]

from helix.env_api import AFLevelMetrics, AFMetrics, af_feature_vector, extract_af_metrics

from .dataset import build_af_examples


@dataclass
class AFObservation:
    """Structured observation emitted by the interactive AF environment."""

    depth: int
    feature_vector: np.ndarray
    n_regions: int
    mass_error: float
    wasted_regions: int
    combinatorial_entropy: float

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - tiny wrapper
        return {
            "depth": self.depth,
            "features": self.feature_vector,
            "n_regions": self.n_regions,
            "mass_error": self.mass_error,
            "wasted_regions": self.wasted_regions,
            "combinatorial_entropy": self.combinatorial_entropy,
        }


class AFPartitionEnv(_VFEnv):
    """Roll over AF partition levels and surface Helix diagnostics as rewards."""

    def __init__(
        self,
        model: Any,
        dataset: np.ndarray,
        *,
        sample_weights: Optional[np.ndarray] = None,
        max_depth: Optional[int] = None,
        mass_tol: float = 1e-10,
        mass_weight: float = 1.0,
        wasted_weight: float = 0.1,
    ) -> None:
        self._dataset = np.asarray(dataset, dtype=np.float32)
        if self._dataset.ndim != 2:
            raise ValueError("dataset must be 2D (N, d)")
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=np.float64)
            if weights.shape != (self._dataset.shape[0],):
                raise ValueError("sample_weights shape must match number of samples")
            if np.any(weights < 0):
                raise ValueError("sample_weights must be non-negative")
            total = float(weights.sum())
            if total <= 0:
                raise ValueError("sample_weights must sum to a positive value")
            if not np.isclose(total, 1.0):
                weights = weights / total
        else:
            weights = None

        metrics = extract_af_metrics(model, self._dataset, sample_weights=weights, mass_tol=mass_tol)
        levels = list(metrics.levels)
        if max_depth is not None:
            levels = levels[:max_depth]
        if not levels:
            raise ValueError("Model produced no ReLU partitions; at least one depth is required.")

        self._metrics = metrics
        self._levels = levels
        self._mass_weight = float(mass_weight)
        self._wasted_weight = float(wasted_weight)
        self._cursor = 0
        self._history: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core Env API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None) -> _VFStep:  # type: ignore[override]
        del seed  # deterministic environment (seed ignored)
        self._cursor = 0
        self._history.clear()
        level = self._levels[self._cursor]
        obs = self._build_observation(level)
        reward = self._reward(level)
        done = len(self._levels) == 1
        self._cursor += 1
        return _VFStep(obs=obs.to_dict(), reward=reward, done=done, info={"depth": level.depth})

    def step(self, action: Optional[Dict[str, Any]] = None) -> _VFStep:  # type: ignore[override]
        self._history.append(action or {})
        if self._cursor >= len(self._levels):
            return _VFStep(obs={}, reward=0.0, done=True, info={"message": "complete"})

        level = self._levels[self._cursor]
        obs = self._build_observation(level)
        reward = self._reward(level)
        done = self._cursor == len(self._levels) - 1
        self._cursor += 1
        info = {"depth": level.depth}
        if action is not None:
            info["action"] = action
        return _VFStep(obs=obs.to_dict(), reward=reward, done=done, info=info)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def levels(self) -> Sequence[AFLevelMetrics]:
        return self._levels

    @property
    def metrics(self) -> AFMetrics:
        return self._metrics

    @property
    def history(self) -> Sequence[Dict[str, Any]]:
        return tuple(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reward(self, level: AFLevelMetrics) -> float:
        return float(-self._mass_weight * level.mass_error - self._wasted_weight * level.wasted_regions)

    def _build_observation(self, level: AFLevelMetrics) -> AFObservation:
        return AFObservation(
            depth=level.depth,
            feature_vector=af_feature_vector(level),
            n_regions=level.n_regions,
            mass_error=level.mass_error,
            wasted_regions=level.wasted_regions,
            combinatorial_entropy=level.combinatorial_entropy,
        )


# ---------------------------------------------------------------------------
# Verifiers loader
# ---------------------------------------------------------------------------


def load_verifiers_environment(
    *,
    system_prompt: Optional[str] = None,
    use_think: bool = False,
    seed: int = 1234,
    max_episodes: Optional[int] = None,
    enable_llm_judge: bool = False,
    llm_judge_model: str = "gpt-4.1-mini",
    llm_judge_base_url: str = "https://api.openai.com/v1",
    llm_judge_api_key_var: str = "OPENAI_API_KEY",
    llm_judge_prompt: Optional[str] = None,
    **env_kwargs: Any,
):
    """Create a verifiers SingleTurnEnv for operator-algebra diagnostics."""

    if vf is None or Parser is None:
        raise RuntimeError(
            "verifiers package is required to load the operator algebra training environment."
        )

    api_key: str = ""
    if enable_llm_judge:
        api_key = os.getenv(llm_judge_api_key_var, "")
        if not api_key:
            raise RuntimeError(
                "enable_llm_judge=True but no API key found. Set the key via the CLI (helix helixenv "
                f"--api-key ...) or export {llm_judge_api_key_var}."
            )

    dataset_cls = _ensure_datasets_available()

    examples = build_af_examples(seed=seed)
    dataset = dataset_cls.from_list(examples)

    if max_episodes is not None:
        max_n = min(max_episodes, len(dataset))
        dataset = dataset.select(range(max_n))

    parser = vf.ThinkParser(_extract_letter) if use_think else Parser(_extract_letter)

    def score_completion(completion: Messages, answer: Any, **kwargs: Any) -> float:
        data = _ensure_dict(answer)
        target = (data.get("label") or "").strip().upper()
        if not target:
            return 0.0
        prediction = parser.parse_answer(completion)
        if prediction is None:
            return 0.0
        pred_letter = _normalize_letter(str(prediction))
        if not pred_letter:
            normalized_text = _normalize_text(str(prediction))
            for letter, tag in (("A", "stable"), ("B", "capacity"), ("C", "collapsed")):
                if tag in normalized_text:
                    pred_letter = letter
                    break
        return 1.0 if pred_letter == target else 0.0

    rubric = vf.Rubric(funcs=[score_completion], weights=[1.0], parser=parser)

    if enable_llm_judge:
        if not hasattr(vf, "JudgeRubric"):
            raise RuntimeError("verifiers.JudgeRubric not available; upgrade verifiers package.")
        from openai import AsyncOpenAI  # defer import until key available

        judge_prompt_text = llm_judge_prompt or (
            "You are validating Helix operator-algebra diagnostics. Given the prompt (with metrics), "
            "the assistant's letter answer, and the gold label, decide if the answer is correct.\n"
            "Reply using either:\n"
            "  verdict: correct\n"
            "or\n"
            "  verdict: incorrect\n"
            "You may optionally add a short explanation on a new line starting with 'explanation:'."
        )

        judge_client = AsyncOpenAI(api_key=api_key, base_url=llm_judge_base_url)
        judge_rubric = vf.JudgeRubric(
            judge_client=judge_client,
            judge_model=llm_judge_model,
            judge_prompt=judge_prompt_text,
            parser=parser,
        )

        async def judge_score(prompt, completion, answer, state, **kwargs) -> float:
            data = _ensure_dict(answer)
            target = (data.get("label") or "").strip().upper()
            if not target:
                return 0.0

            resp = await judge_rubric.judge(
                prompt=prompt,
                completion=completion,
                answer=answer,
                state=state,
            )

            text = str(resp)
            verdict_match = re.search(r"verdict\s*[:=]\s*(correct|incorrect)", text, re.IGNORECASE)
            if verdict_match:
                return 1.0 if verdict_match.group(1).lower() == "correct" else 0.0

            pred_letter = _normalize_letter(text)
            if not pred_letter:
                letter_match = re.search(r"letter\s*[:=]\s*([ABC])", text, re.IGNORECASE)
                if letter_match:
                    pred_letter = letter_match.group(1).upper()
            if not pred_letter:
                normalized_text = _normalize_text(text)
                for letter, tag in (("A", "stable"), ("B", "capacity"), ("C", "collapsed")):
                    if tag in normalized_text:
                        pred_letter = letter
                        break
            if pred_letter:
                return 1.0 if pred_letter == target else 0.0
            return 0.0

        judge_rubric.add_reward_func(judge_score, weight=1.0)
        rubric = judge_rubric

    default_system_prompt = (
        "You are a Helix operator-algebra analyst. Each prompt provides AF partition diagnostics "
        "for a trained model. Respond with a single letter (A, B, or C) indicating the best category."
    )

    prompt_text = system_prompt or default_system_prompt

    env = vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        system_prompt=prompt_text,
        **env_kwargs,
    )
    return env


def load_environment(*args: Any, **kwargs: Any):
    """Backward compatible alias for verifiers loaders."""

    return load_verifiers_environment(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helper utilities for parsing answers
# ---------------------------------------------------------------------------


def _normalize_letter(raw: str) -> str:
    text = raw.strip().upper()
    if not text:
        return ""
    if text[0] in {"A", "B", "C"}:
        return text[0]
    match = re.search(r"\b([ABC])\b", text)
    return match.group(1) if match else ""


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _extract_letter(output: str) -> str | None:
    if not output:
        return None
    letter = _normalize_letter(output)
    if letter:
        return letter
    normalized = _normalize_text(output)
    if "stable" in normalized:
        return "A"
    if any(word in normalized for word in ["capacity", "over", "overpartition", "waste"]):
        return "B"
    if any(word in normalized for word in ["collapse", "unstable", "fail", "degenerate"]):
        return "C"
    return None


def _ensure_dict(answer: Any) -> dict[str, Any]:
    if isinstance(answer, dict):
        return answer
    if isinstance(answer, str):
        try:
            return json.loads(answer)
        except Exception:
            return {}
    return {}


def _ensure_datasets_available():
    """Import datasets.Dataset, installing the package if required."""

    global Dataset
    if Dataset is not None:  # type: ignore[truthy-bool]
        return Dataset  # pragma: no cover - already available

    try:
        from datasets import Dataset as _Dataset

        Dataset = _Dataset
        return Dataset
    except Exception:
        _install_datasets()
        try:
            from datasets import Dataset as _Dataset

            Dataset = _Dataset
            return Dataset
        except Exception as exc:  # pragma: no cover - install failed
            raise RuntimeError(
                "Unable to import the 'datasets' package even after attempting installation. "
                "Install it manually with 'pip install datasets'."
            ) from exc


def _install_datasets() -> None:
    """Attempt to install the datasets package using pip."""

    print("[helix] Installing required dependency 'datasets'...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "datasets"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - install failure
        raise RuntimeError(
            "Automatic installation of 'datasets' failed. Install it manually with 'pip install datasets'."
        ) from exc


__all__ = [
    "AFPartitionEnv",
    "AFObservation",
    "load_environment",
    "load_verifiers_environment",
]
