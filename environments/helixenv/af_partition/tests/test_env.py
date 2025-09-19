from __future__ import annotations

import json
import numpy as np
import pytest
import torch
import torch.nn as nn

from helix.env_api import extract_af_metrics

from environments.helixenv.af_partition.dataset import build_af_examples
from environments.helixenv.af_partition.env import AFPartitionEnv, load_verifiers_environment


def _make_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
    )


def _make_data(n: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2)).astype(np.float32)
    return X


def test_extract_af_metrics_roundtrip() -> None:
    model = _make_model()
    X = _make_data()
    metrics = extract_af_metrics(model, X)
    assert len(metrics.levels) == 2
    first = metrics.levels[0]
    assert first.depth == 1
    assert first.n_regions > 0
    assert first.mass_error >= 0.0


def test_environment_emits_levels_in_order() -> None:
    model = _make_model()
    X = _make_data()
    env = AFPartitionEnv(model, X, max_depth=2)

    step0 = env.reset()
    assert step0.info["depth"] == 1
    assert step0.obs["depth"] == 1
    assert step0.done is False

    step1 = env.step({"note": "advance"})
    assert step1.info["depth"] == 2
    assert step1.obs["depth"] == 2
    assert step1.done is True
    assert env.history[-1]["note"] == "advance"

    # Additional step after completion returns terminal payload
    step2 = env.step()
    assert step2.done is True
    assert step2.obs == {}


def test_environment_validates_weights_shape() -> None:
    model = _make_model()
    X = _make_data()
    bad_weights = np.ones(10, dtype=np.float64) / 10.0
    with pytest.raises(ValueError):
        AFPartitionEnv(model, X, sample_weights=bad_weights)

    zero_weights = np.zeros(X.shape[0], dtype=np.float64)
    with pytest.raises(ValueError):
        AFPartitionEnv(model, X, sample_weights=zero_weights)


def test_build_af_examples_has_all_labels() -> None:
    examples = build_af_examples(seed=123)
    letters = {json.loads(item["answer"]) ["label"] for item in examples}
    assert letters == {"A", "B", "C"}


def test_llm_judge_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        load_verifiers_environment(enable_llm_judge=True)
