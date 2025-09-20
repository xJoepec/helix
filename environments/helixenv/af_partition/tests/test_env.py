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


def test_swiss_roll_dataset() -> None:
    """Test Swiss Roll dataset generation."""
    from environments.helixenv.af_partition.dataset import _make_swiss_roll

    X, y = _make_swiss_roll(100, 0.1, 42)
    assert X.shape == (100, 3)  # Swiss roll is 3D
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2  # Binary classification
    assert X.dtype == np.float32


def test_swiss_roll_with_hole() -> None:
    """Test Swiss Roll with hole variant."""
    from environments.helixenv.af_partition.dataset import _make_swiss_roll

    X, y = _make_swiss_roll(100, 0.1, 42, hole=True)
    assert X.shape[1] == 3  # Still 3D
    assert X.shape[0] <= 100  # Fewer points due to hole removal
    assert y.shape[0] == X.shape[0]


def test_concentric_circles_dataset() -> None:
    """Test concentric circles dataset generation."""
    from environments.helixenv.af_partition.dataset import _make_concentric_circles

    X, y = _make_concentric_circles(100, 0.05, 42)
    assert X.shape == (100, 2)  # 2D circles
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2  # Binary classification
    assert X.dtype == np.float32


def test_xor_dataset() -> None:
    """Test XOR dataset generation."""
    from environments.helixenv.af_partition.dataset import _make_xor

    X, y = _make_xor(100, 0.1, 42)
    assert X.shape == (100, 2)  # 2D XOR
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2  # Binary classification
    assert X.dtype == np.float32


def test_s_curve_dataset() -> None:
    """Test S-curve dataset generation."""
    from environments.helixenv.af_partition.dataset import _make_s_curve

    X, y = _make_s_curve(100, 0.1, 42)
    assert X.shape == (100, 3)  # S-curve is 3D
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2  # Binary classification
    assert X.dtype == np.float32


def test_new_scenario_spec_attributes() -> None:
    """Test ScenarioSpec with new attributes."""
    from environments.helixenv.af_partition.dataset import ScenarioSpec

    spec = ScenarioSpec(
        seed=42,
        dataset_type="swiss_roll",
        dataset_kwargs={"hole": True}
    )
    assert spec.dataset_type == "swiss_roll"
    assert spec.dataset_kwargs == {"hole": True}


def test_environment_with_different_datasets() -> None:
    """Test environment works with different dataset types."""
    from environments.helixenv.af_partition.dataset import ScenarioSpec, _scenario_from_spec

    # Test each dataset type
    dataset_types = ["moons", "swiss_roll", "circles", "xor", "s_curve"]

    for dataset_type in dataset_types:
        spec = ScenarioSpec(
            seed=42,
            samples=64,
            noise=0.1,
            width=8,
            epochs=5,
            dataset_type=dataset_type
        )

        example, label = _scenario_from_spec(spec, 0, seed_offset=0)

        assert isinstance(example, dict)
        assert "question" in example
        assert "answer" in example
        assert label in {"A", "B", "C"}


def test_environment_adapts_to_input_dimension() -> None:
    """Test that model input dimension adapts to dataset dimension."""
    from environments.helixenv.af_partition.dataset import ScenarioSpec, _scenario_from_spec

    # 2D dataset (moons)
    spec_2d = ScenarioSpec(seed=42, samples=64, dataset_type="moons", epochs=5)
    example_2d, _ = _scenario_from_spec(spec_2d, 0, seed_offset=0)

    # 3D dataset (swiss_roll)
    spec_3d = ScenarioSpec(seed=42, samples=64, dataset_type="swiss_roll", epochs=5)
    example_3d, _ = _scenario_from_spec(spec_3d, 0, seed_offset=0)

    # Both should succeed despite different input dimensions
    assert "question" in example_2d
    assert "question" in example_3d
