from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency for runtime generation
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

from helix import build_V_from_incidence, sanity_check_ucp
from helix.env_api import AFLevelMetrics, AFMetrics, extract_af_metrics


@dataclass(frozen=True)
class ScenarioSpec:
    seed: int
    samples: int = 512
    noise: float = 0.08
    width: int = 16
    epochs: int = 60
    train: bool = True
    description: str = ""


DEFAULT_SCENARIOS: Sequence[ScenarioSpec] = (
    ScenarioSpec(seed=11, samples=512, noise=0.08, width=8, epochs=40, train=True, description="balanced moderate capacity"),
    ScenarioSpec(seed=21, samples=512, noise=0.02, width=48, epochs=140, train=True, description="very wide network"),
    ScenarioSpec(seed=31, samples=512, noise=0.10, width=4, epochs=30, train=True, description="narrow architecture"),
    ScenarioSpec(seed=41, samples=640, noise=0.07, width=12, epochs=80, train=True, description="well trained baseline"),
    ScenarioSpec(seed=51, samples=512, noise=0.03, width=32, epochs=120, train=True, description="high expressivity"),
    ScenarioSpec(seed=61, samples=384, noise=0.20, width=5, epochs=0, train=False, description="untrained noisy model"),
)


LABEL_MAP = {
    "stable": ("A", "Stable / Healthy"),
    "capacity": ("B", "Over-partitioned / Wasted capacity"),
    "collapsed": ("C", "Collapsed / Unstable"),
}


def build_af_examples(
    *,
    specs: Sequence[ScenarioSpec] | None = None,
    seed: int = 1234,
) -> list[dict[str, Any]]:
    """Generate AF diagnostic scenarios suitable for verifiers datasets."""

    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required to synthesize operator algebra scenarios.")

    specs = tuple(specs or DEFAULT_SCENARIOS)
    examples: list[dict[str, Any]] = []
    observed_labels: set[str] = set()

    for idx, spec in enumerate(specs):
        example, label_letter = _scenario_from_spec(spec, idx, seed_offset=0)
        examples.append(example)
        observed_labels.add(label_letter)

    expected_labels = {label for label, _ in LABEL_MAP.values()}
    if observed_labels != expected_labels:
        missing = expected_labels - observed_labels
        raise RuntimeError(f"Missing label coverage in AF examples: {missing}")

    return examples


# ---------------------------------------------------------------------------
# Scenario synthesis helpers
# ---------------------------------------------------------------------------


def _scenario_from_spec(spec: ScenarioSpec, index: int, *, seed_offset: int) -> tuple[dict[str, Any], str]:
    torch.manual_seed(spec.seed + seed_offset)
    np.random.seed(spec.seed + seed_offset)

    X, y = _make_moons(spec.samples, spec.noise, spec.seed + seed_offset)
    model = _build_mlp(d_in=2, width=spec.width, d_out=2)

    if spec.train and spec.epochs > 0 and y is not None:
        _train_model(model, X, y, epochs=spec.epochs)

    metrics = extract_af_metrics(model, X)
    cp_stats = _compute_cp_stats(metrics)
    tag, rationale = _classify(metrics, cp_stats)
    label_letter, label_name = LABEL_MAP[tag]

    prompt = _format_prompt(index, spec, metrics, cp_stats)
    answer_payload = {
        "label": label_letter,
        "label_name": label_name,
        "tag": tag,
        "rationale": rationale,
        "criteria": _collect_criteria(metrics, cp_stats),
    }
    info = {
        "id": f"oa-af-{index}",
        "seed": spec.seed,
        "samples": spec.samples,
        "noise": spec.noise,
        "width": spec.width,
        "epochs": spec.epochs,
        "trained": bool(spec.train and spec.epochs > 0),
        "description": spec.description,
    }

    example = {
        "question": prompt,
        "answer": json.dumps(answer_payload),
        "task": "helix-oa-af",
        "info": info,
    }
    return example, label_letter


def _make_moons(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, math.pi, n // 2)
    x1 = np.c_[np.cos(theta), np.sin(theta)]
    x2 = np.c_[1 - np.cos(theta), 1 - np.sin(theta)] + np.array([0.1, -0.2])
    X = np.vstack([x1, x2]).astype(np.float32)
    X += noise * rng.standard_normal(X.shape).astype(np.float32)
    y = np.r_[np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)]
    return X, y


def _build_mlp(d_in: int, width: int, d_out: int) -> nn.Module:
    layers: list[nn.Module] = []
    last = d_in
    for _ in range(2):
        layers.append(nn.Linear(last, width))
        layers.append(nn.ReLU(inplace=False))
        last = width
    layers.append(nn.Linear(last, d_out))
    return nn.Sequential(*layers)


def _train_model(model: nn.Module, X: np.ndarray, y: np.ndarray, *, epochs: int) -> None:
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(max(1, epochs)):
        opt.zero_grad()
        logits = model(X_t)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()
        opt.step()


def _compute_cp_stats(metrics: AFMetrics) -> list[dict[str, float]]:
    stats: list[dict[str, float]] = []
    extraction = metrics.extraction
    for idx, B in enumerate(extraction.B_list):
        tau_prev = np.array([1.0]) if idx == 0 else extraction.tau_list[idx - 1]
        tau_cur = extraction.tau_list[idx]
        V = build_V_from_incidence(B, tau_prev, tau_cur)
        diag = sanity_check_ucp(V, trials=8)
        stats.append(diag)
    return stats


def _collect_criteria(metrics: AFMetrics, cp_stats: Sequence[dict[str, float]]) -> dict[str, float]:
    last = metrics.levels[-1]
    wasted_ratio = last.wasted_regions / max(last.n_regions, 1)
    mass_err_max = max(level.mass_error for level in metrics.levels)
    entropy_last = last.combinatorial_entropy
    n_regions_last = last.n_regions
    coiso_max = max(stat["coisometry_err_fro"] for stat in cp_stats)
    return {
        "wasted_ratio": float(wasted_ratio),
        "max_mass_error": float(mass_err_max),
        "entropy_last": float(entropy_last),
        "n_regions_last": float(n_regions_last),
        "max_coiso": float(coiso_max),
    }


def _classify(metrics: AFMetrics, cp_stats: Sequence[dict[str, float]]) -> tuple[str, str]:
    last = metrics.levels[-1]
    wasted_ratio = last.wasted_regions / max(last.n_regions, 1)
    mass_err_max = max(level.mass_error for level in metrics.levels)
    entropy_last = last.combinatorial_entropy
    n_regions_last = last.n_regions
    start_regions = metrics.levels[0].n_regions if metrics.levels else 1
    growth_factor = n_regions_last / max(start_regions, 1)
    coiso_max = max(stat["coisometry_err_fro"] for stat in cp_stats)

    if (
        wasted_ratio >= 0.2
        or n_regions_last <= 6
        or growth_factor <= 1.2
        or mass_err_max > 1e-3
        or coiso_max > 0.12
    ):
        rationale = (
            "Wasted mass or limited refinement (very low region growth) indicates the partition collapsed."
        )
        return "collapsed", rationale

    if n_regions_last >= 28 or (growth_factor >= 3.5 and entropy_last >= 1.3):
        rationale = "Region growth and entropy are disproportionately large, signalling wasted capacity."
        return "capacity", rationale

    rationale = "Mass is consistent and region growth is moderate, indicating a stable extraction."
    return "stable", rationale


def _format_prompt(
    index: int,
    spec: ScenarioSpec,
    metrics: AFMetrics,
    cp_stats: Sequence[dict[str, float]],
) -> str:
    lines: list[str] = []
    lines.append(f"Scenario {index + 1}: Helix AF diagnostics")
    lines.append(
        "Model setup: "
        f"samples={spec.samples}, noise={spec.noise:.3f}, width={spec.width}, epochs={spec.epochs}, "
        f"trained={'yes' if spec.train and spec.epochs > 0 else 'no'}, seed={spec.seed}."
    )
    if spec.description:
        lines.append(f"Notes: {spec.description}.")
    lines.append("")
    lines.append("Depth summary (wasted_ratio = zero-mass cells / total):")
    header = f"{'depth':>5} {'regions':>9} {'mass_err':>12} {'wasted':>10} {'wasted_ratio':>14} {'entropy':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for level in metrics.levels:
        wasted_ratio = level.wasted_regions / max(level.n_regions, 1)
        lines.append(
            f"{level.depth:>5} {level.n_regions:>9} {level.mass_error:>12.3e} "
            f"{level.wasted_regions:>10} {wasted_ratio:>14.2f} {level.combinatorial_entropy:>10.3f}"
        )
    lines.append("")
    lines.append("CP diagnostics (per depth):")
    lines.append(f"{'depth':>5} {'unital_err':>12} {'coiso_err':>12} {'min_psd':>12}")
    lines.append("-" * 47)
    for idx, diag in enumerate(cp_stats, start=1):
        lines.append(
            f"{idx:>5} {diag['unital_err_fro']:>12.3e} {diag['coisometry_err_fro']:>12.3e} {diag['psd_min_eig_violation']:>12.3e}"
        )
    lines.append("")
    lines.append("Classify this run into one of the following categories:")
    lines.append("  A. Stable / healthy partitioning")
    lines.append("  B. Over-partitioned / wasted capacity")
    lines.append("  C. Collapsed / unstable")
    lines.append("")
    lines.append("Respond with a single letter (A, B, or C).")
    return "\n".join(lines)


__all__ = [
    "ScenarioSpec",
    "DEFAULT_SCENARIOS",
    "build_af_examples",
]
