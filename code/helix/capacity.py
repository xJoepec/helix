from __future__ import annotations

"""Capacity loss diagnostics based on trainable layer spectra."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

try:  # pragma: no cover - optional torch dependency
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CapacityLossMetrics:
    """Aggregate view of per-layer capacity loss scores."""

    layer_scores: Dict[str, float]
    mean_loss: float
    max_loss: float
    nontrainable_layers: int
    total_layers: int
    computed: bool
    notes: tuple[str, ...] = ()


def compute_capacity_loss(
    model: "nn.Module",
    *,
    singular_value_threshold: float = 1e-3,
) -> CapacityLossMetrics:
    """Estimate capacity loss as the fraction of collapsed singular values per layer."""

    if torch is None or nn is None:
        return CapacityLossMetrics(
            layer_scores={},
            mean_loss=float("nan"),
            max_loss=float("nan"),
            nontrainable_layers=0,
            total_layers=0,
            computed=False,
            notes=("PyTorch is required to inspect model capacity.",),
        )

    layer_scores: Dict[str, float] = {}
    nontrainable = 0
    total = 0
    notes: list[str] = []

    for name, module in model.named_modules():
        if name == "":
            continue
        if not any(p.requires_grad for p in module.parameters(recurse=False)):
            nontrainable += 1
            total += 1
            continue
        weight: torch.Tensor | None = getattr(module, "weight", None)
        if weight is None or weight.ndim < 2:
            total += 1
            continue

        with torch.no_grad():
            matrix = weight.detach().cpu().float().reshape(weight.shape[0], -1)
        if matrix.numel() == 0:
            total += 1
            continue

        try:
            singular_values = torch.linalg.svdvals(matrix)
        except RuntimeError:
            # Fallback to numpy if torch.linalg.svdvals unavailable
            singular_values = torch.from_numpy(np.linalg.svd(matrix.numpy(), compute_uv=False))

        max_sv = float(torch.max(singular_values))
        if max_sv <= 0:
            ratio = 1.0
        else:
            collapsed = torch.sum(singular_values < singular_value_threshold * max_sv)
            ratio = float(collapsed) / float(singular_values.numel())
        layer_scores[name] = ratio
        total += 1

    if not layer_scores:
        notes.append("No trainable matrix weights detected; capacity metrics unavailable.")
        return CapacityLossMetrics(
            layer_scores={},
            mean_loss=float("nan"),
            max_loss=float("nan"),
            nontrainable_layers=nontrainable,
            total_layers=total,
            computed=False,
            notes=tuple(notes),
        )

    values = np.fromiter(layer_scores.values(), dtype=np.float64)
    mean_loss = float(values.mean())
    max_loss = float(values.max())

    return CapacityLossMetrics(
        layer_scores=layer_scores,
        mean_loss=mean_loss,
        max_loss=max_loss,
        nontrainable_layers=nontrainable,
        total_layers=total,
        computed=True,
        notes=tuple(notes),
    )


__all__ = ["CapacityLossMetrics", "compute_capacity_loss"]

