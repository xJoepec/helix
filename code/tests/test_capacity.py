from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from helix.capacity import compute_capacity_loss


def test_capacity_loss_detects_collapsed_layer() -> None:
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    metrics = compute_capacity_loss(model)
    assert metrics.computed is True
    assert "0" in metrics.layer_scores
    assert 0.0 <= metrics.layer_scores["0"] <= 1.0

    with torch.no_grad():
        model[0].weight.zero_()

    metrics_zero = compute_capacity_loss(model)
    assert pytest.approx(metrics_zero.layer_scores["0"], abs=1e-6) == 1.0
    assert metrics_zero.max_loss == pytest.approx(1.0, abs=1e-6)
