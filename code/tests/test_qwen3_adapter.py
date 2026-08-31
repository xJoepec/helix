import pytest
import torch
from helix.architectures.qwen3_adapter import lora_delta_diagnostics, residual_diagnostics


def test_lora_delta_uses_alpha_over_rank_scaling() -> None:
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    result = lora_delta_diagnostics(a, b, alpha=4.0, rank=2)
    assert result.frobenius_norm == pytest.approx(torch.linalg.norm(2 * b @ a).item())
    assert result.effective_rank == 2


def test_residual_diagnostics_are_finite() -> None:
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = residual_diagnostics(hidden)
    assert result.covariance_trace > 0
    assert 1 <= result.effective_rank <= 2
    assert 0 <= result.anisotropy <= 1
