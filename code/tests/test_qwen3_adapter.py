import pytest
import torch
from helix import architectures
from helix.architectures.qwen3_adapter import (
    LoraDeltaDiagnostics,
    ResidualDiagnostics,
    lora_delta_diagnostics,
    residual_diagnostics,
)


def test_qwen_diagnostics_are_exported_from_architectures() -> None:
    assert architectures.LoraDeltaDiagnostics is LoraDeltaDiagnostics
    assert architectures.ResidualDiagnostics is ResidualDiagnostics
    assert architectures.lora_delta_diagnostics is lora_delta_diagnostics
    assert architectures.residual_diagnostics is residual_diagnostics


def test_lora_delta_uses_alpha_over_rank_scaling() -> None:
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    result = lora_delta_diagnostics(a, b, alpha=4.0, rank=2)
    assert isinstance(result, LoraDeltaDiagnostics)
    assert result.frobenius_norm == pytest.approx(torch.linalg.norm(2 * b @ a).item())
    assert result.effective_rank == 2


def test_lora_delta_matches_dense_reference_spectrum() -> None:
    a = torch.tensor(
        [[1.0, -2.0, 0.0, 3.0], [0.5, 1.0, -1.0, 2.0], [2.0, 0.0, 1.0, -1.0]]
    )
    b = torch.tensor(
        [[1.0, 0.0, 2.0], [-1.0, 3.0, 0.5], [2.0, 1.0, -2.0], [0.0, 1.0, 1.0]]
    )
    scale = 2.5 / 3
    singular_values = torch.linalg.svdvals(scale * b @ a)
    expected_rank = int(
        (singular_values > singular_values.max() * 1e-6).sum().item()
    )
    probabilities = singular_values / singular_values.sum()
    expected_entropy = float(
        (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()
    )

    result = lora_delta_diagnostics(a, b, alpha=2.5, rank=3)

    assert result.frobenius_norm == pytest.approx(
        torch.linalg.vector_norm(singular_values).item()
    )
    assert result.effective_rank == expected_rank
    assert result.spectral_entropy == pytest.approx(expected_entropy)


def test_lora_delta_avoids_dense_feature_space_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = 4
    a = torch.randn(rank, 16_384)
    b = torch.randn(12_288, rank)
    original_matmul = torch.Tensor.__matmul__

    def reject_dense_delta(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if tuple(left.shape) == tuple(b.shape) and tuple(right.shape) == tuple(a.shape):
            raise AssertionError("dense B @ A product must not be materialized")
        return original_matmul(left, right)

    monkeypatch.setattr(torch.Tensor, "__matmul__", reject_dense_delta)

    result = lora_delta_diagnostics(a, b, alpha=8.0, rank=rank)

    assert result.frobenius_norm > 0
    assert 1 <= result.effective_rank <= rank


@pytest.mark.parametrize(
    ("a_shape", "b_shape", "rank"),
    [((2, 4), (3, 2), 3), ((2, 4), (3, 3), 2)],
)
def test_lora_delta_rejects_incompatible_factor_shapes(
    a_shape: tuple[int, int], b_shape: tuple[int, int], rank: int
) -> None:
    with pytest.raises(ValueError):
        lora_delta_diagnostics(torch.ones(a_shape), torch.ones(b_shape), alpha=1.0, rank=rank)


def test_residual_diagnostics_are_finite() -> None:
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = residual_diagnostics(hidden)
    assert isinstance(result, ResidualDiagnostics)
    assert result.covariance_trace > 0
    assert 1 <= result.effective_rank <= 2
    assert 0 <= result.anisotropy <= 1


def test_residual_diagnostics_uses_low_rank_spectrum_for_wide_hidden_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    hidden = torch.randn(12, 1_024)
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    expected_trace = float(centered.square().sum().item() / (hidden.shape[0] - 1))
    original_eigvalsh = torch.linalg.eigvalsh

    def reject_hidden_covariance(matrix: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if matrix.ndim == 2 and matrix.shape == (hidden.shape[-1], hidden.shape[-1]):
            raise AssertionError("hidden-dimension covariance must not be materialized")
        return original_eigvalsh(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigvalsh", reject_hidden_covariance)

    result = residual_diagnostics(hidden, max_components=4)

    assert result.covariance_trace == pytest.approx(expected_trace)
    assert 1 <= result.effective_rank <= 4
    assert 0 <= result.anisotropy <= 1


def test_residual_diagnostics_rejects_nonpositive_component_limit() -> None:
    with pytest.raises(ValueError, match="max_components must be positive"):
        residual_diagnostics(torch.ones(2, 3), max_components=0)
