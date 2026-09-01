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
    torch.manual_seed(0)
    a = torch.randn(rank, 16_384)
    b = torch.randn(12_288, rank)

    def reject_dense_delta(
        left: torch.Tensor, right: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        if (
            tuple(left.shape) == tuple(b.shape)
            and tuple(right.shape) == tuple(a.shape)
            and left.data_ptr() == b.data_ptr()
            and right.data_ptr() == a.data_ptr()
        ):
            raise AssertionError("dense B @ A product must not be materialized")
        return original_matmul(left, right, *args, **kwargs)

    original_matmul = torch.matmul
    monkeypatch.setattr(torch, "matmul", reject_dense_delta)
    monkeypatch.setattr(torch, "mm", reject_dense_delta)
    monkeypatch.setattr(torch.Tensor, "__matmul__", reject_dense_delta)

    result = lora_delta_diagnostics(a, b, alpha=8.0, rank=rank)

    assert result.frobenius_norm > 0
    assert 1 <= result.effective_rank <= rank


@pytest.mark.parametrize(
    ("alpha", "bad_factor", "message"),
    [
        (float("nan"), None, "alpha must be finite"),
        (float("inf"), None, "alpha must be finite"),
        (1.0, "a", "lora_a must contain only finite values"),
        (1.0, "b", "lora_b must contain only finite values"),
    ],
)
def test_lora_delta_rejects_nonfinite_inputs_before_linear_algebra(
    monkeypatch: pytest.MonkeyPatch,
    alpha: float,
    bad_factor: str | None,
    message: str,
) -> None:
    a = torch.eye(2)
    b = torch.eye(2)
    if bad_factor == "a":
        a[0, 0] = float("nan")
    elif bad_factor == "b":
        b[0, 0] = float("inf")

    def reject_linear_algebra(*args, **kwargs):
        raise AssertionError("linear algebra must not run for nonfinite inputs")

    monkeypatch.setattr(torch, "matmul", reject_linear_algebra)
    monkeypatch.setattr(torch, "mm", reject_linear_algebra)
    monkeypatch.setattr(torch.Tensor, "__matmul__", reject_linear_algebra)
    monkeypatch.setattr(torch.linalg, "eigh", reject_linear_algebra)
    monkeypatch.setattr(torch.linalg, "eigvalsh", reject_linear_algebra)

    with pytest.raises(ValueError, match=message):
        lora_delta_diagnostics(a, b, alpha=alpha, rank=2)


@pytest.mark.parametrize(
    ("alpha", "a_scale", "b_scale"),
    [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)],
)
def test_lora_delta_zero_inputs_have_zero_diagnostics(
    alpha: float, a_scale: float, b_scale: float
) -> None:
    a = a_scale * torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    b = b_scale * torch.tensor([[2.0, 0.0], [0.5, -3.0]])

    result = lora_delta_diagnostics(a, b, alpha=alpha, rank=2)

    assert result == LoraDeltaDiagnostics(0.0, 0, 0.0)


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

    def reject_hidden_covariance(
        left: torch.Tensor, right: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        if (
            left.ndim == 2
            and right.ndim == 2
            and tuple(left.shape) == (hidden.shape[-1], hidden.shape[0])
            and tuple(right.shape) == (hidden.shape[0], hidden.shape[-1])
        ):
            raise AssertionError("hidden-dimension covariance must not be materialized")
        return original_matmul(left, right, *args, **kwargs)

    original_matmul = torch.matmul
    monkeypatch.setattr(torch, "matmul", reject_hidden_covariance)
    monkeypatch.setattr(torch, "mm", reject_hidden_covariance)
    monkeypatch.setattr(torch.Tensor, "__matmul__", reject_hidden_covariance)

    result = residual_diagnostics(hidden, max_components=4)

    assert result.covariance_trace == pytest.approx(expected_trace)
    assert 1 <= result.effective_rank <= 4
    assert 0 <= result.anisotropy <= 1


def test_residual_diagnostics_matches_exact_spectrum_when_fully_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden = torch.tensor(
        [[1.0, 0.0, 2.0], [2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [3.0, 1.0, 4.0], [1.0, 3.0, 2.0]]
    )
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square() / (hidden.shape[0] - 1)
    expected_trace = float(eigenvalues.sum().item())
    expected_rank = int(
        (eigenvalues > eigenvalues.max() * 1e-6).sum().item()
    )
    expected_anisotropy = float((eigenvalues.max() / eigenvalues.sum()).item())

    def reject_bounded_path(*args, **kwargs):
        raise AssertionError("full retained rank must use the exact singular-value path")

    monkeypatch.setattr(torch, "pca_lowrank", reject_bounded_path)

    result = residual_diagnostics(hidden, max_components=3)

    assert result.covariance_trace == pytest.approx(expected_trace)
    assert result.effective_rank == expected_rank
    assert result.anisotropy == pytest.approx(expected_anisotropy)


def test_residual_diagnostics_bounded_spectrum_is_reproducible() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(12, 64)
    torch.manual_seed(123)
    first = residual_diagnostics(hidden, max_components=2)
    torch.manual_seed(456)
    second = residual_diagnostics(hidden, max_components=2)

    assert first == second


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_residual_diagnostics_rejects_nonfinite_hidden_states(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    hidden = torch.ones(4, 32)
    hidden[0, 0] = value

    def reject_pca(*args, **kwargs):
        raise AssertionError("PCA must not run for nonfinite hidden states")

    monkeypatch.setattr(torch, "pca_lowrank", reject_pca)

    with pytest.raises(ValueError, match="hidden_states must contain only finite values"):
        residual_diagnostics(hidden, max_components=2)


def test_residual_diagnostics_rejects_nonpositive_component_limit() -> None:
    with pytest.raises(ValueError, match="max_components must be positive"):
        residual_diagnostics(torch.ones(2, 3), max_components=0)
