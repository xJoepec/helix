"""Device-agnostic structural diagnostics for Qwen-style tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LoraDeltaDiagnostics:
    frobenius_norm: float
    effective_rank: int
    spectral_entropy: float


@dataclass(frozen=True)
class ResidualDiagnostics:
    covariance_trace: float
    effective_rank: int
    anisotropy: float


def _spectrum(values: torch.Tensor) -> tuple[int, float]:
    singular = torch.linalg.svdvals(values.float())
    if singular.numel() == 0 or float(singular.max()) == 0:
        return 0, 0.0
    rank = int((singular > singular.max() * 1e-6).sum().item())
    probs = singular / singular.sum()
    entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum()).item())
    return rank, entropy


@torch.no_grad()
def lora_delta_diagnostics(lora_a, lora_b, *, alpha: float, rank: int) -> LoraDeltaDiagnostics:
    if rank <= 0:
        raise ValueError("rank must be positive")
    delta = (alpha / rank) * (lora_b.float() @ lora_a.float())
    effective_rank, entropy = _spectrum(delta)
    return LoraDeltaDiagnostics(float(torch.linalg.norm(delta).item()), effective_rank, entropy)


@torch.no_grad()
def residual_diagnostics(hidden_states) -> ResidualDiagnostics:
    matrix = hidden_states.float().reshape(-1, hidden_states.shape[-1])
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(matrix.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    trace = float(eigenvalues.sum().item())
    rank = int((eigenvalues > eigenvalues.max().clamp_min(1e-12) * 1e-6).sum().item())
    anisotropy = float((eigenvalues.max() / eigenvalues.sum().clamp_min(1e-12)).item())
    return ResidualDiagnostics(trace, rank, anisotropy)
