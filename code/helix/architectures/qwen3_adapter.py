"""Device-agnostic structural diagnostics for Qwen-style tensors."""

from __future__ import annotations

import math
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


def _spectrum_from_singular_values(singular: torch.Tensor) -> tuple[int, float]:
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
    if not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("LoRA factors must be two-dimensional")
    if lora_a.shape[0] != rank:
        raise ValueError("rank must match lora_a.shape[0]")
    if lora_b.shape[1] != rank:
        raise ValueError("lora_b.shape[1] must match rank")
    if lora_a.device != lora_b.device:
        raise ValueError("LoRA factors must be on the same device")
    if not torch.isfinite(lora_a).all().item():
        raise ValueError("lora_a must contain only finite values")
    if not torch.isfinite(lora_b).all().item():
        raise ValueError("lora_b must contain only finite values")

    a = lora_a.float()
    b = lora_b.float()
    a_gram = a @ a.T
    b_gram = b.T @ b
    a_eigenvalues, a_eigenvectors = torch.linalg.eigh(a_gram)
    a_gram_sqrt = (a_eigenvectors * a_eigenvalues.clamp_min(0).sqrt()) @ a_eigenvectors.T
    squared_singular = torch.linalg.eigvalsh(a_gram_sqrt @ b_gram @ a_gram_sqrt).clamp_min(0)
    singular = squared_singular.sqrt() * (abs(float(alpha)) / rank)
    effective_rank, entropy = _spectrum_from_singular_values(singular)
    return LoraDeltaDiagnostics(
        float(torch.linalg.vector_norm(singular).item()), effective_rank, entropy
    )


@torch.no_grad()
def residual_diagnostics(hidden_states, *, max_components: int = 32) -> ResidualDiagnostics:
    if max_components <= 0:
        raise ValueError("max_components must be positive")
    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must include a hidden dimension")
    if not torch.isfinite(hidden_states).all().item():
        raise ValueError("hidden_states must contain only finite values")
    matrix = hidden_states.float().reshape(-1, hidden_states.shape[-1])
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("hidden_states must contain at least one feature vector")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    denominator = max(matrix.shape[0] - 1, 1)
    trace = float((centered.square().sum() / denominator).item())
    if trace == 0:
        return ResidualDiagnostics(0.0, 0, 0.0)

    available_components = min(matrix.shape[0] - 1, matrix.shape[1])
    if max_components >= available_components:
        singular = torch.linalg.svdvals(centered)[:available_components]
    else:
        rng_devices = [matrix.device.index] if matrix.device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(0)
            _, singular, _ = torch.pca_lowrank(
                centered, q=max_components, center=False
            )
    eigenvalues = singular.square().div(denominator).clamp_min(0)
    rank = int((eigenvalues > eigenvalues.max().clamp_min(1e-12) * 1e-6).sum().item())
    anisotropy = float((eigenvalues.max() / max(trace, 1e-12)).item())
    return ResidualDiagnostics(trace, rank, anisotropy)
