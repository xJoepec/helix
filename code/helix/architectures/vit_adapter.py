"""Vision Transformer attention diagnostics and AF-style partitions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _as_float_tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return a detached floating-point tensor without forcing a device move."""
    if isinstance(value, torch.Tensor):
        return value.detach().float()
    return torch.as_tensor(value, dtype=torch.float32)


@dataclass
class AttentionPartition:
    """Attention and gauge diagnostics for one head in one ViT layer."""

    layer_idx: int
    head_idx: int
    attention_weights: torch.Tensor
    gauge_connection: torch.Tensor
    sequence_length: int
    embed_dim: int

    def compute_attention_entropy(self) -> float:
        """Return mean categorical entropy over attention query rows."""
        probabilities = _as_float_tensor(self.attention_weights).clamp_min(0)
        normalizer = probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probabilities = probabilities / normalizer
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        return float(entropy.mean().item()) if entropy.numel() else 0.0

    def compute_gauge_field_strength(self) -> float:
        """Return RMS curvature of the antisymmetric gauge connection."""
        connection = _as_float_tensor(self.gauge_connection)
        if connection.ndim < 2 or connection.numel() == 0:
            return 0.0
        curvature = connection - connection.transpose(-1, -2)
        return float(curvature.square().mean().sqrt().item())

    def to_af_partition_metrics(self) -> dict[str, float | int]:
        """Convert this partition to scalar AF diagnostics."""
        return {
            "attention_entropy": self.compute_attention_entropy(),
            "gauge_field_strength": self.compute_gauge_field_strength(),
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim,
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
        }


@dataclass
class ViTPartitionExtraction:
    """Collection of per-head attention partitions from a ViT."""

    partitions: list[AttentionPartition]
    total_layers: int
    total_heads: int
    global_embed_dim: int

    def analyze_layer_wise_attention(self) -> dict[int, dict[str, float | int]]:
        """Aggregate entropy and field strength independently for each layer."""
        analysis: dict[int, dict[str, float | int]] = {}
        for layer_idx in sorted({partition.layer_idx for partition in self.partitions}):
            layer = [p for p in self.partitions if p.layer_idx == layer_idx]
            analysis[layer_idx] = {
                "avg_entropy": float(
                    np.mean([partition.compute_attention_entropy() for partition in layer])
                ),
                "avg_field_strength": float(
                    np.mean([partition.compute_gauge_field_strength() for partition in layer])
                ),
                "num_heads": len(layer),
            }
        return analysis

    def compute_global_metrics(self) -> dict[str, float]:
        """Aggregate scalar diagnostics across every captured head."""
        if not self.partitions:
            return {
                "total_entropy": 0.0,
                "avg_field_strength": 0.0,
                "attention_diversity": 0.0,
            }

        entropies = np.asarray(
            [partition.compute_attention_entropy() for partition in self.partitions],
            dtype=np.float64,
        )
        field_strengths = np.asarray(
            [partition.compute_gauge_field_strength() for partition in self.partitions],
            dtype=np.float64,
        )
        return {
            "total_entropy": float(entropies.sum()),
            "avg_field_strength": float(field_strengths.mean()),
            "attention_diversity": float(entropies.std()),
        }


class ViTAdapter:
    """Capture attention tensors from ViT-like PyTorch modules."""

    def __init__(
        self,
        model: nn.Module,
        *,
        target_layers: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")

        self.model = model
        self.target_layers = None if target_layers is None else set(target_layers)
        self.attention_modules = self._find_attention_modules()
        self.attention_weights: dict[int, torch.Tensor | None] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _layer_index(name: str, fallback: int) -> int:
        match = re.search(r"(?:^|\.)(\d+)(?:\.|$)", name)
        return int(match.group(1)) if match else fallback

    @staticmethod
    def _is_attention_module(name: str, module: nn.Module) -> bool:
        lowered = name.lower()
        named_like_attention = any(
            token in lowered for token in ("attention", "attn", "self_attn")
        )
        has_heads = hasattr(module, "num_heads") or hasattr(
            module, "num_attention_heads"
        )
        return named_like_attention and has_heads

    def _find_attention_modules(self) -> list[tuple[int, nn.Module]]:
        modules: list[tuple[int, nn.Module]] = []
        named_modules = (item for item in self.model.named_modules() if item[0])
        for fallback, (name, module) in enumerate(named_modules):
            if not self._is_attention_module(name, module):
                continue
            layer_idx = self._layer_index(name, fallback)
            if self.target_layers is None or layer_idx in self.target_layers:
                modules.append((layer_idx, module))
        return modules

    @staticmethod
    def _attention_from_output(
        module: nn.Module, output: Any
    ) -> torch.Tensor | None:
        stored = getattr(module, "attention_weights", None)
        if isinstance(stored, torch.Tensor) and stored.ndim == 4:
            return stored

        candidates = output if isinstance(output, (tuple, list)) else (output,)
        for candidate in candidates:
            if (
                isinstance(candidate, torch.Tensor)
                and candidate.ndim == 4
                and candidate.shape[-1] == candidate.shape[-2]
            ):
                return candidate
        return None

    def register_attention_hooks(self) -> None:
        """Register idempotent hooks and initialize capture slots."""
        if self._hook_handles:
            return

        self.attention_weights = {
            layer_idx: None for layer_idx, _ in self.attention_modules
        }
        for layer_idx, module in self.attention_modules:

            def capture(
                hooked_module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                captured_layer: int = layer_idx,
            ) -> None:
                attention = self._attention_from_output(hooked_module, output)
                if attention is not None:
                    self.attention_weights[captured_layer] = attention.detach()

            self._hook_handles.append(module.register_forward_hook(capture))

    def remove_attention_hooks(self) -> None:
        """Remove every hook owned by this adapter."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    @staticmethod
    def _gauge_connection(attention: torch.Tensor) -> torch.Tensor:
        connection = attention.mean(dim=0)
        norm = torch.linalg.vector_norm(connection).clamp_min(1e-12)
        return connection / norm

    def _embed_dim_for(self, module: nn.Module) -> int:
        for owner in (module, self.model):
            for attribute in ("embed_dim", "hidden_size", "all_head_size"):
                value = getattr(owner, attribute, None)
                if isinstance(value, int):
                    return value
        config = getattr(self.model, "config", None)
        return int(getattr(config, "hidden_size", 0))

    def extract_attention_partitions(self) -> ViTPartitionExtraction:
        """Build flat per-head partitions from the latest hook captures."""
        partitions: list[AttentionPartition] = []
        heads_per_layer: list[int] = []
        modules_by_layer = dict(self.attention_modules)

        for layer_idx, attention in sorted(self.attention_weights.items()):
            if attention is None:
                continue
            if attention.ndim != 4 or attention.shape[-1] != attention.shape[-2]:
                continue

            num_heads = int(attention.shape[1])
            heads_per_layer.append(num_heads)
            module = modules_by_layer[layer_idx]
            embed_dim = self._embed_dim_for(module)
            for head_idx in range(num_heads):
                head_attention = attention[:, head_idx : head_idx + 1]
                partitions.append(
                    AttentionPartition(
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        attention_weights=head_attention,
                        gauge_connection=self._gauge_connection(head_attention),
                        sequence_length=int(attention.shape[-1]),
                        embed_dim=embed_dim,
                    )
                )

        layers = {partition.layer_idx for partition in partitions}
        return ViTPartitionExtraction(
            partitions=partitions,
            total_layers=len(layers),
            total_heads=max(heads_per_layer, default=0),
            global_embed_dim=self._embed_dim_for(self.model),
        )

    def extract_partitions(
        self,
        inputs: torch.Tensor | np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> ViTPartitionExtraction:
        """Run the model once and extract its attention partitions."""
        del sample_weights  # Legacy argument; attention diagnostics are per head.
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        tensor = _as_float_tensor(inputs).to(model_device)
        was_training = self.model.training
        self.register_attention_hooks()
        try:
            self.model.eval()
            with torch.no_grad():
                self.model(tensor)
            return self.extract_attention_partitions()
        finally:
            self.remove_attention_hooks()
            self.model.train(was_training)


def extract_vit_partitions(
    model: nn.Module,
    inputs: torch.Tensor | np.ndarray,
    sample_weights: np.ndarray | None = None,
    **adapter_kwargs: Any,
) -> ViTPartitionExtraction:
    """Extract ViT attention partitions in one call."""
    adapter = ViTAdapter(model, **adapter_kwargs)
    return adapter.extract_partitions(inputs, sample_weights)


def interpret_attention_as_gauge_field(
    partition: AttentionPartition,
) -> dict[str, Any]:
    """Return simple gauge-theoretic diagnostics for an attention partition."""
    connection = _as_float_tensor(partition.gauge_connection)
    field_strength = connection - connection.transpose(-1, -2)
    sequence_length = connection.shape[-1] if connection.ndim >= 2 else 0
    if sequence_length >= 3:
        wilson_loops = (
            connection[..., 0, 1]
            * connection[..., 1, 2]
            * connection[..., 2, 0]
        )
    else:
        wilson_loops = torch.empty(0, device=connection.device)
    return {
        "field_strength": field_strength.cpu().numpy(),
        "wilson_loops": wilson_loops.cpu().numpy(),
        "gauge_coupling": float(connection.abs().mean().item()),
        "topological_charge": field_strength.sum(dim=(-1, -2)).cpu().numpy(),
    }


__all__ = [
    "ViTAdapter",
    "AttentionPartition",
    "ViTPartitionExtraction",
    "extract_vit_partitions",
    "interpret_attention_as_gauge_field",
]
