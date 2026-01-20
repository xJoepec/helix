"""Vision Transformer (ViT) adapter for AF partition extraction.

This module provides specialized support for extracting AF partitions from
Vision Transformer architectures, treating attention mechanisms as gauge
field interactions between token representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
except ImportError:
    torch = None
    nn = None
    F = None


@dataclass
class AttentionPartition:
    """AF partition extracted from a single attention head."""

    head_idx: int
    layer_idx: int
    gate_patterns: np.ndarray  # (N, seq_len) binary attention patterns
    attention_weights: np.ndarray  # (N, seq_len, seq_len) attention matrices
    gauge_connection: np.ndarray  # Approximated gauge field components
    partition_cells: List[np.ndarray]  # Grouped tokens with similar patterns
    masses: np.ndarray  # Token mass distribution


@dataclass
class ViTPartitionExtraction:
    """Complete AF partition extraction from a Vision Transformer."""

    layer_partitions: List[List[AttentionPartition]]  # [layer][head]
    mlp_partitions: List[np.ndarray]  # Standard ReLU partitions from MLP blocks
    B_list: List[np.ndarray]  # Incidence matrices (combined attention + MLP)
    tau_list: List[np.ndarray]  # Mass vectors
    token_trajectories: np.ndarray  # (N, seq_len, layers, dim) token evolution
    gauge_field_strength: List[np.ndarray]  # Curvature tensors per layer


class ViTAdapter:
    """Adapter for extracting AF partitions from Vision Transformer models.

    This adapter treats ViT components as follows:
    - Attention heads: Gauge field interactions with Q/K/V as connection components
    - MLP blocks: Standard ReLU partitions
    - Layer interactions: Higher-order gauge field dynamics
    """

    def __init__(
        self,
        model: Any,
        *,
        extract_attention_patterns: bool = True,
        extract_mlp_patterns: bool = True,
        compute_gauge_fields: bool = True,
        attention_threshold: float = 0.1,
        max_tokens: int = 1000
    ):
        """Initialize ViT adapter.

        Parameters
        ----------
        model : Any
            Vision Transformer model (transformers library or similar)
        extract_attention_patterns : bool
            Whether to extract attention-based partitions
        extract_mlp_patterns : bool
            Whether to extract MLP-based partitions
        compute_gauge_fields : bool
            Whether to compute gauge field approximations
        attention_threshold : float
            Threshold for binarizing attention weights
        max_tokens : int
            Maximum number of tokens to process (for memory efficiency)
        """
        if torch is None:
            raise RuntimeError("PyTorch required for ViT adapter")

        self.model = model
        self.extract_attention = extract_attention_patterns
        self.extract_mlp = extract_mlp_patterns
        self.compute_gauge = compute_gauge_fields
        self.attention_threshold = attention_threshold
        self.max_tokens = max_tokens

        # Identify ViT components
        self._identify_vit_components()

    def _identify_vit_components(self) -> None:
        """Identify attention and MLP components in the model."""
        self.attention_layers = []
        self.mlp_layers = []

        for name, module in self.model.named_modules():
            # Look for attention components
            if any(pattern in name.lower() for pattern in ['attention', 'attn', 'self_attn']):
                self.attention_layers.append((name, module))

            # Look for MLP/feed-forward components
            if any(pattern in name.lower() for pattern in ['mlp', 'ffn', 'feed_forward']):
                # Find ReLU activations within MLP
                for sub_name, sub_module in module.named_modules():
                    if isinstance(sub_module, nn.ReLU):
                        self.mlp_layers.append((f"{name}.{sub_name}", sub_module))

    def extract_partitions(
        self,
        X: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> ViTPartitionExtraction:
        """Extract AF partitions from ViT model.

        Parameters
        ----------
        X : np.ndarray
            Input data (N, C, H, W) for vision tasks or (N, seq_len, dim) for sequences
        sample_weights : Optional[np.ndarray]
            Optional sample weights

        Returns
        -------
        ViTPartitionExtraction
            Complete partition extraction with attention and MLP components
        """
        self.model.eval()
        X_t = torch.from_numpy(X.astype(np.float32))

        # Limit number of samples for memory efficiency
        if len(X_t) > self.max_tokens:
            indices = np.random.choice(len(X_t), self.max_tokens, replace=False)
            X_t = X_t[indices]
            if sample_weights is not None:
                sample_weights = sample_weights[indices]

        # Storage for extracted components
        layer_partitions = []
        mlp_partitions = []
        token_trajectories = []
        gauge_field_strength = []

        # Hook storage
        attention_outputs = {}
        mlp_outputs = {}

        def attention_hook(name):
            def hook(module, input, output):
                # Store attention weights and patterns
                if hasattr(output, 'detach'):
                    attention_outputs[name] = output.detach().cpu().numpy()
                elif isinstance(output, tuple) and len(output) > 1:
                    # Some models return (output, attention_weights)
                    attention_outputs[name] = output[1].detach().cpu().numpy()
            return hook

        def mlp_hook(name):
            def hook(module, input, output):
                if hasattr(input[0], 'detach'):
                    pre_activation = input[0].detach().cpu().numpy()
                    mlp_outputs[name] = (pre_activation > 0).astype(np.uint8)
            return hook

        # Register hooks
        attention_handles = []
        mlp_handles = []

        if self.extract_attention:
            for name, module in self.attention_layers:
                handle = module.register_forward_hook(attention_hook(name))
                attention_handles.append(handle)

        if self.extract_mlp:
            for name, module in self.mlp_layers:
                handle = module.register_forward_hook(mlp_hook(name))
                mlp_handles.append(handle)

        # Forward pass to collect data
        with torch.no_grad():
            _ = self.model(X_t)

        # Process attention partitions
        if self.extract_attention:
            layer_partitions = self._process_attention_outputs(
                attention_outputs, X_t, sample_weights
            )

            if self.compute_gauge:
                gauge_field_strength = self._compute_gauge_fields(attention_outputs)

        # Process MLP partitions
        if self.extract_mlp:
            mlp_partitions = self._process_mlp_outputs(mlp_outputs, sample_weights)

        # Combine into unified incidence matrices
        B_list, tau_list = self._combine_partitions(
            layer_partitions, mlp_partitions, sample_weights
        )

        # Clean up hooks
        for handle in attention_handles + mlp_handles:
            handle.remove()

        return ViTPartitionExtraction(
            layer_partitions=layer_partitions,
            mlp_partitions=mlp_partitions,
            B_list=B_list,
            tau_list=tau_list,
            token_trajectories=np.array(token_trajectories) if token_trajectories else np.array([]),
            gauge_field_strength=gauge_field_strength
        )

    def _process_attention_outputs(
        self,
        attention_outputs: Dict[str, np.ndarray],
        X_t: torch.Tensor,
        sample_weights: Optional[np.ndarray]
    ) -> List[List[AttentionPartition]]:
        """Process attention outputs into AF partitions."""
        layer_partitions = []

        for layer_name, attention_data in attention_outputs.items():
            if attention_data.ndim != 4:  # Expected: (batch, heads, seq_len, seq_len)
                continue

            batch_size, num_heads, seq_len, _ = attention_data.shape
            head_partitions = []

            for head_idx in range(num_heads):
                # Extract attention weights for this head
                head_attention = attention_data[:, head_idx, :, :]  # (batch, seq_len, seq_len)

                # Binarize attention patterns
                attention_patterns = (head_attention > self.attention_threshold).astype(np.uint8)

                # Create partition based on attention patterns
                partition = self._create_attention_partition(
                    head_attention, attention_patterns, head_idx, layer_name, sample_weights
                )

                head_partitions.append(partition)

            layer_partitions.append(head_partitions)

        return layer_partitions

    def _create_attention_partition(
        self,
        attention_weights: np.ndarray,
        attention_patterns: np.ndarray,
        head_idx: int,
        layer_name: str,
        sample_weights: Optional[np.ndarray]
    ) -> AttentionPartition:
        """Create AF partition from attention patterns."""
        batch_size, seq_len, _ = attention_weights.shape

        # Flatten patterns for clustering
        flattened_patterns = attention_patterns.reshape(batch_size, -1)

        # Group samples with similar attention patterns
        unique_patterns, inverse_indices = np.unique(
            flattened_patterns, axis=0, return_inverse=True
        )

        # Create partition cells
        partition_cells = []
        for pattern_idx in range(len(unique_patterns)):
            cell_indices = np.where(inverse_indices == pattern_idx)[0]
            partition_cells.append(cell_indices)

        # Compute masses
        if sample_weights is None:
            weights = np.ones(batch_size) / batch_size
        else:
            weights = sample_weights / sample_weights.sum()

        masses = np.array([weights[cell].sum() for cell in partition_cells])

        # Approximate gauge connection (simplified)
        gauge_connection = self._approximate_gauge_connection(attention_weights)

        # Parse layer index from name
        layer_idx = self._extract_layer_index(layer_name)

        return AttentionPartition(
            head_idx=head_idx,
            layer_idx=layer_idx,
            gate_patterns=attention_patterns,
            attention_weights=attention_weights,
            gauge_connection=gauge_connection,
            partition_cells=partition_cells,
            masses=masses
        )

    def _process_mlp_outputs(
        self,
        mlp_outputs: Dict[str, np.ndarray],
        sample_weights: Optional[np.ndarray]
    ) -> List[np.ndarray]:
        """Process MLP outputs into standard ReLU partitions."""
        mlp_partitions = []

        for layer_name, gate_patterns in mlp_outputs.items():
            # gate_patterns is already binary (N, seq_len, hidden_dim)
            if gate_patterns.ndim == 3:
                # Flatten spatial dimensions for partition creation
                batch_size, seq_len, hidden_dim = gate_patterns.shape
                flattened = gate_patterns.reshape(batch_size, -1)

                # Create partition signatures
                signatures = [tuple(row) for row in flattened]
                unique_sigs, inverse = np.unique(signatures, return_inverse=True)

                # Group into cells
                cells = []
                for sig_idx in range(len(unique_sigs)):
                    cell_indices = np.where(inverse == sig_idx)[0]
                    cells.append(cell_indices)

                mlp_partitions.append(np.array(cells, dtype=object))

        return mlp_partitions

    def _combine_partitions(
        self,
        layer_partitions: List[List[AttentionPartition]],
        mlp_partitions: List[np.ndarray],
        sample_weights: Optional[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Combine attention and MLP partitions into unified incidence matrices."""
        B_list = []
        tau_list = []

        # Process each layer
        max_layers = max(len(layer_partitions), len(mlp_partitions))

        for layer_idx in range(max_layers):
            # Combine all partitions for this layer
            all_cells = []

            # Add attention partitions
            if layer_idx < len(layer_partitions):
                for head_partition in layer_partitions[layer_idx]:
                    all_cells.extend(head_partition.partition_cells)

            # Add MLP partitions
            if layer_idx < len(mlp_partitions):
                all_cells.extend(mlp_partitions[layer_idx])

            if not all_cells:
                continue

            # Compute masses for combined partition
            if sample_weights is None:
                total_samples = len(all_cells[0]) if all_cells else 1
                weights = np.ones(total_samples) / total_samples
            else:
                weights = sample_weights / sample_weights.sum()

            masses = np.array([weights[cell].sum() for cell in all_cells if len(cell) > 0])

            # Create incidence matrix (simplified - assumes refinement structure)
            if layer_idx == 0:
                # First layer connects to single root
                B = np.ones((1, len(all_cells)), dtype=np.int32)
            else:
                # Create connections based on partition overlap
                prev_cells = len(tau_list[-1]) if tau_list else 1
                B = np.zeros((prev_cells, len(all_cells)), dtype=np.int32)

                # Simple parent-child assignment
                for child_idx, child_cell in enumerate(all_cells):
                    if len(child_cell) > 0:
                        parent_idx = min(child_idx, prev_cells - 1)
                        B[parent_idx, child_idx] = 1

            B_list.append(B)
            tau_list.append(masses)

        return B_list, tau_list

    def _approximate_gauge_connection(self, attention_weights: np.ndarray) -> np.ndarray:
        """Approximate gauge connection from attention weights.

        This treats attention as a discrete gauge field where:
        - Q, K, V projections are components of the connection A_μ
        - Attention weights represent parallel transport
        """
        batch_size, seq_len, _ = attention_weights.shape

        # Compute "covariant derivative" approximation
        # ∇_μ = ∂_μ + A_μ where A_μ is derived from attention
        connection = np.zeros((batch_size, seq_len, seq_len))

        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    # Connection strength proportional to attention
                    connection[:, i, j] = attention_weights[:, i, j]

        # Normalize to make it closer to a proper connection
        connection = connection / (np.linalg.norm(connection, axis=(1, 2), keepdims=True) + 1e-8)

        return connection

    def _compute_gauge_fields(self, attention_outputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Compute gauge field strength tensors (curvature) from attention patterns."""
        gauge_fields = []

        for layer_name, attention_data in attention_outputs.items():
            if attention_data.ndim != 4:
                continue

            batch_size, num_heads, seq_len, _ = attention_data.shape

            # Compute field strength tensor F_μν = ∂_μ A_ν - ∂_ν A_μ + [A_μ, A_ν]
            field_strength = np.zeros((batch_size, num_heads, seq_len, seq_len))

            for head in range(num_heads):
                A = attention_data[:, head, :, :]  # Connection for this head

                # Discrete derivatives and commutators
                for i in range(seq_len):
                    for j in range(seq_len):
                        if i != j:
                            # Simplified field strength (anti-symmetric part)
                            field_strength[:, head, i, j] = A[:, i, j] - A[:, j, i]

            gauge_fields.append(field_strength)

        return gauge_fields

    def _extract_layer_index(self, layer_name: str) -> int:
        """Extract layer index from layer name."""
        import re
        match = re.search(r'(\d+)', layer_name)
        return int(match.group(1)) if match else 0


def extract_vit_partitions(
    model: Any,
    X: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    **adapter_kwargs
) -> ViTPartitionExtraction:
    """Convenience function for extracting ViT partitions.

    Parameters
    ----------
    model : Any
        Vision Transformer model
    X : np.ndarray
        Input data
    sample_weights : Optional[np.ndarray]
        Optional sample weights
    **adapter_kwargs
        Additional arguments for ViTAdapter

    Returns
    -------
    ViTPartitionExtraction
        Complete ViT partition extraction
    """
    adapter = ViTAdapter(model, **adapter_kwargs)
    return adapter.extract_partitions(X, sample_weights)


# Physics interpretation utilities
def interpret_attention_as_gauge_field(partition: AttentionPartition) -> Dict[str, Any]:
    """Interpret attention partition in terms of gauge field theory."""
    gauge_conn = partition.gauge_connection

    # Compute gauge field properties
    field_strength = np.zeros_like(gauge_conn)
    batch_size, seq_len, _ = gauge_conn.shape

    for i in range(seq_len):
        for j in range(seq_len):
            # Field strength F_ij = A_ij - A_ji (simplified)
            field_strength[:, i, j] = gauge_conn[:, i, j] - gauge_conn[:, j, i]

    # Gauge invariant quantities
    wilson_loops = []
    for b in range(batch_size):
        # Compute simple Wilson loop around tokens 0→1→2→0
        if seq_len >= 3:
            loop = (
                gauge_conn[b, 0, 1] *
                gauge_conn[b, 1, 2] *
                gauge_conn[b, 2, 0]
            )
            wilson_loops.append(loop)

    return {
        "field_strength": field_strength,
        "wilson_loops": np.array(wilson_loops) if wilson_loops else np.array([]),
        "gauge_coupling": np.mean(np.abs(gauge_conn)),
        "topological_charge": np.sum(field_strength, axis=(1, 2)),  # Simplified
    }


__all__ = [
    "ViTAdapter",
    "AttentionPartition",
    "ViTPartitionExtraction",
    "extract_vit_partitions",
    "interpret_attention_as_gauge_field",
]