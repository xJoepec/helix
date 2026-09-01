from __future__ import annotations

import os
import sys
import unittest
from typing import Optional

import torch
import torch.nn as nn

# Ensure the 'code' directory (package root for 'helix') is on sys.path
PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

"""Tests for Vision Transformer adapter for AF partition analysis.

Imports from the package are done inside test bodies to avoid E402 with sys.path edits.
"""


class MockAttentionModule(nn.Module):
    """Mock attention module for testing purposes."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Store attention weights for testing
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        # Store for testing
        self.attention_weights = attn.detach()

        # Apply attention
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(out)


class MockViTBlock(nn.Module):
    """Mock ViT transformer block."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MockAttentionModule(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MockViT(nn.Module):
    """Mock Vision Transformer for testing."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Patch embedding (simplified)
        self.patch_embed = nn.Linear(16, embed_dim)  # Assume 4x4 patches

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MockViTBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Classification head
        self.head = nn.Linear(embed_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified forward pass
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = x.mean(dim=1)
        return self.head(x)


class TestAttentionPartition(unittest.TestCase):
    def test_attention_partition_creation(self):
        from helix.architectures.vit_adapter import AttentionPartition

        # Create test data
        attention_weights = torch.randn(2, 4, 8, 8)  # batch, heads, seq, seq
        gauge_connection = torch.randn(4, 8, 8)  # heads, seq, seq

        partition = AttentionPartition(
            layer_idx=1,
            head_idx=2,
            attention_weights=attention_weights,
            gauge_connection=gauge_connection,
            sequence_length=8,
            embed_dim=64
        )

        self.assertEqual(partition.layer_idx, 1)
        self.assertEqual(partition.head_idx, 2)
        self.assertEqual(partition.sequence_length, 8)
        self.assertEqual(partition.embed_dim, 64)
        self.assertEqual(partition.attention_weights.shape, (2, 4, 8, 8))

    def test_attention_partition_methods(self):
        from helix.architectures.vit_adapter import AttentionPartition

        # Create normalized attention weights (should sum to 1 along last dim)
        attn = torch.softmax(torch.randn(1, 2, 4, 4), dim=-1)
        gauge = torch.randn(2, 4, 4)

        partition = AttentionPartition(
            layer_idx=0,
            head_idx=0,
            attention_weights=attn,
            gauge_connection=gauge,
            sequence_length=4,
            embed_dim=32
        )

        # Test attention entropy computation
        entropy = partition.compute_attention_entropy()
        self.assertIsInstance(entropy, float)
        self.assertGreater(entropy, 0.0)

        # Test gauge field strength
        field_strength = partition.compute_gauge_field_strength()
        self.assertIsInstance(field_strength, float)
        self.assertGreaterEqual(field_strength, 0.0)

        # Test AF partition metrics
        af_metrics = partition.to_af_partition_metrics()
        self.assertIn('attention_entropy', af_metrics)
        self.assertIn('gauge_field_strength', af_metrics)
        self.assertIn('sequence_length', af_metrics)


class TestViTPartitionExtraction(unittest.TestCase):
    def test_vit_partition_extraction_creation(self):
        from helix.architectures.vit_adapter import AttentionPartition, ViTPartitionExtraction

        # Create test partitions
        partitions = []
        for layer in range(2):
            for head in range(2):
                attn = torch.softmax(torch.randn(1, 1, 4, 4), dim=-1)
                gauge = torch.randn(1, 4, 4)
                partition = AttentionPartition(
                    layer_idx=layer,
                    head_idx=head,
                    attention_weights=attn,
                    gauge_connection=gauge,
                    sequence_length=4,
                    embed_dim=32
                )
                partitions.append(partition)

        extraction = ViTPartitionExtraction(
            partitions=partitions,
            total_layers=2,
            total_heads=2,
            global_embed_dim=32
        )

        self.assertEqual(len(extraction.partitions), 4)  # 2 layers × 2 heads
        self.assertEqual(extraction.total_layers, 2)
        self.assertEqual(extraction.total_heads, 2)

    def test_vit_partition_extraction_methods(self):
        from helix.architectures.vit_adapter import AttentionPartition, ViTPartitionExtraction

        # Create extraction with test data
        partitions = []
        attn = torch.softmax(torch.randn(1, 1, 3, 3), dim=-1)
        gauge = torch.randn(1, 3, 3)
        partition = AttentionPartition(
            layer_idx=0,
            head_idx=0,
            attention_weights=attn,
            gauge_connection=gauge,
            sequence_length=3,
            embed_dim=24
        )
        partitions.append(partition)

        extraction = ViTPartitionExtraction(
            partitions=partitions,
            total_layers=1,
            total_heads=1,
            global_embed_dim=24
        )

        # Test layer-wise analysis
        layer_analysis = extraction.analyze_layer_wise_attention()
        self.assertIn(0, layer_analysis)
        self.assertIn('avg_entropy', layer_analysis[0])
        self.assertIn('avg_field_strength', layer_analysis[0])

        # Test global metrics
        global_metrics = extraction.compute_global_metrics()
        self.assertIn('total_entropy', global_metrics)
        self.assertIn('avg_field_strength', global_metrics)
        self.assertIn('attention_diversity', global_metrics)


class TestViTAdapter(unittest.TestCase):
    def test_vit_adapter_creation(self):
        from helix.architectures.vit_adapter import ViTAdapter

        model = MockViT(embed_dim=32, num_heads=2, num_layers=2)
        adapter = ViTAdapter(model)

        self.assertEqual(adapter.model, model)
        self.assertEqual(len(adapter.attention_modules), 2)  # 2 layers

    def test_vit_adapter_hook_registration(self):
        from helix.architectures.vit_adapter import ViTAdapter

        model = MockViT(embed_dim=32, num_heads=2, num_layers=2)
        adapter = ViTAdapter(model)

        # Register hooks
        adapter.register_attention_hooks()

        # Check that hooks were registered
        self.assertGreater(len(adapter.attention_weights), 0)

        # Run forward pass to trigger hooks
        x = torch.randn(1, 8, 16)  # batch, seq_len, patch_dim
        with torch.no_grad():
            _ = model(x)

        # Check that attention weights were captured
        self.assertEqual(len(adapter.attention_weights), 2)  # 2 layers

    def test_vit_adapter_extraction(self):
        from helix.architectures.vit_adapter import ViTAdapter

        model = MockViT(embed_dim=32, num_heads=2, num_layers=2)
        adapter = ViTAdapter(model)

        # Register hooks and run forward pass
        adapter.register_attention_hooks()
        x = torch.randn(2, 6, 16)  # batch=2, seq_len=6

        with torch.no_grad():
            _ = model(x)

        # Extract partitions
        extraction = adapter.extract_attention_partitions()

        self.assertIsNotNone(extraction)
        self.assertEqual(extraction.total_layers, 2)
        self.assertEqual(extraction.total_heads, 2)
        self.assertEqual(len(extraction.partitions), 4)  # 2 layers × 2 heads


class TestExtractViTPartitions(unittest.TestCase):
    def test_extract_vit_partitions_function(self):
        from helix.architectures.vit_adapter import extract_vit_partitions

        model = MockViT(embed_dim=32, num_heads=2, num_layers=2)
        X = torch.randn(3, 5, 16)  # batch=3, seq_len=5

        # Extract partitions using the main function
        extraction = extract_vit_partitions(model, X)

        self.assertIsNotNone(extraction)
        self.assertEqual(extraction.total_layers, 2)
        self.assertEqual(extraction.total_heads, 2)
        self.assertGreater(len(extraction.partitions), 0)

    def test_extract_vit_partitions_with_target_layers(self):
        from helix.architectures.vit_adapter import extract_vit_partitions

        model = MockViT(embed_dim=32, num_heads=2, num_layers=3)
        X = torch.randn(2, 4, 16)

        # Extract only specific layers
        extraction = extract_vit_partitions(model, X, target_layers=[0, 2])

        # Should only have partitions from layers 0 and 2
        layer_indices = {p.layer_idx for p in extraction.partitions}
        self.assertEqual(layer_indices, {0, 2})

    def test_extract_vit_partitions_error_handling(self):
        from helix.architectures.vit_adapter import extract_vit_partitions

        # Test with incompatible model (no attention modules)
        simple_model = nn.Linear(10, 5)
        X = torch.randn(2, 10)

        # Should handle gracefully (might return empty extraction or raise informative error)
        try:
            extraction = extract_vit_partitions(simple_model, X)
            # If it succeeds, should have no partitions
            self.assertEqual(len(extraction.partitions), 0)
        except (ValueError, AttributeError):
            # Expected if model doesn't have the required structure
            pass


class TestGaugeFieldComputations(unittest.TestCase):
    def test_gauge_connection_approximation(self):
        from helix.architectures.vit_adapter import AttentionPartition

        # Create attention weights with clear structure
        seq_len = 4
        attn = torch.zeros(1, 1, seq_len, seq_len)

        # Create a "local" attention pattern (neighboring tokens)
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) <= 1:  # Attend to neighbors
                    attn[0, 0, i, j] = 1.0

        # Normalize
        attn = torch.softmax(attn, dim=-1)

        # Compute gauge connection (this is done internally)
        gauge = torch.randn(1, seq_len, seq_len)

        partition = AttentionPartition(
            layer_idx=0,
            head_idx=0,
            attention_weights=attn,
            gauge_connection=gauge,
            sequence_length=seq_len,
            embed_dim=32
        )

        # Test that gauge field strength is computed correctly
        field_strength = partition.compute_gauge_field_strength()
        self.assertIsInstance(field_strength, float)
        self.assertGreaterEqual(field_strength, 0.0)

    def test_attention_entropy_properties(self):
        from helix.architectures.vit_adapter import AttentionPartition

        seq_len = 3

        # Test uniform attention (maximum entropy)
        uniform_attn = torch.ones(1, 1, seq_len, seq_len) / seq_len
        gauge = torch.zeros(1, seq_len, seq_len)

        partition_uniform = AttentionPartition(
            layer_idx=0,
            head_idx=0,
            attention_weights=uniform_attn,
            gauge_connection=gauge,
            sequence_length=seq_len,
            embed_dim=24
        )

        # Test concentrated attention (minimum entropy)
        concentrated_attn = torch.zeros(1, 1, seq_len, seq_len)
        concentrated_attn[0, 0, :, 0] = 1.0  # All attention to first token

        partition_concentrated = AttentionPartition(
            layer_idx=0,
            head_idx=0,
            attention_weights=concentrated_attn,
            gauge_connection=gauge,
            sequence_length=seq_len,
            embed_dim=24
        )

        entropy_uniform = partition_uniform.compute_attention_entropy()
        entropy_concentrated = partition_concentrated.compute_attention_entropy()

        # Uniform should have higher entropy than concentrated
        self.assertGreater(entropy_uniform, entropy_concentrated)


class TestIntegrationTests(unittest.TestCase):
    def test_full_vit_analysis_pipeline(self):
        """Test the complete pipeline from model to AF metrics."""
        from helix.architectures.vit_adapter import extract_vit_partitions

        # Create a slightly larger model for more realistic testing
        model = MockViT(embed_dim=48, num_heads=3, num_layers=2)

        # Create input data
        batch_size, seq_len = 2, 8
        X = torch.randn(batch_size, seq_len, 16)

        # Extract attention partitions
        extraction = extract_vit_partitions(model, X)

        # Verify structure
        self.assertEqual(extraction.total_layers, 2)
        self.assertEqual(extraction.total_heads, 3)
        self.assertEqual(len(extraction.partitions), 6)  # 2 layers × 3 heads

        # Test layer-wise analysis
        layer_analysis = extraction.analyze_layer_wise_attention()
        self.assertEqual(len(layer_analysis), 2)  # 2 layers

        for layer_idx, analysis in layer_analysis.items():
            self.assertIn('avg_entropy', analysis)
            self.assertIn('avg_field_strength', analysis)
            self.assertIn('num_heads', analysis)
            self.assertEqual(analysis['num_heads'], 3)

        # Test global metrics
        global_metrics = extraction.compute_global_metrics()
        required_keys = ['total_entropy', 'avg_field_strength', 'attention_diversity']
        for key in required_keys:
            self.assertIn(key, global_metrics)
            self.assertIsInstance(global_metrics[key], (int, float))

    def test_different_sequence_lengths(self):
        """Test adapter with different sequence lengths."""
        from helix.architectures.vit_adapter import extract_vit_partitions

        model = MockViT(embed_dim=32, num_heads=2, num_layers=1)

        # Test with different sequence lengths
        for seq_len in [4, 8, 16]:
            X = torch.randn(1, seq_len, 16)
            extraction = extract_vit_partitions(model, X)

            # All partitions should have the correct sequence length
            for partition in extraction.partitions:
                self.assertEqual(partition.sequence_length, seq_len)

    def test_batch_size_handling(self):
        """Test that different batch sizes are handled correctly."""
        from helix.architectures.vit_adapter import extract_vit_partitions

        model = MockViT(embed_dim=32, num_heads=2, num_layers=1)
        seq_len = 6

        # Test with different batch sizes
        for batch_size in [1, 3, 5]:
            X = torch.randn(batch_size, seq_len, 16)
            extraction = extract_vit_partitions(model, X)

            # Should work regardless of batch size
            self.assertGreater(len(extraction.partitions), 0)

            # Attention weights should reflect the batch size
            for partition in extraction.partitions:
                self.assertEqual(partition.attention_weights.shape[0], batch_size)


if __name__ == "__main__":
    unittest.main()
