#!/usr/bin/env python3
"""
Tensor Train Ulam-Perron-Frobenius Demo

This example demonstrates how to use the tensor train implementation
for high-dimensional Ulam discretization, showcasing memory efficiency
and computational advantages over standard dense methods.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for helix imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helix.ulam_tensor import (
    TensorTrainCore,
    TensorTrainOperator,
    enhanced_ulam_pf,
    spectral_gap_tt,
    tensor_ulam_pf,
)


def example_1_basic_tensor_train():
    """Example 1: Basic tensor train Ulam for a simple high-dimensional system."""
    print("=" * 60)
    print("Example 1: Basic Tensor Train Ulam")
    print("=" * 60)

    # Define a simple high-dimensional dynamical system
    def high_dim_contraction(X):
        """Simple contraction in each dimension with slight coupling."""
        d = X.shape[1]
        # Linear contraction with small coupling matrix
        A = 0.8 * np.eye(d) + 0.05 * np.ones((d, d)) / d
        return X @ A.T

    # Set up high-dimensional problem
    dimensions = 4
    box = (np.full(dimensions, -1.0), np.full(dimensions, 1.0))

    print(f"Problem: {dimensions}D dynamical system")
    print(f"Domain: {box[0]} to {box[1]}")

    # Compare different tensor train ranks
    ranks_to_test = [2, 4, 6, 8]
    results = {}

    for max_rank in ranks_to_test:
        print(f"\nTesting max_rank = {max_rank}...")

        # Build tensor train operator
        tt_operator, centers = tensor_ulam_pf(
            high_dim_contraction, box, bins_per_dim=6, max_rank=max_rank, tolerance=1e-6
        )

        # Compute spectral gap
        gap = spectral_gap_tt(tt_operator, num_eigenvalues=3)

        # Memory usage estimate
        total_elements = sum(core.data.size for core in tt_operator.cores)
        memory_mb = total_elements * 8 / (1024**2)  # 8 bytes per float64

        # Compression ratio
        dense_size = (6**dimensions) ** 2
        compression_ratio = dense_size / total_elements

        results[max_rank] = {
            "spectral_gap": gap,
            "memory_mb": memory_mb,
            "compression_ratio": compression_ratio,
            "actual_ranks": tt_operator.ranks,
        }

        print(f"  Spectral gap: {gap:.4f}")
        print(f"  Memory usage: {memory_mb:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.1f}x")
        print(f"  Actual TT ranks: {tt_operator.ranks}")

    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    for rank, result in results.items():
        print(
            f"Rank {rank:2d}: Gap={result['spectral_gap']:.4f}, "
            f"Memory={result['memory_mb']:5.1f}MB, "
            f"Compression={result['compression_ratio']:6.1f}x"
        )

    return results


def example_2_memory_comparison():
    """Example 2: Memory scaling comparison between dense and tensor train."""
    print("\n" + "=" * 60)
    print("Example 2: Memory Scaling Analysis")
    print("=" * 60)

    def linear_map(X):
        return 0.9 * X

    dimensions_list = [2, 3, 4, 5]
    bins_per_dim = 8
    max_rank = 6

    print(f"Bins per dimension: {bins_per_dim}")
    print(f"Max tensor rank: {max_rank}")
    print()

    results = []

    for d in dimensions_list:
        print(f"Testing {d}D system...")

        box = (np.full(d, -1.0), np.full(d, 1.0))

        # Dense matrix size (would be prohibitive for high dimensions)
        dense_matrix_size = (bins_per_dim**d) ** 2
        dense_memory_gb = dense_matrix_size * 8 / (1024**3)

        # Tensor train approach
        try:
            tt_operator, _ = tensor_ulam_pf(
                linear_map, box, bins_per_dim=bins_per_dim, max_rank=max_rank
            )

            tt_elements = sum(core.data.size for core in tt_operator.cores)
            tt_memory_mb = tt_elements * 8 / (1024**2)
            compression_ratio = dense_matrix_size / tt_elements

            results.append(
                {
                    "dimensions": d,
                    "dense_memory_gb": dense_memory_gb,
                    "tt_memory_mb": tt_memory_mb,
                    "compression_ratio": compression_ratio,
                    "feasible_dense": dense_memory_gb < 1.0,  # Less than 1GB
                }
            )

            print(f"  Dense matrix: {dense_memory_gb:.2f} GB")
            print(f"  Tensor train: {tt_memory_mb:.2f} MB")
            print(f"  Compression: {compression_ratio:.1f}x")
            print(f"  Dense feasible: {'Yes' if dense_memory_gb < 1.0 else 'No'}")

        except Exception as e:
            print(f"  Error: {e}")
            results.append(
                {
                    "dimensions": d,
                    "dense_memory_gb": dense_memory_gb,
                    "tt_memory_mb": float("inf"),
                    "compression_ratio": 0,
                    "feasible_dense": False,
                }
            )

    # Plot results if matplotlib is available
    try:
        dims = [r["dimensions"] for r in results]
        tt_memory = [r["tt_memory_mb"] for r in results if r["tt_memory_mb"] != float("inf")]
        dense_memory_mb = [r["dense_memory_gb"] * 1024 for r in results]

        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.semilogy(dims, dense_memory_mb, "r-o", label="Dense Matrix")
        plt.semilogy(dims[: len(tt_memory)], tt_memory, "b-s", label="Tensor Train")
        plt.xlabel("Dimensions")
        plt.ylabel("Memory (MB)")
        plt.title("Memory Usage Comparison")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        compression_ratios = [r["compression_ratio"] for r in results if r["compression_ratio"] > 0]
        plt.semilogy(dims[: len(compression_ratios)], compression_ratios, "g-^")
        plt.xlabel("Dimensions")
        plt.ylabel("Compression Ratio")
        plt.title("Tensor Train Compression")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("tensor_train_memory_scaling.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved as 'tensor_train_memory_scaling.png'")

    except ImportError:
        print("\nMatplotlib not available, skipping plots")

    return results


def example_3_enhanced_ulam_interface():
    """Example 3: Using the enhanced Ulam interface with automatic tensor train selection."""
    print("\n" + "=" * 60)
    print("Example 3: Enhanced Ulam Interface")
    print("=" * 60)

    def nonlinear_map(X):
        """Nonlinear map with different behavior in each dimension."""
        result = X.copy()
        result[:, 0] = 0.9 * np.tanh(X[:, 0])  # Saturating nonlinearity
        if X.shape[1] > 1:
            result[:, 1] = 0.8 * X[:, 1] + 0.1 * X[:, 0] ** 2  # Coupling
        if X.shape[1] > 2:
            result[:, 2:] = 0.85 * X[:, 2:]  # Linear contraction
        return result

    test_cases = [
        {"name": "2D (Standard Ulam)", "dimensions": 2, "use_tt": False},
        {"name": "2D (Force Tensor Train)", "dimensions": 2, "use_tt": True},
        {"name": "4D (Auto Tensor Train)", "dimensions": 4, "use_tt": True},
    ]

    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        print("-" * 40)

        d = case["dimensions"]
        box = (np.full(d, -1.5), np.full(d, 1.5))

        # Use enhanced interface
        P, centers, gap = enhanced_ulam_pf(
            nonlinear_map, box, bins_per_dim=8, use_tensor_train=case["use_tt"], max_rank=8
        )

        print(f"Matrix shape: {P.shape}")
        print(f"Spectral gap: {gap:.4f}")

        if case["use_tt"] and d > 2:
            print("Using tensor train representation")
        else:
            print("Using standard dense representation")

        # Analyze spectrum
        if P.shape[0] <= 1000:  # Only for reasonably sized matrices
            eigenvals = np.linalg.eigvals(P.T)
            eigenvals = np.sort(np.abs(eigenvals))[::-1]
            print(f"Top 5 eigenvalue magnitudes: {eigenvals[:5]}")


def example_4_custom_tensor_train():
    """Example 4: Building custom tensor train operators."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Tensor Train Construction")
    print("=" * 60)

    # Create a simple tensor train operator manually
    print("Building a 3D tensor train operator manually...")

    # Core 1: (1, 4, 3) - first core has r_left = 1
    core1_data = np.random.randn(1, 4, 3) * 0.5
    core1 = TensorTrainCore(core1_data)

    # Core 2: (3, 4, 2) - middle core
    core2_data = np.random.randn(3, 4, 2) * 0.5
    core2 = TensorTrainCore(core2_data)

    # Core 3: (2, 4, 1) - last core has r_right = 1
    core3_data = np.random.randn(2, 4, 1) * 0.5
    core3 = TensorTrainCore(core3_data)

    # Build tensor train operator
    tt_operator = TensorTrainOperator([core1, core2, core3])

    print(f"Tensor shape: {tt_operator.shape}")
    print(f"TT ranks: {tt_operator.ranks}")

    # Test matrix-vector multiplication
    vector_size = np.prod(tt_operator.shape)
    test_vector = np.random.randn(vector_size)

    print(f"Testing matvec with vector of size {vector_size}...")
    result = tt_operator.matvec(test_vector)
    print(f"Result vector size: {result.shape}")

    # Convert to dense for small operators (for verification)
    if vector_size <= 1000 and result.size == vector_size:
        print("Converting to dense matrix for verification...")
        dense_matrix = tt_operator.to_dense()
        dense_result = dense_matrix @ test_vector

        # Check consistency
        error = np.linalg.norm(result - dense_result)
        print(f"TT vs Dense error: {error:.2e}")

        if error < 1e-12:
            print("✓ Tensor train matvec is consistent with dense computation")
        else:
            print("✗ Tensor train matvec has significant error")
    elif result.size != vector_size:
        print("Skipping dense verification (TT basis compresses state space)")
    else:
        print("Skipping dense verification (matrix too large)")

    return tt_operator


def example_5_performance_tips():
    """Example 5: Performance tips and best practices."""
    print("\n" + "=" * 60)
    print("Example 5: Performance Tips and Best Practices")
    print("=" * 60)

    print("Tensor Train Performance Guidelines:")
    print()

    print("1. RANK SELECTION:")
    print("   - Start with max_rank = 4-8 for most problems")
    print("   - Higher ranks = better accuracy but more memory")
    print("   - Monitor actual ranks vs max_rank")
    print()

    print("2. MEMORY SCALING:")
    print("   - Dense matrix: O(bins^(2d)) memory")
    print("   - Tensor train: O(d × bins² × rank²) memory")
    print("   - Use tensor trains for d ≥ 3")
    print()

    print("3. NUMERICAL STABILITY:")
    print("   - Use tolerance ≥ 1e-12 for numerical stability")
    print("   - Monitor condition numbers of TT cores")
    print("   - Consider regularization for ill-conditioned problems")
    print()

    print("4. PERFORMANCE OPTIMIZATION:")
    print("   - Use show_progress=True for long computations")
    print("   - Reduce bins_per_dim for initial exploration")
    print("   - Cache TT operators for repeated analysis")
    print()

    # Demonstrate rank selection heuristic
    print("RANK SELECTION HEURISTIC:")

    def quick_rank_test(dimensions, bins_per_dim):
        """Quick test to suggest appropriate rank."""

        def test_map(X):
            return 0.9 * X

        box = (np.full(dimensions, -1.0), np.full(dimensions, 1.0))

        for test_rank in [2, 4, 6, 8, 12]:
            try:
                tt_op, _ = tensor_ulam_pf(
                    test_map, box, bins_per_dim=bins_per_dim, max_rank=test_rank, tolerance=1e-8
                )

                actual_max_rank = max(tt_op.ranks[1:-1]) if len(tt_op.ranks) > 2 else 1
                compression = (bins_per_dim**dimensions) ** 2 / sum(
                    c.data.size for c in tt_op.cores
                )

                print(
                    f"  max_rank={test_rank:2d} → actual_max={actual_max_rank:2d}, "
                    f"compression={compression:8.1f}x"
                )

                if actual_max_rank < test_rank * 0.7:
                    print(f"    → Suggested rank: {test_rank} (efficient)")
                    return test_rank

            except Exception as e:
                print(f"  max_rank={test_rank:2d} → failed: {e}")

        return 8  # Default fallback

    print("\nFor 4D problem with 6 bins per dimension:")
    suggested_rank = quick_rank_test(4, 6)
    print(f"Suggested starting rank: {suggested_rank}")


def main():
    """Run all tensor train examples."""
    print("Tensor Train Ulam-Perron-Frobenius Examples")
    print("=" * 60)
    print("This script demonstrates the tensor train implementation")
    print("for high-dimensional Ulam discretization.")
    print()

    try:
        # Run examples
        example_1_basic_tensor_train()
        example_2_memory_comparison()
        example_3_enhanced_ulam_interface()
        example_4_custom_tensor_train()
        example_5_performance_tips()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\nNext steps:")
        print("- Try the benchmark script: python code/scripts/benchmark_tensor_train.py")
        print("- Experiment with your own dynamical systems")
        print("- Adjust tensor train ranks for your specific problems")
        print("- Use show_progress=True for long-running computations")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
