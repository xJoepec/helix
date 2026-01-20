#!/usr/bin/env python3
"""Performance benchmarking for tensor train Ulam-Perron-Frobenius operations.

This script benchmarks different tensor train ranks to understand memory/accuracy tradeoffs
and provides guidance for optimal parameter selection in various scenarios.
"""

from __future__ import annotations

import argparse
import json

# Add the parent directory to path for helix imports
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix.ulam_tensor import TensorTrainOperator, spectral_gap_tt, tensor_ulam_pf


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    dimensions: int
    bins_per_dim: int
    max_rank: int
    actual_ranks: List[int]
    construction_time: float
    spectral_gap_time: float
    memory_usage_mb: float
    spectral_gap_value: float
    approximation_error: float
    compression_ratio: float


class DynamicalSystemBenchmarks:
    """Collection of test dynamical systems for benchmarking."""

    @staticmethod
    def linear_contraction(scale: float = 0.8):
        """Simple linear contraction map."""
        def map_func(X: np.ndarray) -> np.ndarray:
            return scale * X
        return map_func, f"Linear contraction (scale={scale})"

    @staticmethod
    def rotation_contraction(angle: float = 0.1, scale: float = 0.9):
        """2D rotation with contraction."""
        def map_func(X: np.ndarray) -> np.ndarray:
            if X.shape[1] != 2:
                raise ValueError("Rotation map requires 2D input")

            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            return scale * (X @ rotation.T)
        return map_func, f"2D rotation-contraction (angle={angle}, scale={scale})"

    @staticmethod
    def logistic_map_nd(r: float = 3.8):
        """N-dimensional logistic map."""
        def map_func(X: np.ndarray) -> np.ndarray:
            # Apply logistic map component-wise, then rescale to [-1, 1]
            # Logistic: x -> r * x * (1 - x) for x in [0, 1]
            # Transform input from [-1, 1] to [0, 1]
            X_01 = (X + 1) / 2
            result_01 = r * X_01 * (1 - X_01) / 4  # Normalize by max value
            return 2 * result_01 - 1  # Transform back to [-1, 1]
        return map_func, f"ND Logistic map (r={r})"

    @staticmethod
    def tent_map_nd():
        """N-dimensional tent map."""
        def map_func(X: np.ndarray) -> np.ndarray:
            return np.where(X < 0, 2 * X + 1, 1 - 2 * X)
        return map_func, "ND Tent map"


class TensorTrainBenchmarker:
    """Main benchmarking class for tensor train performance analysis."""

    def __init__(self, output_dir: str = "tt_benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def estimate_memory_usage(self, tt_operator: TensorTrainOperator) -> float:
        """Estimate memory usage of tensor train operator in MB."""
        total_elements = 0
        for core in tt_operator.cores:
            total_elements += core.data.size

        # Each element is a float64 (8 bytes)
        total_bytes = total_elements * 8
        return total_bytes / (1024 * 1024)  # Convert to MB

    def compute_approximation_error(
        self,
        tt_operator: TensorTrainOperator,
        reference_operator: np.ndarray = None
    ) -> float:
        """Compute approximation error against reference (if small enough)."""
        if reference_operator is None:
            total_size = np.prod(tt_operator.shape)
            if total_size > 10000:  # Too large for dense computation
                return 0.0  # Cannot compute error for large operators

            try:
                dense_tt = tt_operator.to_dense()
                # Compute error as Frobenius norm difference from identity pattern
                # This is a heuristic since we don't have a true reference
                identity_like = np.eye(total_size) * 0.1  # Weak identity pattern
                error = np.linalg.norm(dense_tt - identity_like, 'fro')
                return error / np.linalg.norm(identity_like, 'fro')
            except Exception:
                return 0.0

        try:
            dense_tt = tt_operator.to_dense()
            error = np.linalg.norm(dense_tt - reference_operator, 'fro')
            return error / np.linalg.norm(reference_operator, 'fro')
        except Exception:
            return 0.0

    def compute_compression_ratio(self, tt_operator: TensorTrainOperator) -> float:
        """Compute compression ratio compared to dense storage."""
        dense_size = np.prod(tt_operator.shape) ** 2  # Full matrix size
        tt_size = sum(core.data.size for core in tt_operator.cores)
        return dense_size / tt_size

    def benchmark_single_configuration(
        self,
        map_func,
        dimensions: int,
        bins_per_dim: int,
        max_rank: int,
        box: Tuple[np.ndarray, np.ndarray] = None
    ) -> BenchmarkResult:
        """Benchmark a single tensor train configuration."""

        if box is None:
            box = (np.full(dimensions, -1.0), np.full(dimensions, 1.0))

        # Time tensor train construction
        start_time = time.time()
        try:
            tt_operator, centers = tensor_ulam_pf(
                map_func, box, bins_per_dim=bins_per_dim, max_rank=max_rank
            )
            construction_time = time.time() - start_time
        except Exception as e:
            print(f"Failed to construct TT operator: {e}")
            return BenchmarkResult(
                dimensions=dimensions,
                bins_per_dim=bins_per_dim,
                max_rank=max_rank,
                actual_ranks=[0],
                construction_time=float('inf'),
                spectral_gap_time=float('inf'),
                memory_usage_mb=float('inf'),
                spectral_gap_value=0.0,
                approximation_error=float('inf'),
                compression_ratio=0.0
            )

        # Time spectral gap computation
        start_time = time.time()
        try:
            spectral_gap_value = spectral_gap_tt(tt_operator, num_eigenvalues=3)
            spectral_gap_time = time.time() - start_time
        except Exception as e:
            print(f"Failed to compute spectral gap: {e}")
            spectral_gap_value = 0.0
            spectral_gap_time = float('inf')

        # Compute metrics
        memory_usage = self.estimate_memory_usage(tt_operator)
        approximation_error = self.compute_approximation_error(tt_operator)
        compression_ratio = self.compute_compression_ratio(tt_operator)

        return BenchmarkResult(
            dimensions=dimensions,
            bins_per_dim=bins_per_dim,
            max_rank=max_rank,
            actual_ranks=tt_operator.ranks,
            construction_time=construction_time,
            spectral_gap_time=spectral_gap_time,
            memory_usage_mb=memory_usage,
            spectral_gap_value=spectral_gap_value,
            approximation_error=approximation_error,
            compression_ratio=compression_ratio
        )

    def benchmark_rank_scaling(
        self,
        map_func,
        map_name: str,
        dimensions: int = 3,
        bins_per_dim: int = 8,
        max_ranks: List[int] = None
    ) -> None:
        """Benchmark performance across different ranks."""

        if max_ranks is None:
            max_ranks = [2, 4, 6, 8, 10, 12, 16]

        print(f"\nBenchmarking rank scaling for {map_name}")
        print(f"Dimensions: {dimensions}, Bins per dim: {bins_per_dim}")
        print("-" * 60)

        rank_results = []

        for max_rank in max_ranks:
            print(f"Testing max_rank = {max_rank}...")

            result = self.benchmark_single_configuration(
                map_func, dimensions, bins_per_dim, max_rank
            )

            rank_results.append(result)
            self.results.append(result)

            print(f"  Construction time: {result.construction_time:.3f}s")
            print(f"  Memory usage: {result.memory_usage_mb:.2f} MB")
            print(f"  Compression ratio: {result.compression_ratio:.1f}x")
            print(f"  Spectral gap: {result.spectral_gap_value:.4f}")

        # Save results for this map
        self.save_rank_scaling_results(rank_results, map_name, dimensions, bins_per_dim)

    def benchmark_dimension_scaling(
        self,
        map_func,
        map_name: str,
        dimensions_list: List[int] = None,
        bins_per_dim: int = 6,
        max_rank: int = 8
    ) -> None:
        """Benchmark performance across different dimensions."""

        if dimensions_list is None:
            dimensions_list = [2, 3, 4, 5]

        print(f"\nBenchmarking dimension scaling for {map_name}")
        print(f"Bins per dim: {bins_per_dim}, Max rank: {max_rank}")
        print("-" * 60)

        dim_results = []

        for dimensions in dimensions_list:
            print(f"Testing dimensions = {dimensions}...")

            # Create appropriate map function for this dimension
            if "2D rotation" in map_name and dimensions != 2:
                continue  # Skip 2D-specific maps for other dimensions

            result = self.benchmark_single_configuration(
                map_func, dimensions, bins_per_dim, max_rank
            )

            dim_results.append(result)
            self.results.append(result)

            print(f"  Construction time: {result.construction_time:.3f}s")
            print(f"  Memory usage: {result.memory_usage_mb:.2f} MB")
            print(f"  Spectral gap: {result.spectral_gap_value:.4f}")

        # Save results
        self.save_dimension_scaling_results(dim_results, map_name, bins_per_dim, max_rank)

    def save_rank_scaling_results(
        self,
        results: List[BenchmarkResult],
        map_name: str,
        dimensions: int,
        bins_per_dim: int
    ) -> None:
        """Save and plot rank scaling results."""

        if not results:
            return

        # Extract data for plotting
        ranks = [r.max_rank for r in results]
        construction_times = [r.construction_time for r in results]
        memory_usage = [r.memory_usage_mb for r in results]
        compression_ratios = [r.compression_ratio for r in results]
        spectral_gaps = [r.spectral_gap_value for r in results]

        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Construction time vs rank
        ax1.plot(ranks, construction_times, 'bo-')
        ax1.set_xlabel('Max Rank')
        ax1.set_ylabel('Construction Time (s)')
        ax1.set_title('Construction Time vs Rank')
        ax1.grid(True)

        # Memory usage vs rank
        ax2.plot(ranks, memory_usage, 'ro-')
        ax2.set_xlabel('Max Rank')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Rank')
        ax2.grid(True)

        # Compression ratio vs rank
        ax3.plot(ranks, compression_ratios, 'go-')
        ax3.set_xlabel('Max Rank')
        ax3.set_ylabel('Compression Ratio')
        ax3.set_title('Compression Ratio vs Rank')
        ax3.set_yscale('log')
        ax3.grid(True)

        # Spectral gap vs rank
        ax4.plot(ranks, spectral_gaps, 'mo-')
        ax4.set_xlabel('Max Rank')
        ax4.set_ylabel('Spectral Gap')
        ax4.set_title('Spectral Gap vs Rank')
        ax4.grid(True)

        plt.suptitle(f'Rank Scaling: {map_name} ({dimensions}D, {bins_per_dim} bins/dim)')
        plt.tight_layout()

        # Save plot
        safe_name = map_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '')
        plot_path = self.output_dir / f"rank_scaling_{safe_name}_{dimensions}D.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Rank scaling plot saved to {plot_path}")

    def save_dimension_scaling_results(
        self,
        results: List[BenchmarkResult],
        map_name: str,
        bins_per_dim: int,
        max_rank: int
    ) -> None:
        """Save and plot dimension scaling results."""

        if not results:
            return

        # Extract data for plotting
        dimensions = [r.dimensions for r in results]
        construction_times = [r.construction_time for r in results]
        memory_usage = [r.memory_usage_mb for r in results]
        compression_ratios = [r.compression_ratio for r in results]

        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Construction time vs dimensions
        ax1.semilogy(dimensions, construction_times, 'bo-')
        ax1.set_xlabel('Dimensions')
        ax1.set_ylabel('Construction Time (s)')
        ax1.set_title('Construction Time vs Dimensions')
        ax1.grid(True)

        # Memory usage vs dimensions
        ax2.semilogy(dimensions, memory_usage, 'ro-')
        ax2.set_xlabel('Dimensions')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Dimensions')
        ax2.grid(True)

        # Compression ratio vs dimensions
        ax3.semilogy(dimensions, compression_ratios, 'go-')
        ax3.set_xlabel('Dimensions')
        ax3.set_ylabel('Compression Ratio')
        ax3.set_title('Compression Ratio vs Dimensions')
        ax3.grid(True)

        plt.suptitle(f'Dimension Scaling: {map_name} ({bins_per_dim} bins/dim, rank {max_rank})')
        plt.tight_layout()

        # Save plot
        safe_name = map_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '')
        plot_path = self.output_dir / f"dim_scaling_{safe_name}_rank{max_rank}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Dimension scaling plot saved to {plot_path}")

    def generate_summary_report(self) -> None:
        """Generate a comprehensive summary report."""

        if not self.results:
            print("No results to summarize.")
            return

        report = {
            "summary": {
                "total_benchmarks": len(self.results),
                "dimension_range": [min(r.dimensions for r in self.results),
                                   max(r.dimensions for r in self.results)],
                "rank_range": [min(r.max_rank for r in self.results),
                              max(r.max_rank for r in self.results)],
            },
            "recommendations": self.generate_recommendations(),
            "all_results": [
                {
                    "dimensions": r.dimensions,
                    "bins_per_dim": r.bins_per_dim,
                    "max_rank": r.max_rank,
                    "actual_ranks": r.actual_ranks,
                    "construction_time": r.construction_time,
                    "spectral_gap_time": r.spectral_gap_time,
                    "memory_usage_mb": r.memory_usage_mb,
                    "spectral_gap_value": r.spectral_gap_value,
                    "approximation_error": r.approximation_error,
                    "compression_ratio": r.compression_ratio
                }
                for r in self.results
            ]
        }

        # Save report
        report_path = self.output_dir / "benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nBenchmark report saved to {report_path}")

        # Print summary
        print("\n" + "="*60)
        print("TENSOR TRAIN BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total benchmarks run: {len(self.results)}")
        print(f"Dimension range: {report['summary']['dimension_range']}")
        print(f"Rank range: {report['summary']['rank_range']}")

        print("\nRECOMMENDations:")
        for rec in report['recommendations']:
            print(f"• {rec}")

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on benchmark results."""

        recommendations = []

        if not self.results:
            return ["No results available for recommendations."]

        # Analyze construction time vs rank
        fast_results = [r for r in self.results if r.construction_time < 1.0]
        if fast_results:
            avg_rank = np.mean([r.max_rank for r in fast_results])
            recommendations.append(
                f"For fast construction (<1s), use max_rank ≤ {int(avg_rank)}"
            )

        # Analyze memory usage
        low_memory_results = [r for r in self.results if r.memory_usage_mb < 100]
        if low_memory_results:
            avg_rank = np.mean([r.max_rank for r in low_memory_results])
            recommendations.append(
                f"For low memory usage (<100MB), use max_rank ≤ {int(avg_rank)}"
            )

        # Analyze compression ratio
        high_compression = [r for r in self.results if r.compression_ratio > 1000]
        if high_compression:
            avg_rank = np.mean([r.max_rank for r in high_compression])
            recommendations.append(
                f"High compression (>1000x) achieved with max_rank ≈ {int(avg_rank)}"
            )

        # Analyze by dimension
        for dim in [2, 3, 4, 5]:
            dim_results = [r for r in self.results if r.dimensions == dim]
            if dim_results:
                best_result = min(dim_results, key=lambda r: r.construction_time)
                recommendations.append(
                    f"For {dim}D problems, max_rank={best_result.max_rank} gives good performance"
                )

        return recommendations


def main():
    """Main benchmarking function."""

    parser = argparse.ArgumentParser(description="Benchmark tensor train performance")
    parser.add_argument("--output-dir", default="tt_benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with fewer configurations")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 3, 4],
                       help="Dimensions to test")
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 6, 8, 10],
                       help="Ranks to test")
    parser.add_argument("--bins-per-dim", type=int, default=8,
                       help="Number of bins per dimension")

    args = parser.parse_args()

    # Initialize benchmarker
    benchmarker = TensorTrainBenchmarker(args.output_dir)

    # Get test systems
    systems = DynamicalSystemBenchmarks()

    # Test systems to benchmark
    test_maps = [
        systems.linear_contraction(0.8),
        systems.rotation_contraction(0.1, 0.9),
        systems.logistic_map_nd(3.5),
        systems.tent_map_nd(),
    ]

    # Adjust parameters for quick mode
    if args.quick:
        dimensions_list = args.dimensions[:2]  # Test fewer dimensions
        ranks_list = args.ranks[:3]  # Test fewer ranks
    else:
        dimensions_list = args.dimensions
        ranks_list = args.ranks

    print("Starting Tensor Train Performance Benchmark")
    print("="*50)
    print(f"Output directory: {args.output_dir}")
    print(f"Testing dimensions: {dimensions_list}")
    print(f"Testing ranks: {ranks_list}")
    print(f"Bins per dimension: {args.bins_per_dim}")

    # Run benchmarks
    for map_func, map_name in test_maps:
        try:
            # Skip 2D-specific maps for other dimensions
            if "2D rotation" in map_name:
                if 2 in dimensions_list:
                    benchmarker.benchmark_rank_scaling(
                        map_func, map_name, dimensions=2,
                        bins_per_dim=args.bins_per_dim, max_ranks=ranks_list
                    )
            else:
                # Test rank scaling for 3D
                if 3 in dimensions_list:
                    benchmarker.benchmark_rank_scaling(
                        map_func, map_name, dimensions=3,
                        bins_per_dim=args.bins_per_dim, max_ranks=ranks_list
                    )

                # Test dimension scaling
                benchmarker.benchmark_dimension_scaling(
                    map_func, map_name, dimensions_list=dimensions_list,
                    bins_per_dim=args.bins_per_dim, max_rank=8
                )

        except Exception as e:
            print(f"Error benchmarking {map_name}: {e}")
            continue

    # Generate final report
    benchmarker.generate_summary_report()

    print(f"\nBenchmarking complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()