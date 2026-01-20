#!/usr/bin/env python3
"""Comprehensive benchmark suite for Helix optimizations.

This script benchmarks all major components before and after optimization,
measuring speedups across different platforms.

Usage:
    python benchmark_optimizations.py [--save-results] [--compare-with baseline.json]
"""

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for helix imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings during benchmarking
warnings.filterwarnings('ignore')


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    time_original: float
    time_optimized: float
    speedup: float
    parameters: Dict[str, Any]
    platform_info: Dict[str, str]


class BenchmarkSuite:
    """Comprehensive benchmark suite for Helix optimizations."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
        self.platform_info = self._get_platform_info()

    def _get_platform_info(self) -> Dict[str, str]:
        """Get platform information."""
        import platform
        import sys

        info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }

        # Check for optimization libraries
        try:
            import numba
            info['numba'] = numba.__version__
        except ImportError:
            info['numba'] = 'not installed'

        try:
            import joblib
            info['joblib'] = joblib.__version__
        except ImportError:
            info['joblib'] = 'not installed'

        try:
            import scipy
            info['scipy'] = scipy.__version__
        except ImportError:
            info['scipy'] = 'not installed'

        # Check BLAS configuration
        try:
            import numpy as np
            config_str = str(np.show_config(mode='dicts'))
            if 'accelerate' in config_str.lower():
                info['blas'] = 'accelerate'
            elif 'mkl' in config_str.lower():
                info['blas'] = 'mkl'
            elif 'openblas' in config_str.lower():
                info['blas'] = 'openblas'
            else:
                info['blas'] = 'generic'
        except Exception:
            info['blas'] = 'unknown'

        return info

    def print_header(self, text: str):
        """Print a formatted header."""
        if self.verbose:
            print("\n" + "=" * 60)
            print(text)
            print("=" * 60)

    def time_function(self, func: Callable, *args, n_runs: int = 3, **kwargs) -> float:
        """Time a function execution."""
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        return np.median(times)

    def benchmark_ulam_deposition(self):
        """Benchmark Ulam barycentric deposition optimization."""
        self.print_header("Benchmarking Ulam Barycentric Deposition")

        try:
            from helix.ulam import ulam_pf

            # Test parameters
            dims = [2, 3]
            bins_per_dim = [10, 15, 20]

            def test_func(x):
                return np.clip(x * 1.1 + 0.1 * np.sin(10 * x), -2, 2)

            for d in dims:
                for bins in bins_per_dim:
                    box = (np.ones(d) * -2, np.ones(d) * 2)

                    if self.verbose:
                        print(f"\nTesting {d}D with {bins} bins/dim...")

                    # Original implementation
                    time_orig = self.time_function(
                        ulam_pf, test_func, box, bins,
                        samples_per_cell=10, use_vectorized=False
                    )

                    # Optimized implementation
                    time_opt = self.time_function(
                        ulam_pf, test_func, box, bins,
                        samples_per_cell=10, use_vectorized=True
                    )

                    speedup = time_orig / time_opt

                    result = BenchmarkResult(
                        name="ulam_deposition",
                        time_original=time_orig,
                        time_optimized=time_opt,
                        speedup=speedup,
                        parameters={'dimensions': d, 'bins_per_dim': bins},
                        platform_info=self.platform_info
                    )
                    self.results.append(result)

                    if self.verbose:
                        print(f"  Original: {time_orig:.4f}s")
                        print(f"  Optimized: {time_opt:.4f}s")
                        print(f"  Speedup: {speedup:.2f}x")

        except Exception as e:
            print(f"Error benchmarking Ulam deposition: {e}")

    def benchmark_eigenvalue_computation(self):
        """Benchmark eigenvalue computation optimization."""
        self.print_header("Benchmarking Eigenvalue Computation")

        try:
            from helix.ulam import spectral_gap

            matrix_sizes = [50, 100, 200, 500]

            for size in matrix_sizes:
                if self.verbose:
                    print(f"\nTesting {size}x{size} matrix...")

                # Create a random stochastic matrix
                P = np.random.rand(size, size)
                P = P / P.sum(axis=1, keepdims=True)

                # Original (dense)
                time_orig = self.time_function(spectral_gap, P, use_sparse=False)

                # Optimized (sparse)
                time_opt = self.time_function(spectral_gap, P, use_sparse=True)

                speedup = time_orig / time_opt

                result = BenchmarkResult(
                    name="spectral_gap",
                    time_original=time_orig,
                    time_optimized=time_opt,
                    speedup=speedup,
                    parameters={'matrix_size': size},
                    platform_info=self.platform_info
                )
                self.results.append(result)

                if self.verbose:
                    print(f"  Original: {time_orig:.4f}s")
                    print(f"  Optimized: {time_opt:.4f}s")
                    print(f"  Speedup: {speedup:.2f}x")

        except Exception as e:
            print(f"Error benchmarking eigenvalue computation: {e}")

    def benchmark_partition_extraction(self):
        """Benchmark partition signature generation optimization."""
        self.print_header("Benchmarking Partition Extraction")

        try:
            import torch.nn as nn
            from helix.partitions import extract_partitions

            # Create a simple test network
            class TestNet(nn.Module):
                def __init__(self, width=64, depth=3):
                    super().__init__()
                    layers = []
                    in_dim = 2
                    for i in range(depth):
                        layers.append(nn.Linear(in_dim, width))
                        layers.append(nn.ReLU())
                        in_dim = width
                    layers.append(nn.Linear(width, 2))
                    self.net = nn.Sequential(*layers)

                def forward(self, x):
                    return self.net(x)

            sample_sizes = [100, 500, 1000]
            widths = [32, 64]

            for n_samples in sample_sizes:
                for width in widths:
                    if self.verbose:
                        print(f"\nTesting {n_samples} samples, width {width}...")

                    model = TestNet(width=width)
                    X = np.random.randn(n_samples, 2)

                    # Original implementation
                    time_orig = self.time_function(
                        extract_partitions, model, X, use_vectorized=False
                    )

                    # Optimized implementation
                    time_opt = self.time_function(
                        extract_partitions, model, X, use_vectorized=True
                    )

                    speedup = time_orig / time_opt

                    result = BenchmarkResult(
                        name="partition_extraction",
                        time_original=time_orig,
                        time_optimized=time_opt,
                        speedup=speedup,
                        parameters={'n_samples': n_samples, 'width': width},
                        platform_info=self.platform_info
                    )
                    self.results.append(result)

                    if self.verbose:
                        print(f"  Original: {time_orig:.4f}s")
                        print(f"  Optimized: {time_opt:.4f}s")
                        print(f"  Speedup: {speedup:.2f}x")

        except Exception as e:
            print(f"Error benchmarking partition extraction: {e}")

    def benchmark_tensor_train(self):
        """Benchmark tensor train decomposition optimization."""
        self.print_header("Benchmarking Tensor Train Decomposition")

        try:
            from helix.ulam_tensor import tt_cross_approximation

            def test_func(indices):
                """Simple test function for TT decomposition."""
                return np.sum(indices) / len(indices)

            dims = [3, 4]
            sizes = [10, 15]

            for d in dims:
                for s in sizes:
                    if self.verbose:
                        print(f"\nTesting {d}D tensor, size {s} per dim...")

                    shape = tuple([s] * d)

                    # Original (no parallelization/batching)
                    time_orig = self.time_function(
                        tt_cross_approximation, test_func, shape,
                        use_parallel=False, batch_size=1
                    )

                    # Optimized (with batching)
                    time_opt = self.time_function(
                        tt_cross_approximation, test_func, shape,
                        use_parallel=False, batch_size=100
                    )

                    speedup = time_orig / time_opt

                    result = BenchmarkResult(
                        name="tensor_train",
                        time_original=time_orig,
                        time_optimized=time_opt,
                        speedup=speedup,
                        parameters={'dimensions': d, 'size_per_dim': s},
                        platform_info=self.platform_info
                    )
                    self.results.append(result)

                    if self.verbose:
                        print(f"  Original: {time_orig:.4f}s")
                        print(f"  Optimized: {time_opt:.4f}s")
                        print(f"  Speedup: {speedup:.2f}x")

        except Exception as e:
            print(f"Error benchmarking tensor train: {e}")

    def benchmark_matrix_operations(self):
        """Benchmark basic matrix operations to verify BLAS optimization."""
        self.print_header("Benchmarking Matrix Operations (BLAS)")

        sizes = [100, 500, 1000, 2000]

        for n in sizes:
            if self.verbose:
                print(f"\nTesting {n}x{n} matrix multiplication...")

            A = np.random.randn(n, n).astype(np.float64)
            B = np.random.randn(n, n).astype(np.float64)

            time_matmul = self.time_function(lambda: A @ B)

            gflops = 2 * n**3 / time_matmul / 1e9

            result = BenchmarkResult(
                name="matrix_multiply",
                time_original=time_matmul,
                time_optimized=time_matmul,
                speedup=1.0,
                parameters={'size': n, 'gflops': gflops},
                platform_info=self.platform_info
            )
            self.results.append(result)

            if self.verbose:
                print(f"  Time: {time_matmul:.4f}s")
                print(f"  Performance: {gflops:.2f} GFLOPS")

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all benchmark results."""
        summary = {
            'platform': self.platform_info,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': [asdict(r) for r in self.results],
            'aggregate_stats': {}
        }

        # Calculate aggregate statistics
        if self.results:
            speedups_by_test = {}
            for result in self.results:
                if result.name not in speedups_by_test:
                    speedups_by_test[result.name] = []
                speedups_by_test[result.name].append(result.speedup)

            for test_name, speedups in speedups_by_test.items():
                summary['aggregate_stats'][test_name] = {
                    'mean_speedup': np.mean(speedups),
                    'median_speedup': np.median(speedups),
                    'min_speedup': np.min(speedups),
                    'max_speedup': np.max(speedups),
                }

            summary['overall_mean_speedup'] = np.mean([r.speedup for r in self.results])
            summary['overall_median_speedup'] = np.median([r.speedup for r in self.results])

        return summary

    def print_summary(self):
        """Print a summary of benchmark results."""
        self.print_header("Benchmark Summary")

        summary = self.generate_summary()

        print("\nPlatform Information:")
        for key, value in summary['platform'].items():
            print(f"  {key}: {value}")

        print("\nAggregate Results by Test:")
        for test_name, stats in summary['aggregate_stats'].items():
            print(f"\n{test_name}:")
            print(f"  Mean speedup: {stats['mean_speedup']:.2f}x")
            print(f"  Median speedup: {stats['median_speedup']:.2f}x")
            print(f"  Range: {stats['min_speedup']:.2f}x - {stats['max_speedup']:.2f}x")

        if 'overall_mean_speedup' in summary:
            print("\nOverall Performance:")
            print(f"  Mean speedup: {summary['overall_mean_speedup']:.2f}x")
            print(f"  Median speedup: {summary['overall_median_speedup']:.2f}x")

    def save_results(self, filename: str):
        """Save benchmark results to JSON file."""
        summary = self.generate_summary()
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {filename}")

    def compare_with_baseline(self, baseline_file: str):
        """Compare current results with a baseline."""
        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)

            self.print_header("Comparison with Baseline")

            current = self.generate_summary()

            # Compare overall speedups
            if 'overall_mean_speedup' in current and 'overall_mean_speedup' in baseline:
                current_speedup = current['overall_mean_speedup']
                baseline_speedup = baseline['overall_mean_speedup']
                improvement = (current_speedup - baseline_speedup) / baseline_speedup * 100

                print("\nOverall Performance Change:")
                print(f"  Baseline: {baseline_speedup:.2f}x")
                print(f"  Current: {current_speedup:.2f}x")
                print(f"  Change: {improvement:+.1f}%")

            # Compare by test
            print("\nPer-Test Comparison:")
            for test_name in current['aggregate_stats']:
                if test_name in baseline.get('aggregate_stats', {}):
                    curr_mean = current['aggregate_stats'][test_name]['mean_speedup']
                    base_mean = baseline['aggregate_stats'][test_name]['mean_speedup']
                    change = (curr_mean - base_mean) / base_mean * 100
                    print(f"  {test_name}: {base_mean:.2f}x → {curr_mean:.2f}x ({change:+.1f}%)")

        except Exception as e:
            print(f"Error comparing with baseline: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark suite for Helix optimizations"
    )
    parser.add_argument(
        '--save-results',
        type=str,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--compare-with',
        type=str,
        help='Compare with baseline results file'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmarks only'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )

    args = parser.parse_args()

    # Create benchmark suite
    suite = BenchmarkSuite(verbose=not args.quiet)

    # Run benchmarks
    if not args.quick:
        suite.benchmark_matrix_operations()
        suite.benchmark_ulam_deposition()
        suite.benchmark_eigenvalue_computation()
        suite.benchmark_partition_extraction()
        suite.benchmark_tensor_train()
    else:
        # Quick tests only
        suite.benchmark_matrix_operations()
        suite.benchmark_ulam_deposition()

    # Print summary
    suite.print_summary()

    # Save results if requested
    if args.save_results:
        suite.save_results(args.save_results)

    # Compare with baseline if provided
    if args.compare_with:
        suite.compare_with_baseline(args.compare_with)


if __name__ == "__main__":
    main()