"""Platform detection and optimization utilities for Helix.

This module detects the current platform and available acceleration libraries,
then configures optimal backends for maximum performance.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import subprocess
import sys
import warnings
from typing import Dict, Optional, Tuple


def detect_platform() -> Dict[str, str]:
    """Detect current platform and architecture details.

    Returns:
        Dictionary with platform information including:
        - system: Operating system name
        - machine: Machine architecture
        - processor: Processor type
        - python_version: Python version
        - is_arm: Whether running on ARM architecture
        - is_apple_silicon: Whether running on Apple Silicon
    """
    system = platform.system()
    machine = platform.machine().lower()
    processor = platform.processor()

    return {
        'system': system,
        'machine': machine,
        'processor': processor,
        'python_version': sys.version,
        'is_arm': 'arm' in machine or 'aarch' in machine,
        'is_apple_silicon': system == 'Darwin' and ('arm64' in machine or 'aarch64' in machine)
    }


def detect_blas_config() -> Optional[str]:
    """Detect the current BLAS configuration used by NumPy.

    Returns:
        Name of BLAS library or None if detection fails
    """
    try:
        import numpy as np
        config = np.show_config(mode='dicts')

        # Check for various BLAS implementations
        if 'accelerate' in str(config).lower():
            return 'accelerate'
        elif 'mkl' in str(config).lower():
            return 'mkl'
        elif 'openblas' in str(config).lower():
            return 'openblas'
        elif 'blis' in str(config).lower():
            return 'blis'

        # Alternative detection method
        if hasattr(np, '__config__'):
            config_str = str(np.__config__.show())
            if 'accelerate' in config_str.lower():
                return 'accelerate'
            elif 'mkl' in config_str.lower():
                return 'mkl'
            elif 'openblas' in config_str.lower():
                return 'openblas'
    except Exception:
        pass

    return None


def detect_gpu_backend() -> Dict[str, bool]:
    """Detect available GPU backends.

    Returns:
        Dictionary with GPU backend availability
    """
    gpu_info = {
        'cuda': False,
        'cuda_version': None,
        'mps': False,
        'metal': False,
        'rocm': False,
    }

    # Check for CUDA
    try:
        import torch
        gpu_info['cuda'] = torch.cuda.is_available()
        if gpu_info['cuda']:
            gpu_info['cuda_version'] = torch.version.cuda
    except ImportError:
        pass

    # Check for Apple Metal Performance Shaders
    try:
        import torch
        if hasattr(torch.backends, 'mps'):
            gpu_info['mps'] = torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        pass

    # Check for MLX (Apple Metal)
    try:
        importlib.import_module("mlx.core")
        gpu_info['metal'] = True
    except ImportError:
        pass

    # Check for ROCm (AMD GPUs)
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info['rocm'] = True
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    return gpu_info


def get_optimal_backend() -> str:
    """Determine the optimal backend for the current platform.

    Returns:
        String identifier for the recommended backend
    """
    platform_info = detect_platform()
    blas = detect_blas_config()
    gpu = detect_gpu_backend()

    # Apple Silicon - prefer Accelerate and Metal
    if platform_info['is_apple_silicon']:
        if blas == 'accelerate':
            return 'accelerate'
        else:
            warnings.warn(
                "Running on Apple Silicon but not using Accelerate framework. "
                "For optimal performance, reinstall NumPy with Accelerate support."
            )
            return 'accelerate_recommended'

    # Linux with CUDA
    if platform_info['system'] == 'Linux':
        if gpu['cuda']:
            return 'cuda'
        elif blas == 'mkl':
            return 'mkl'
        elif blas == 'openblas':
            return 'openblas'
        else:
            return 'openblas_recommended'

    # Windows - prefer MKL
    if platform_info['system'] == 'Windows':
        if gpu['cuda']:
            return 'cuda'
        elif blas == 'mkl':
            return 'mkl'
        else:
            return 'mkl_recommended'

    # Default fallback
    return 'numpy_default'


def configure_threading(backend: Optional[str] = None) -> None:
    """Configure optimal threading for the detected or specified backend.

    Args:
        backend: Override backend detection with specific backend
    """
    if backend is None:
        backend = get_optimal_backend()

    # Configure based on backend
    if backend in ['accelerate', 'accelerate_recommended']:
        # Apple Accelerate works best with limited threading
        os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
        os.environ.setdefault('OMP_NUM_THREADS', '1')

    elif backend == 'mkl':
        # Intel MKL threading configuration
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
        os.environ.setdefault('MKL_NUM_THREADS', str(n_cores))
        os.environ.setdefault('OMP_NUM_THREADS', str(n_cores))

    elif backend in ['openblas', 'openblas_recommended']:
        # OpenBLAS threading
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
        os.environ.setdefault('OPENBLAS_NUM_THREADS', str(n_cores))
        os.environ.setdefault('OMP_NUM_THREADS', str(n_cores))

    elif backend == 'cuda':
        # For CUDA, limit CPU threads to avoid contention
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')


def get_device(prefer_gpu: bool = True) -> Tuple[str, Optional[object]]:
    """Get the optimal computation device.

    Args:
        prefer_gpu: Whether to prefer GPU if available

    Returns:
        Tuple of (device_type, device_object) where device_object
        is a torch device if available, otherwise None
    """
    if not prefer_gpu:
        return 'cpu', None

    gpu_info = detect_gpu_backend()
    platform_info = detect_platform()

    # Try CUDA first (most common for scientific computing)
    if gpu_info['cuda']:
        try:
            import torch
            return 'cuda', torch.device('cuda')
        except ImportError:
            pass

    # Try MPS for Apple Silicon
    if gpu_info['mps'] and platform_info['is_apple_silicon']:
        try:
            import torch
            return 'mps', torch.device('mps')
        except ImportError:
            pass

    # Try MLX for Apple Silicon
    if gpu_info['metal'] and platform_info['is_apple_silicon']:
        return 'metal', None  # MLX doesn't use device objects

    return 'cpu', None


def print_optimization_report() -> None:
    """Print a detailed optimization report for the current system."""
    platform_info = detect_platform()
    blas = detect_blas_config()
    gpu = detect_gpu_backend()
    backend = get_optimal_backend()
    device_type, _ = get_device()

    print("=" * 60)
    print("Helix Platform Optimization Report")
    print("=" * 60)

    print("\nPlatform Information:")
    print(f"  System: {platform_info['system']}")
    print(f"  Architecture: {platform_info['machine']}")
    print(f"  Processor: {platform_info['processor']}")
    print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    if platform_info['is_apple_silicon']:
        print("  ✓ Apple Silicon detected")

    print("\nBLAS Configuration:")
    if blas:
        print(f"  Current: {blas}")
    else:
        print("  ⚠ No optimized BLAS detected")

    print("\nGPU Backends:")
    if gpu['cuda']:
        print(f"  ✓ CUDA available (version {gpu['cuda_version']})")
    if gpu['mps']:
        print("  ✓ Metal Performance Shaders available")
    if gpu['metal']:
        print("  ✓ MLX (Metal) available")
    if gpu['rocm']:
        print("  ✓ ROCm available")
    if not any(gpu.values()):
        print("  ⚠ No GPU acceleration available")

    print("\nRecommended Configuration:")
    print(f"  Backend: {backend}")
    print(f"  Device: {device_type}")

    # Platform-specific recommendations
    print("\nOptimization Recommendations:")
    if platform_info['is_apple_silicon'] and blas != 'accelerate':
        print("  ⚠ For optimal performance on Apple Silicon:")
        print("    pip uninstall numpy")
        print("    pip install numpy --no-binary :all: --no-cache-dir")
        print("    This will compile NumPy with Accelerate framework")

    if platform_info['system'] == 'Linux' and not gpu['cuda'] and blas != 'mkl':
        print("  ⚠ For optimal performance on Linux:")
        print("    conda install mkl numpy")
        print("    or")
        print("    apt-get install libopenblas-dev")
        print("    pip install numpy --no-binary :all:")

    if platform_info['system'] == 'Windows' and blas != 'mkl':
        print("  ⚠ For optimal performance on Windows:")
        print("    pip install numpy-mkl")

    print("=" * 60)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# Optional dependencies with graceful detection
NUMBA_AVAILABLE = _has_module("numba")
JOBLIB_AVAILABLE = _has_module("joblib")
CUPY_AVAILABLE = _has_module("cupy")


def check_optimization_dependencies() -> Dict[str, bool]:
    """Check which optimization dependencies are available.

    Returns:
        Dictionary mapping dependency names to availability
    """
    deps = {
        'numba': NUMBA_AVAILABLE,
        'joblib': JOBLIB_AVAILABLE,
        'cupy': CUPY_AVAILABLE,
        'scipy': False,
        'torch': False,
        'mlx': False,
    }

    for mod_name, key in [('scipy', 'scipy'), ('torch', 'torch'), ('mlx', 'mlx')]:
        deps[key] = _has_module(mod_name)

    return deps


# Auto-configure on import
configure_threading()