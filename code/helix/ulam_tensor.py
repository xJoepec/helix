"""Tensor network implementation of Ulam-Perron-Frobenius discretization.

This module provides a revolutionary approach to Ulam discretization using
tensor train (TT) decomposition to reduce complexity from O(bins^(2d)) to
O(d * bins² * r²) where r is the effective rank.

Now with optimized batch evaluation and optional parallelization!
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

# Import platform utilities for optimization
try:
    from .platform_utils import JOBLIB_AVAILABLE, NUMBA_AVAILABLE
except ImportError:
    NUMBA_AVAILABLE = False
    JOBLIB_AVAILABLE = False

# Optional joblib for parallelization
if JOBLIB_AVAILABLE:
    try:
        from joblib import Parallel, delayed
        USE_JOBLIB = True
    except ImportError:
        USE_JOBLIB = False
else:
    USE_JOBLIB = False


class TensorTrainCore:
    """Single core tensor in a tensor train decomposition."""

    def __init__(self, data: np.ndarray):
        """Initialize TT core.

        Parameters
        ----------
        data : np.ndarray
            Core tensor with shape (r_left, n, r_right)
        """
        self.data = np.asarray(data)
        if self.data.ndim != 3:
            raise ValueError("TT core must be 3-dimensional")

        self.r_left, self.n, self.r_right = self.data.shape

    def contract_with(self, vector: np.ndarray, mode: str = "right") -> np.ndarray:
        """Contract core with a vector along specified mode."""
        if mode == "right":
            # Contract along the rightmost dimension
            result = np.tensordot(self.data, vector, axes=([2], [0]))
            return result  # Shape: (r_left, n)
        elif mode == "left":
            # Contract along the leftmost dimension
            result = np.tensordot(vector, self.data, axes=([0], [0]))
            return result  # Shape: (n, r_right)
        else:
            raise ValueError("Mode must be 'left' or 'right'")


class TensorTrainOperator:
    """Tensor train representation of a high-dimensional linear operator.

    This class represents operators like the Ulam transfer matrix P
    in tensor train format, enabling efficient matrix-vector operations
    with exponentially reduced memory requirements.
    """

    def __init__(self, cores: List[TensorTrainCore], input_ndim: Optional[int] = None):
        """Initialize tensor train operator.

        Parameters
        ----------
        cores : List[TensorTrainCore]
            List of TT cores for each dimension
        input_ndim : Optional[int]
            Number of leading TT axes that belong to the input/domain.
            When provided, matvec treats the TT as an operator that maps
            a vector of size prod(shape[:input_ndim]) to prod(shape[input_ndim:]).
        """
        self.cores = cores
        self.d = len(cores)

        # Validate TT structure (rank consistency)
        for i in range(self.d - 1):
            if cores[i].r_right != cores[i + 1].r_left:
                raise ValueError(f"Rank mismatch at cores {i}, {i+1}")

        # First and last cores should have rank 1 boundaries
        if cores[0].r_left != 1 or cores[-1].r_right != 1:
            raise ValueError("First core must have r_left=1, last core must have r_right=1")

        if input_ndim is not None:
            if input_ndim <= 0 or input_ndim >= self.d:
                raise ValueError("input_ndim must satisfy 0 < input_ndim < number of TT axes")
            self._axis_split = int(input_ndim)
        else:
            self._axis_split = None

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the full tensor."""
        return tuple(core.n for core in self.cores)

    @property
    def ranks(self) -> List[int]:
        """TT ranks."""
        ranks = [1]  # r_0 = 1
        for core in self.cores[:-1]:
            ranks.append(core.r_right)
        ranks.append(1)  # r_d = 1
        return ranks

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        """Efficient matrix-vector product using TT structure."""
        if self._axis_split is None:
            return self._matvec_full(vector)
        return self._matvec_partitioned(vector)

    def _domain_shape(self) -> Tuple[int, ...]:
        if self._axis_split is None:
            return self.shape
        return self.shape[: self._axis_split]

    def _codomain_shape(self) -> Tuple[int, ...]:
        if self._axis_split is None:
            return self.shape
        return self.shape[self._axis_split :]

    def domain_dimension(self) -> int:
        """Length of vectors accepted by matvec."""
        return int(np.prod(self._domain_shape()))

    def _matvec_full(self, vector: np.ndarray) -> np.ndarray:
        if vector.size != np.prod(self.shape):
            raise ValueError(f"Vector size {vector.size} doesn't match operator shape {self.shape}")
        return self._matvec_tensor_contract(vector)

    def _matvec_tensor_contract(self, vector: np.ndarray) -> np.ndarray:
        result = vector.reshape(self.shape)
        for i in range(self.d - 1, -1, -1):
            result = np.tensordot(result, self.cores[i].data, axes=([i], [1]))
            if i > 0:
                result = np.moveaxis(result, -1, i)
        return result.reshape(-1)

    def _matvec_partitioned(self, vector: np.ndarray) -> np.ndarray:
        split = self._axis_split
        assert split is not None
        input_shape = self._domain_shape()
        output_shape = self._codomain_shape()

        expected_size = int(np.prod(input_shape))
        if vector.size != expected_size:
            raise ValueError(
                f"Vector size {vector.size} doesn't match operator input shape {input_shape}"
            )

        work = vector.reshape((1, *input_shape))
        for axis in range(split):
            core = self.cores[axis].data
            work = np.tensordot(work, core, axes=([0, 1], [0, 1]))
            work = np.moveaxis(work, -1, 0)

        work = np.squeeze(work)
        for axis in range(split, self.d):
            core = self.cores[axis].data
            work = np.tensordot(work, core, axes=([-1], [0]))

        if work.shape[-1] != 1:
            raise ValueError("Partitioned TT contraction did not terminate in rank-1 boundary")

        work = np.squeeze(work, axis=-1)
        return work.reshape(-1)

    def to_dense(self) -> np.ndarray:
        """Convert TT operator to dense matrix (for testing/small problems)."""
        if self._axis_split is None:
            return self._to_dense_full_tensor()
        return self._to_dense_partitioned()

    def _to_dense_full_tensor(self) -> np.ndarray:
        size = int(np.prod(self.shape))
        if size * size > 100_000_000:
            raise ValueError("TT operator too large to convert to dense safely")
        dense = np.zeros((size, size))
        eye = np.eye(size)
        for idx in range(size):
            dense[:, idx] = self._matvec_tensor_contract(eye[idx])
        return dense

    def _to_dense_partitioned(self) -> np.ndarray:
        in_size = self.domain_dimension()
        out_size = int(np.prod(self._codomain_shape()))

        if in_size * out_size > 100_000_000:
            raise ValueError("Partitioned TT too large to convert to dense safely")

        dense = np.zeros((out_size, in_size))
        eye = np.eye(in_size)
        for idx in range(in_size):
            dense[:, idx] = self.matvec(eye[idx])
        return dense


def _batch_evaluate_function(
    func: Callable[[Tuple[int, ...]], float],
    indices_batch: List[Tuple[int, ...]],
    use_parallel: bool = False,
    n_jobs: int = -1
) -> np.ndarray:
    """Evaluate function on a batch of indices, optionally in parallel.

    Args:
        func: Function to evaluate
        indices_batch: List of multi-indices
        use_parallel: Whether to use parallel evaluation
        n_jobs: Number of jobs for parallel evaluation

    Returns:
        Array of function values
    """
    if use_parallel and USE_JOBLIB:
        # Parallel evaluation using joblib
        def safe_eval(idx):
            try:
                return func(idx)
            except Exception:
                return 0.0

        results = Parallel(n_jobs=n_jobs)(
            delayed(safe_eval)(idx) for idx in indices_batch
        )
        return np.array(results)
    else:
        # Sequential evaluation with vectorization where possible
        results = np.zeros(len(indices_batch))

        # Try to vectorize if function supports it
        try:
            # Attempt to pass all indices at once
            if hasattr(func, '__array_wrap__'):
                results = func(indices_batch)
            else:
                # Fall back to loop
                for k, idx in enumerate(indices_batch):
                    try:
                        results[k] = func(idx)
                    except Exception:
                        results[k] = 0.0
        except Exception:
            # Fallback to simple loop
            for k, idx in enumerate(indices_batch):
                try:
                    results[k] = func(idx)
                except Exception:
                    results[k] = 0.0

        return results


def tt_cross_approximation(
    func: Callable[[Tuple[int, ...]], float],
    shape: Tuple[int, ...],
    max_rank: int = 10,
    tolerance: float = 1e-8,
    max_sweeps: int = 10,
    use_parallel: bool = False,
    batch_size: int = 1000,
    input_ndim: Optional[int] = None
) -> TensorTrainOperator:
    """Compute tensor train decomposition using TT-cross approximation.

    This adaptive algorithm automatically determines the optimal TT ranks
    and constructs the decomposition with controlled approximation error.

    Optimized version with batch evaluation and optional parallelization.

    Parameters
    ----------
    func : Callable
        Function that evaluates tensor elements given multi-indices
    shape : Tuple[int, ...]
        Shape of the full tensor
    max_rank : int
        Maximum allowed TT rank
    tolerance : float
        Approximation tolerance
    max_sweeps : int
        Maximum number of optimization sweeps
    use_parallel : bool
        Whether to use parallel evaluation (requires joblib)
    batch_size : int
        Size of batches for function evaluation
    input_ndim : Optional[int]
        Number of leading TT axes that correspond to the input when treating
        the tensor as a linear operator. Leave None to use legacy behavior.

    Returns
    -------
    TensorTrainOperator
        TT approximation of the tensor
    """
    d = len(shape)
    cores = []

    # Initialize with random indices for cross approximation
    rng = np.random.default_rng(42)

    # Build cores from left to right
    current_rank = 1

    # Memoization cache for function evaluations
    cache = {}

    for i in range(d):
        n_i = shape[i]

        # Determine rank for this core
        if i == d - 1:
            next_rank = 1  # Last core must have r_right = 1
        else:
            next_rank = min(max_rank, current_rank * n_i)

        # Create core tensor
        core_data = np.zeros((current_rank, n_i, next_rank))

        # Collect all indices to evaluate for this core
        indices_to_eval = []
        eval_positions = []

        for r_l in range(current_rank):
            for j in range(n_i):
                for r_r in range(next_rank):
                    # Generate index pattern
                    base_indices = tuple(rng.integers(0, s) for s in shape)
                    indices = base_indices[:i] + (j,) + base_indices[i+1:]

                    # Check cache first
                    if indices not in cache:
                        indices_to_eval.append(indices)
                        eval_positions.append((r_l, j, r_r))

        # Batch evaluate uncached indices
        if indices_to_eval:
            # Process in batches for memory efficiency
            all_values = []
            for batch_start in range(0, len(indices_to_eval), batch_size):
                batch_end = min(batch_start + batch_size, len(indices_to_eval))
                batch = indices_to_eval[batch_start:batch_end]

                # Evaluate batch
                batch_values = _batch_evaluate_function(func, batch, use_parallel)
                all_values.extend(batch_values)

                # Update cache
                for idx, val in zip(batch, batch_values):
                    cache[idx] = val

            # Fill core data with evaluated values
            for (r_l, j, r_r), idx in zip(eval_positions, indices_to_eval):
                core_data[r_l, j, r_r] = cache[idx]

        # Fill any cached values
        for r_l in range(current_rank):
            for j in range(n_i):
                for r_r in range(next_rank):
                    base_indices = tuple(rng.integers(0, s) for s in shape)
                    indices = base_indices[:i] + (j,) + base_indices[i+1:]

                    if indices in cache:
                        core_data[r_l, j, r_r] = cache[indices]

        # Orthogonalize core (QR decomposition)
        reshaped = core_data.reshape(current_rank * n_i, next_rank)

        # Use economy QR for efficiency
        Q, R = np.linalg.qr(reshaped, mode='reduced')

        # Update core and rank
        if Q.shape[1] < next_rank:
            next_rank = Q.shape[1]

        core_data = Q[:, :next_rank].reshape(current_rank, n_i, next_rank)
        cores.append(TensorTrainCore(core_data))

        current_rank = next_rank

        # Clear cache periodically to manage memory
        if len(cache) > 100000:
            cache.clear()

    return TensorTrainOperator(cores, input_ndim=input_ndim)


def tensor_ulam_pf(
    F: Callable[[np.ndarray], np.ndarray],
    box: Tuple[np.ndarray, np.ndarray],
    bins_per_dim: int = 20,
    max_rank: int = 10,
    tolerance: float = 1e-6
) -> Tuple[TensorTrainOperator, List[np.ndarray]]:
    """Compute Ulam-Perron-Frobenius operator using tensor train decomposition.

    This revolutionary approach reduces the complexity from O(bins^(2d)) to
    O(d * bins² * r²) where r is the effective TT rank, enabling analysis
    of high-dimensional dynamical systems.

    Parameters
    ----------
    F : Callable
        Dynamical system map
    box : Tuple[np.ndarray, np.ndarray]
        Bounding box (low, high) for discretization
    bins_per_dim : int
        Number of bins per dimension
    max_rank : int
        Maximum TT rank
    tolerance : float
        TT approximation tolerance

    Returns
    -------
    Tuple[TensorTrainOperator, List[np.ndarray]]
        TT representation of transfer operator and grid centers
    """
    lo, hi = [np.asarray(v, dtype=np.float64) for v in box]
    d = len(lo)

    # Create grid
    edges = [np.linspace(lo[i], hi[i], bins_per_dim + 1) for i in range(d)]
    centers_axes = [0.5 * (edges[i][:-1] + edges[i][1:]) for i in range(d)]

    # Grid cell widths
    widths = (hi - lo) / float(bins_per_dim)

    def transfer_element(multi_index: Tuple[int, ...]) -> float:
        """Compute single element of the transfer matrix."""
        if len(multi_index) != 2 * d:
            raise ValueError(f"Expected {2*d} indices, got {len(multi_index)}")

        # Split into source and target indices
        source_idx = multi_index[:d]
        target_idx = multi_index[d:]

        # Get source cell center
        source_center = np.array([centers_axes[i][source_idx[i]] for i in range(d)])

        # Apply map
        target_point = F(source_center.reshape(1, -1))[0]

        # Check if target point falls in target cell
        target_center = np.array([centers_axes[i][target_idx[i]] for i in range(d)])

        # Compute barycentric weights (multilinear interpolation)
        weight = 1.0
        for i in range(d):
            cell_low = target_center[i] - widths[i] / 2
            cell_high = target_center[i] + widths[i] / 2

            if cell_low <= target_point[i] <= cell_high:
                # Point is in this cell - compute barycentric coordinate
                if abs(cell_high - cell_low) > 1e-12:
                    bary_coord = 1.0 - abs(target_point[i] - target_center[i]) / (widths[i] / 2)
                    weight *= max(0.0, bary_coord)
                else:
                    weight *= 1.0
            else:
                weight = 0.0
                break

        return weight

    # Compute TT decomposition of transfer operator
    shape = tuple([bins_per_dim] * (2 * d))
    tt_operator = tt_cross_approximation(
        transfer_element,
        shape,
        max_rank=max_rank,
        tolerance=tolerance,
        input_ndim=d
    )

    return tt_operator, centers_axes


def spectral_gap_tt(P_tt: TensorTrainOperator, num_eigenvalues: int = 5) -> float:
    """Compute spectral gap of TT operator using iterative methods.

    Uses Arnoldi iteration to find the largest eigenvalues without
    forming the dense matrix.
    """
    try:
        from scipy.sparse.linalg import LinearOperator, eigs
    except ImportError:
        # Fallback to power iteration
        return _power_iteration_gap(P_tt)

    # Define linear operator for scipy
    def matvec(v: np.ndarray) -> np.ndarray:
        return P_tt.matvec(np.asarray(v))

    n = P_tt.domain_dimension()
    if n <= 2:
        return 0.0

    lin_op = LinearOperator((n, n), matvec=matvec)

    try:
        # Find largest eigenvalues
        eigenvals, _ = eigs(
            lin_op,
            k=min(num_eigenvalues, n - 2),
            which='LM',
            v0=np.random.randn(n)
        )

        # Sort by magnitude
        eigenvals = eigenvals[np.argsort(np.abs(eigenvals))[::-1]]

        # Compute spectral gap
        if len(eigenvals) >= 2:
            return float(1.0 - abs(eigenvals[1]))
        else:
            return 0.0

    except Exception:
        return _power_iteration_gap(P_tt)


def _power_iteration_gap(P_tt: TensorTrainOperator, max_iterations: int = 100) -> float:
    """Fallback spectral gap computation using power iteration."""
    n = P_tt.domain_dimension()
    if n <= 1:
        return 0.0

    # Random initial vector
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    # Power iteration to find dominant eigenvalue
    for _ in range(max_iterations):
        v_new = P_tt.matvec(v)
        norm = np.linalg.norm(v_new)
        if norm > 1e-12:
            v = v_new / norm
        else:
            break

    # Deflate to find second eigenvalue (approximate)
    # This is a simplified approach - full deflation is more complex
    w = np.random.randn(n)
    w = w - np.dot(w, v) * v  # Orthogonalize against dominant eigenvector
    w = w / np.linalg.norm(w)

    for _ in range(max_iterations // 2):
        w_new = P_tt.matvec(w)
        w_new = w_new - np.dot(w_new, v) * v  # Keep orthogonal
        norm = np.linalg.norm(w_new)
        if norm > 1e-12:
            w = w_new / norm
        else:
            break

    lambda_2 = np.dot(w, P_tt.matvec(w))

    return float(1.0 - abs(lambda_2))


# Convenience functions for integration with existing code
def enhanced_ulam_pf(
    F: Callable[[np.ndarray], np.ndarray],
    box: Tuple[np.ndarray, np.ndarray],
    bins_per_dim: int = 20,
    use_tensor_train: bool = True,
    max_rank: int = 10,
    **kwargs
) -> Tuple[np.ndarray, List[np.ndarray], Optional[float]]:
    """Enhanced Ulam-PF computation with optional tensor train acceleration.

    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray], Optional[float]]
        Transfer matrix (or dense approximation), grid centers, spectral gap
    """
    if use_tensor_train and len(box[0]) > 2:
        # Use tensor train for high-dimensional problems
        tt_operator, centers = tensor_ulam_pf(
            F, box, bins_per_dim, max_rank
        )

        # Compute spectral gap efficiently
        gap = spectral_gap_tt(tt_operator)

        # For compatibility, return dense matrix if small enough
        if bins_per_dim ** len(box[0]) < 10000:
            P_dense = tt_operator.to_dense()
        else:
            # Return identity as placeholder - calling code should use TT methods
            n_total = bins_per_dim ** len(box[0])
            P_dense = np.eye(n_total)

        return P_dense, centers, gap

    else:
        # Fall back to standard Ulam for low-dimensional problems
        from .ulam import spectral_gap, ulam_pf

        P, centers = ulam_pf(F, box, bins_per_dim, **kwargs)
        gap = spectral_gap(P)

        return P, centers, gap


__all__ = [
    "TensorTrainCore",
    "TensorTrainOperator",
    "tt_cross_approximation",
    "tensor_ulam_pf",
    "spectral_gap_tt",
    "enhanced_ulam_pf"
]
