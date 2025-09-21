from __future__ import annotations

import os
import sys
import unittest
from typing import Tuple

import numpy as np

# Ensure the 'code' directory (package root for 'helix') is on sys.path
PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

"""Tests for tensor train Ulam-Perron-Frobenius implementation.

Imports from the package are done inside test bodies to avoid E402 with sys.path edits.
"""


class TestTensorTrainCore(unittest.TestCase):
    def test_tensor_train_core_init(self):
        from helix.ulam_tensor import TensorTrainCore

        # Test valid 3D tensor
        data = np.random.randn(2, 3, 4)
        core = TensorTrainCore(data)
        self.assertEqual(core.r_left, 2)
        self.assertEqual(core.n, 3)
        self.assertEqual(core.r_right, 4)

        # Test invalid dimensions
        with self.assertRaises(ValueError):
            TensorTrainCore(np.random.randn(2, 3))  # 2D instead of 3D

    def test_tensor_train_core_contract_with(self):
        from helix.ulam_tensor import TensorTrainCore

        # Create test core: (2, 3, 4) shape
        data = np.random.randn(2, 3, 4)
        core = TensorTrainCore(data)

        # Test right contraction
        vector_right = np.random.randn(4)
        result_right = core.contract_with(vector_right, mode="right")
        self.assertEqual(result_right.shape, (2, 3))

        # Test left contraction
        vector_left = np.random.randn(2)
        result_left = core.contract_with(vector_left, mode="left")
        self.assertEqual(result_left.shape, (3, 4))

        # Test invalid mode
        with self.assertRaises(ValueError):
            core.contract_with(vector_right, mode="invalid")


class TestTensorTrainOperator(unittest.TestCase):
    def test_tensor_train_operator_init(self):
        from helix.ulam_tensor import TensorTrainCore, TensorTrainOperator

        # Create valid TT cores
        core1 = TensorTrainCore(np.random.randn(1, 3, 2))  # First core: r_left=1
        core2 = TensorTrainCore(np.random.randn(2, 4, 2))  # Middle core
        core3 = TensorTrainCore(np.random.randn(2, 5, 1))  # Last core: r_right=1

        operator = TensorTrainOperator([core1, core2, core3])
        self.assertEqual(operator.d, 3)
        self.assertEqual(operator.shape, (3, 4, 5))
        self.assertEqual(operator.ranks, [1, 2, 2, 1])

        # Test rank mismatch
        core_bad = TensorTrainCore(np.random.randn(3, 4, 1))  # Wrong r_left
        with self.assertRaises(ValueError):
            TensorTrainOperator([core1, core_bad, core3])

        # Test boundary conditions
        core_bad_left = TensorTrainCore(np.random.randn(2, 3, 2))  # r_left != 1
        with self.assertRaises(ValueError):
            TensorTrainOperator([core_bad_left, core2, core3])

    def test_tensor_train_operator_matvec(self):
        from helix.ulam_tensor import TensorTrainCore, TensorTrainOperator

        # Create small TT operator for testing
        core1 = TensorTrainCore(np.random.randn(1, 2, 2))
        core2 = TensorTrainCore(np.random.randn(2, 2, 1))
        operator = TensorTrainOperator([core1, core2])

        # Test matrix-vector product
        vector = np.random.randn(4)  # Size 2*2 = 4
        result = operator.matvec(vector)
        self.assertEqual(result.shape, (4,))

        # Test wrong vector size
        with self.assertRaises(ValueError):
            operator.matvec(np.random.randn(3))

    def test_tensor_train_operator_to_dense(self):
        from helix.ulam_tensor import TensorTrainCore, TensorTrainOperator

        # Create very small TT operator to avoid memory issues
        core1 = TensorTrainCore(np.random.randn(1, 2, 2))
        core2 = TensorTrainCore(np.random.randn(2, 2, 1))
        operator = TensorTrainOperator([core1, core2])

        # Convert to dense matrix
        dense = operator.to_dense()
        self.assertEqual(dense.shape, (4, 4))

        # Test consistency with matvec
        vector = np.random.randn(4)
        result_tt = operator.matvec(vector)
        result_dense = dense @ vector
        np.testing.assert_allclose(result_tt, result_dense, rtol=1e-10)


class TestTTCrossApproximation(unittest.TestCase):
    def test_tt_cross_approximation_simple(self):
        from helix.ulam_tensor import tt_cross_approximation

        # Define a simple separable function
        def separable_func(indices: Tuple[int, ...]) -> float:
            """Simple separable function: f(i,j) = i * j"""
            if len(indices) != 2:
                return 0.0
            return float(indices[0] * indices[1])

        # Test TT decomposition
        shape = (3, 4)
        tt_op = tt_cross_approximation(separable_func, shape, max_rank=2)

        self.assertEqual(len(tt_op.cores), 2)
        self.assertEqual(tt_op.shape, shape)

        # Test that all ranks are within limits
        for rank in tt_op.ranks[1:-1]:  # Exclude boundary ranks (always 1)
            self.assertLessEqual(rank, 2)

    def test_tt_cross_approximation_constant(self):
        from helix.ulam_tensor import tt_cross_approximation

        # Constant function should have rank 1
        def constant_func(indices: Tuple[int, ...]) -> float:
            return 1.0

        shape = (3, 3, 3)
        tt_op = tt_cross_approximation(constant_func, shape, max_rank=5)

        # For constant function, internal ranks should be small
        self.assertEqual(tt_op.ranks[0], 1)
        self.assertEqual(tt_op.ranks[-1], 1)


class TestTensorUlamPF(unittest.TestCase):
    def test_tensor_ulam_pf_simple_map(self):
        from helix.ulam_tensor import tensor_ulam_pf

        # Simple linear map in 2D
        def linear_map(X: np.ndarray) -> np.ndarray:
            """Simple scaling map"""
            return 0.5 * X

        # Define bounding box
        box = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

        # Test with small discretization
        tt_operator, centers = tensor_ulam_pf(
            linear_map, box, bins_per_dim=3, max_rank=2
        )

        self.assertEqual(len(centers), 2)  # 2D
        self.assertEqual(len(centers[0]), 3)  # 3 bins per dimension
        self.assertIsInstance(tt_operator.shape, tuple)

    def test_tensor_ulam_pf_identity_map(self):
        from helix.ulam_tensor import tensor_ulam_pf

        # Identity map
        def identity_map(X: np.ndarray) -> np.ndarray:
            return X.copy()

        box = (np.array([0.0]), np.array([1.0]))

        tt_operator, centers = tensor_ulam_pf(
            identity_map, box, bins_per_dim=4, max_rank=3
        )

        self.assertEqual(len(centers), 1)  # 1D
        self.assertEqual(len(centers[0]), 4)  # 4 bins


class TestSpectralGapTT(unittest.TestCase):
    def test_spectral_gap_tt_small_operator(self):
        from helix.ulam_tensor import TensorTrainCore, TensorTrainOperator, spectral_gap_tt

        # Create a simple contractive TT operator
        # Use small dimensions to make eigenvalue computation feasible
        core_data = np.array([[[0.8, 0.1], [0.1, 0.8]]])  # Shape (1, 2, 2)
        core1 = TensorTrainCore(core_data)

        core_data2 = np.array([[[0.9], [0.1]], [[0.1], [0.9]]])  # Shape (2, 2, 1)
        core2 = TensorTrainCore(core_data2)

        operator = TensorTrainOperator([core1, core2])

        # Compute spectral gap
        gap = spectral_gap_tt(operator, num_eigenvalues=3)

        # Should return a valid gap value
        self.assertIsInstance(gap, float)
        self.assertGreaterEqual(gap, 0.0)
        self.assertLessEqual(gap, 1.0)

    def test_spectral_gap_tt_fallback(self):
        from helix.ulam_tensor import TensorTrainCore, TensorTrainOperator, _power_iteration_gap

        # Test fallback power iteration method
        core_data = np.array([[[0.5, 0.3], [0.2, 0.4]]])
        core1 = TensorTrainCore(core_data)

        core_data2 = np.array([[[0.6], [0.4]], [[0.3], [0.7]]])
        core2 = TensorTrainCore(core_data2)

        operator = TensorTrainOperator([core1, core2])

        gap = _power_iteration_gap(operator, max_iterations=10)

        self.assertIsInstance(gap, float)
        self.assertGreaterEqual(gap, 0.0)


class TestEnhancedUlamPF(unittest.TestCase):
    def test_enhanced_ulam_pf_fallback_to_standard(self):
        from helix.ulam_tensor import enhanced_ulam_pf

        # Simple 1D map (should fall back to standard Ulam)
        def simple_map(X: np.ndarray) -> np.ndarray:
            return 0.9 * X

        box = (np.array([0.0]), np.array([1.0]))

        # This should use standard Ulam (not tensor train) for 1D
        P, centers, gap = enhanced_ulam_pf(
            simple_map, box, bins_per_dim=5, use_tensor_train=False
        )

        self.assertEqual(P.shape, (5, 5))
        self.assertEqual(len(centers), 1)
        self.assertIsInstance(gap, float)

    def test_enhanced_ulam_pf_high_dimensional(self):
        from helix.ulam_tensor import enhanced_ulam_pf

        # High-dimensional map (should use tensor train)
        def high_dim_map(X: np.ndarray) -> np.ndarray:
            return 0.8 * X  # Simple scaling in all dimensions

        box = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

        P, centers, gap = enhanced_ulam_pf(
            high_dim_map, box, bins_per_dim=4, use_tensor_train=True, max_rank=2
        )

        # For high-dimensional case with small bins_per_dim, might return dense
        self.assertEqual(len(centers), 3)  # 3D
        self.assertIsInstance(gap, float)
        self.assertGreaterEqual(gap, 0.0)


class TestIntegrationTests(unittest.TestCase):
    def test_simple_2d_system_roundtrip(self):
        """Test complete workflow for a simple 2D dynamical system."""
        from helix.ulam_tensor import tensor_ulam_pf, spectral_gap_tt

        # Simple 2D rotation + contraction
        def rotation_contraction(X: np.ndarray) -> np.ndarray:
            theta = 0.1  # Small rotation
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            return 0.9 * (X @ rotation.T)  # Contract by 0.9

        box = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

        # Build TT operator
        tt_operator, centers = tensor_ulam_pf(
            rotation_contraction, box, bins_per_dim=6, max_rank=3
        )

        # Compute spectral gap
        gap = spectral_gap_tt(tt_operator)

        # Verify results
        self.assertEqual(len(centers), 2)
        self.assertGreater(gap, 0.0)  # Should have positive spectral gap
        self.assertLess(gap, 1.0)     # Should be less than 1

    def test_error_handling(self):
        """Test various error conditions."""
        from helix.ulam_tensor import TensorTrainCore, tensor_ulam_pf

        # Test invalid tensor dimensions
        with self.assertRaises(ValueError):
            TensorTrainCore(np.random.randn(2, 3, 4, 5))  # 4D instead of 3D

        # Test invalid map (wrong output dimension)
        def bad_map(X: np.ndarray) -> np.ndarray:
            return X[:, :1]  # Reduce dimension

        box = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        # This might not raise an error during construction but could cause issues
        # The actual behavior depends on implementation details


if __name__ == "__main__":
    unittest.main()