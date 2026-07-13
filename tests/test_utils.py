"""Tests for math utilities."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.math_utils import (
    cubic_bezier, smooth_step, hat, vee, exp_so3, log_so3,
    quaternion_to_rpy, rpy_to_quaternion, low_pass_filter,
    smc, saturate, pinv_svd
)


class TestBasicFunctions:
    """Tests for basic math utilities."""

    def test_cubic_bezier(self):
        assert np.isclose(cubic_bezier(0.0), 0.0)
        assert np.isclose(cubic_bezier(1.0), 0.0)
        assert cubic_bezier(0.5) > 0.2

    def test_smooth_step(self):
        assert np.isclose(smooth_step(0.0), 0.0)
        assert np.isclose(smooth_step(1.0), 1.0)
        assert np.isclose(smooth_step(0.5), 0.5)

    def test_hat_operator(self):
        v = np.array([1.0, 2.0, 3.0])
        S = hat(v)
        assert S.shape == (3, 3)
        assert np.allclose(S.T, -S)  # Skew-symmetric

    def test_vee_operator(self):
        v = np.array([1.0, 2.0, 3.0])
        S = hat(v)
        v_recovered = vee(S)
        assert np.allclose(v, v_recovered)


class TestRotationFunctions:
    """Tests for rotation utilities."""

    def test_quaternion_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        rpy = quaternion_to_rpy(q)
        assert np.allclose(rpy, 0.0, atol=1e-6)

    def test_roundtrip_rpy_to_quat(self):
        rpy = np.array([0.1, 0.2, 0.3])
        q = rpy_to_quaternion(rpy)
        rpy_recovered = quaternion_to_rpy(q)
        assert np.allclose(rpy, rpy_recovered, atol=1e-4)

    def test_exp_so3_identity(self):
        omega = np.zeros(3)
        R = exp_so3(omega)
        assert np.allclose(R, np.eye(3))

    def test_log_so3_identity(self):
        R = np.eye(3)
        omega = log_so3(R)
        assert np.allclose(omega, 0.0, atol=1e-6)


class TestControlFunctions:
    """Tests for control utilities."""

    def test_low_pass_filter(self):
        current = np.array([1.0, 2.0, 3.0])
        previous = np.zeros(3)
        result = low_pass_filter(current, previous, cutoff=10.0, dt=0.01)
        # At first step, should be between 0 and current
        assert np.all(result > 0)
        assert np.all(result < current + 1e-10)

    def test_smc(self):
        # Test sliding mode control
        x = np.array([1.0])
        result = smc(x, s=1.0, phi=0.01)
        assert result[0] > 0  # Should be positive for positive x

    def test_saturate(self):
        x = np.array([-10.0, 0.0, 10.0])
        result = saturate(x, (-1.0, 1.0))
        assert np.allclose(result, [-1.0, 0.0, 1.0])

    def test_pinv_svd(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        A_pinv = pinv_svd(A)
        # Check pseudoinverse property
        product = A @ A_pinv @ A
        assert np.allclose(product, A, atol=1e-6)
