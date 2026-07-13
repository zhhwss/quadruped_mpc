"""Tests for robot module."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from robot.robot import A1Robot, RobotParams, RobotState


class TestRobotParams:
    """Tests for RobotParams."""

    def test_default_values(self):
        params = RobotParams()
        assert params.mass == 12.0
        assert params.upper_leg_length == 0.213
        assert params.lower_leg_length == 0.213
        assert params.hip_positions is not None
        assert len(params.hip_positions) == 4

    def test_n_joints(self):
        params = RobotParams()
        assert params.n_joints == 12
        assert params.n_legs == 4


class TestRobotState:
    """Tests for RobotState."""

    def test_default_initialization(self):
        state = RobotState()
        assert len(state.base_pos) == 3
        assert len(state.joint_pos) == 12
        assert len(state.foot_pos) == 4
        assert np.all(state.contact_state == False)

    def test_custom_initialization(self):
        state = RobotState(
            base_pos=np.array([1.0, 0.5, 0.3]),
            joint_pos=np.zeros(12),
        )
        assert state.base_pos[0] == 1.0
        assert state.base_pos[2] == 0.3


class TestRobotMath:
    """Tests for robot math utilities."""

    def test_rotation_matrix_from_quat(self):
        # Identity quaternion
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = A1Robot.rotation_matrix_from_quat(q)
        assert np.allclose(R, np.eye(3))

    def test_quat_to_rpy(self):
        # Identity
        q = np.array([1.0, 0.0, 0.0, 0.0])
        rpy = A1Robot.rpy_from_quat(q)
        assert np.allclose(rpy, 0.0, atol=1e-6)

    def test_rpy_to_quat(self):
        # Zero RPY
        rpy = np.array([0.0, 0.0, 0.0])
        q = A1Robot.quat_from_rotation_matrix(np.eye(3))
        assert np.allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
