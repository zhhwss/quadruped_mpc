"""Tests for WBC module."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from wbc.wbc import (
    WBCController, WBCParams, OperationalSpaceController,
    WBCHierarchy
)


class TestWBCParams:
    """Tests for WBCParams."""

    def test_default_values(self):
        params = WBCParams()
        assert params.control_freq == 500.0
        assert params.com_pos_kp == 200.0
        assert params.com_pos_kd == 20.0
        assert params.foot_kp == 800.0

    def test_joint_limits_set(self):
        params = WBCParams()
        assert params.joint_pos_limits is not None
        assert params.joint_vel_limits is not None
        assert params.joint_torque_limits is not None


class TestOperationalSpaceController:
    """Tests for OperationalSpaceController."""

    def test_compute_joint_torques(self):
        osc = OperationalSpaceController()

        J = np.eye(3, 12)
        task_acc = np.zeros(3)
        M = np.eye(12)
        C = np.zeros(12)
        G = np.zeros(12)
        qdot = np.zeros(12)

        tau = osc.compute_joint_torques(J, task_acc, M, C, G, qdot)

        assert tau.shape == (12,)


class TestWBCHierarchy:
    """Tests for WBCHierarchy."""

    def test_solve_task_priority(self):
        hierarchy = WBCHierarchy()

        J_list = [np.eye(3, 12)]
        task_acc_list = [np.zeros(3)]
        M = np.eye(12)
        C = np.zeros(12)
        G = np.zeros(12)
        qdot = np.zeros(12)

        tau, info = hierarchy.solve_task_priority(
            J_list, task_acc_list, M, C, G, qdot
        )

        assert tau.shape == (12,)
        assert isinstance(info, dict)


class TestWBCController:
    """Tests for WBCController."""

    def test_initialization(self):
        wbc = WBCController(n_joints=12)
        assert wbc.n_joints == 12

    def test_compute_joint_torques_basic(self):
        wbc = WBCController(n_joints=12)

        torques, info = wbc.compute_joint_torques(
            com_pos=np.array([0.0, 0.0, 0.3]),
            com_vel=np.zeros(3),
            com_rpy=np.zeros(3),
            com_ang_vel=np.zeros(3),
            foot_positions=np.zeros((4, 3)),
            foot_velocities=np.zeros((4, 3)),
            foot_desired_positions=np.zeros((4, 3)),
            foot_desired_velocities=np.zeros((4, 3)),
            contact_states=np.array([True, True, True, True]),
            joint_positions=np.zeros(12),
            joint_velocities=np.zeros(12),
        )

        assert torques.shape == (12,)
        assert np.all(np.isfinite(torques))

    def test_compute_joint_torques_all_swing(self):
        wbc = WBCController(n_joints=12)

        foot_pos = np.array([
            [0.2, 0.1, -0.3],
            [0.2, -0.1, -0.3],
            [-0.2, 0.1, -0.3],
            [-0.2, -0.1, -0.3],
        ])

        torques, info = wbc.compute_joint_torques(
            com_pos=np.array([0.0, 0.0, 0.3]),
            com_vel=np.zeros(3),
            com_rpy=np.zeros(3),
            com_ang_vel=np.zeros(3),
            foot_positions=foot_pos,
            foot_velocities=np.zeros((4, 3)),
            foot_desired_positions=foot_pos + 0.01,  # Small offset
            foot_desired_velocities=np.zeros((4, 3)),
            contact_states=np.zeros(4, dtype=bool),  # All swing
            joint_positions=np.zeros(12),
            joint_velocities=np.zeros(12),
        )

        assert torques.shape == (12,)

    def test_clip_torques(self):
        wbc = WBCController(n_joints=12)

        # Torque that exceeds limits
        torques = np.full(12, 100.0)
        joint_vel = np.zeros(12)

        clipped = wbc.clip_torques(torques, joint_vel)

        assert np.all(clipped <= 35.0)
        assert np.all(clipped >= -35.0)

    def test_compute_contact_forces(self):
        wbc = WBCController(n_joints=12)

        contact_states = np.array([True, True, False, False])
        foot_positions = np.zeros((4, 3))

        forces = wbc.compute_contact_forces(foot_positions, contact_states)

        assert forces.shape == (4, 3)
        assert np.all(forces[2:, :] == 0.0)  # Swing legs have zero force
        assert np.all(forces[:2, 2] > 0)  # Normal forces are positive
