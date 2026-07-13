"""Tests for MPC module."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from mpc.mpc import MPCController, MPCParams, CentroidalDynamics, FrictionCone


class TestCentroidalDynamics:
    """Tests for CentroidalDynamics."""

    def test_initialization(self):
        dyn = CentroidalDynamics()
        assert dyn.mass == 12.0
        assert dyn.inertia is not None

    def test_build_state_matrix(self):
        dyn = CentroidalDynamics()
        A, B = dyn.build_state_matrix(dt=0.01)

        assert A.shape == (12, 12)
        assert B.shape == (12, 12)

    def test_friction_cone_constraints(self):
        friction = FrictionCone(mu=0.7)

        # These are just structure tests since we can't easily
        # test CVXPY constraints without solving

        # Verify mu is set correctly
        assert friction.mu == 0.7


class TestMPCController:
    """Tests for MPCController."""

    def test_initialization(self):
        mpc = MPCController()
        assert mpc.params.horizon_steps == 10
        assert mpc.params.dt == 0.025

    def test_compute_centroidal_state(self):
        mpc = MPCController()
        state = mpc.compute_centroidal_state(
            com_pos=np.array([0.0, 0.0, 0.3]),
            com_vel=np.zeros(3),
            rpy=np.zeros(3),
            ang_vel=np.zeros(3),
        )
        assert len(state) == 12
        assert np.isclose(state[2], 0.3)

    def test_set_reference(self):
        mpc = MPCController()
        mpc.set_reference(
            com_pos=np.array([0.0, 0.0, 0.3]),
            com_vel=np.zeros(3),
        )
        assert mpc._com_ref is not None

    def test_solve_with_no_contacts(self):
        mpc = MPCController()
        state = np.zeros(12)
        foot_pos = np.zeros((4, 3))
        contact_states = np.zeros(4, dtype=bool)

        forces, foot_pos_out = mpc.solve(state, foot_pos, contact_states)

        assert forces.shape == (4, 3)
        assert np.allclose(forces, 0.0)

    def test_solve_with_contacts(self):
        mpc = MPCController()
        state = np.zeros(12)
        foot_pos = np.array([
            [0.2, 0.1, -0.3],
            [0.2, -0.1, -0.3],
            [-0.2, 0.1, -0.3],
            [-0.2, -0.1, -0.3],
        ])
        contact_states = np.array([True, True, True, True])

        forces, foot_pos_out = mpc.solve(state, foot_pos, contact_states)

        assert forces.shape == (4, 3)
        # Should have positive normal forces
        assert np.all(forces[:, 2] >= 0)

    def test_reset(self):
        mpc = MPCController()
        mpc._last_solution = (np.zeros((4, 3)), np.zeros((4, 3)))
        mpc.reset()
        assert mpc._last_solution is None


class TestFrictionCone:
    """Tests for FrictionCone."""

    def test_default_mu(self):
        fc = FrictionCone()
        assert fc.mu == 0.7

    def test_custom_mu(self):
        fc = FrictionCone(mu=0.5)
        assert fc.mu == 0.5
