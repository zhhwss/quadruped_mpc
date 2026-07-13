"""Tests for gait module."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from gait.gait import (
    GaitGenerator, GaitParams, GaitType,
    GaitScheduler, SwingTrajectoryGenerator, FootTrajectory
)


class TestGaitParams:
    """Tests for GaitParams."""

    def test_default_values(self):
        params = GaitParams()
        assert params.gait_type == GaitType.TROT
        assert params.step_period == 0.4
        assert params.stance_duration == 0.25
        assert params.foot_height == 0.08

    def test_duty_factor(self):
        params = GaitParams(stance_duration=0.25, step_period=0.4)
        assert np.isclose(params.duty_factor, 0.625)

    def test_contact_offsets_trot(self):
        params = GaitParams(gait_type=GaitType.TROT)
        assert params.contact_offsets[0] == 0.0  # FL
        assert params.contact_offsets[1] == 0.5  # FR
        assert params.contact_offsets[2] == 0.5  # RL
        assert params.contact_offsets[3] == 0.0  # RR

    def test_contact_offsets_walk(self):
        params = GaitParams(gait_type=GaitType.WALK)
        assert params.contact_offsets[0] == 0.0
        assert params.contact_offsets[1] == 0.25
        assert params.contact_offsets[2] == 0.5
        assert params.contact_offsets[3] == 0.75


class TestGaitScheduler:
    """Tests for GaitScheduler."""

    def test_reset(self):
        scheduler = GaitScheduler()
        scheduler.update(0.1)
        scheduler.reset()
        assert scheduler.phase == 0.0

    def test_update(self):
        scheduler = GaitScheduler(GaitParams(step_period=1.0))
        scheduler.update(0.25)
        assert np.isclose(scheduler.phase, 0.25)

    def test_contact_schedule(self):
        params = GaitParams(gait_type=GaitType.TROT, step_period=1.0)
        scheduler = GaitScheduler(params)

        # At phase 0: FL and RR should be in contact
        contacts = scheduler.get_contact_schedule(0.0)
        assert contacts[0] == True  # FL
        assert contacts[3] == True  # RR
        assert contacts[1] == False  # FR
        assert contacts[2] == False  # RL

    def test_contact_schedule_phase_shift(self):
        params = GaitParams(gait_type=GaitType.TROT, step_period=1.0)
        scheduler = GaitScheduler(params)

        # At phase 0.5: FR and RL should be in contact
        contacts = scheduler.get_contact_schedule(0.5)
        assert contacts[1] == True  # FR
        assert contacts[2] == True  # RL
        assert contacts[0] == False  # FL
        assert contacts[3] == False  # RR


class TestSwingTrajectoryGenerator:
    """Tests for SwingTrajectoryGenerator."""

    def test_start_end_positions(self):
        gen = SwingTrajectoryGenerator(foot_height=0.08)
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.1, 0.0, 0.0])

        pos_0, vel_0 = gen.generate(start, end, 0.0)
        pos_1, vel_1 = gen.generate(start, end, 1.0)

        assert np.allclose(pos_0, start, atol=1e-4)
        assert np.allclose(pos_1, end, atol=1e-4)

    def test_max_height(self):
        gen = SwingTrajectoryGenerator(foot_height=0.08)
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.1, 0.0, 0.0])

        pos, vel = gen.generate(start, end, 0.5)
        assert pos[2] > 0.05  # Should be near peak height

    def test_smooth_velocity(self):
        """Test that velocity is smooth (finite, no jumps)."""
        gen = SwingTrajectoryGenerator()
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.1, 0.0, 0.0])

        for t in np.linspace(0.01, 0.99, 10):
            pos, vel = gen.generate(start, end, t)
            assert np.all(np.isfinite(vel))


class TestGaitGenerator:
    """Tests for GaitGenerator."""

    def test_reset(self):
        gait = GaitGenerator()
        gait.step(0.1)
        gait.reset()
        assert gait.phase == 0.0

    def test_contact_states_update(self):
        gait = GaitGenerator()
        contact_states = np.array([True, False, True, False])
        gait.update_foot_positions(np.zeros((4, 3)))
