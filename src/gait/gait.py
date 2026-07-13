"""
Gait Generator Module for Quadruped Locomotion.

This module implements various gait patterns for quadruped robots,
including trot, walk, gallop, and bound. It generates foot swing
trajectories and manages the gait schedule for stance and swing phases.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
from utils.math_utils import cubic_bezier, smooth_step


class GaitType(Enum):
    """Gait pattern types."""
    TROT = "trot"          # Diagonal pairs move together
    WALK = "walk"          # Single-leg support
    GALLOP = "gallop"      # Asymmetric, flight phase
    BOUND = "bound"        # Fore and hind synchronous
    PRONK = "pronk"        # All legs together
    STAND = "stand"        # No locomotion


@dataclass
class GaitParams:
    """Parameters defining gait characteristics."""
    # Gait type
    gait_type: GaitType = GaitType.TROT

    # Timing parameters
    step_period: float = 0.4       # Full gait cycle period [s]
    stance_duration: float = 0.25  # Stance phase duration [s]
    swing_duration: float = 0.15   # Swing phase duration [s]

    # Footstep parameters
    foot_height: float = 0.08      # Maximum foot clearance during swing [m]
    foot_depth: float = 0.015      # Foot penetration during stance [m]

    # Terrain interaction
    ground_height: float = 0.0     # Ground plane height [m]
    friction: float = 0.7          # Ground friction coefficient

    # Contact schedule offsets for each leg
    # Order: FL(0), FR(1), RL(2), RR(3)
    contact_offsets: Tuple[float, float, float, float] = None

    def __post_init__(self):
        """Initialize default offsets based on gait type."""
        if self.contact_offsets is None:
            self.contact_offsets = self._default_offsets()

        # Validate timings
        self.phase_shift = self.stance_duration / self.step_period
        self.swing_shift = self.swing_duration / self.step_period

    def _default_offsets(self) -> Tuple[float, float, float, float]:
        """Get default contact offsets for the gait type."""
        if self.gait_type == GaitType.TROT:
            return (0.0, 0.5, 0.5, 0.0)  # Diagonal pairs
        elif self.gait_type == GaitType.WALK:
            return (0.0, 0.25, 0.5, 0.75)  # Sequential
        elif self.gait_type == GaitType.BOUND:
            return (0.0, 0.0, 0.5, 0.5)  # Fore-hind sync
        elif self.gait_type == GaitType.PRONK:
            return (0.0, 0.0, 0.0, 0.0)  # All together
        elif self.gait_type == GaitType.GALLOP:
            return (0.0, 0.25, 0.45, 0.7)  # Asymmetric
        else:
            return (0.0, 0.0, 0.0, 0.0)

    @property
    def duty_factor(self) -> float:
        """Fraction of gait cycle spent in stance."""
        return self.stance_duration / self.step_period

    @property
    def flight_factor(self) -> float:
        """Fraction of gait cycle spent in flight (no stance)."""
        return max(0.0, 1.0 - 2.0 * self.duty_factor)


@dataclass
class Footstep:
    """Single footstep information."""
    leg_id: int
    pos: np.ndarray          # Target position [3]
    vel: Optional[np.ndarray] = None  # Target velocity [3]
    phase: float = 0.0       # Phase in swing cycle [0, 1]


class SwingTrajectoryGenerator:
    """
    Generate smooth swing trajectories for feet.

    Uses Bezier curves to create natural-looking foot trajectories
    with configurable height and shape.
    """

    def __init__(self, foot_height: float = 0.08, toe_clearance: float = 0.02):
        """
        Initialize swing trajectory generator.

        Args:
            foot_height: Maximum foot height during swing [m]
            toe_clearance: Additional toe clearance at start/end
        """
        self.foot_height = foot_height
        self.toe_clearance = toe_clearance

    def generate(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        phase: float,
        ground_height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate swing trajectory point.

        Uses a modified Bezier curve that accounts for ground height
        and provides smooth velocity profiles.

        Args:
            start_pos: Starting foot position [3]
            end_pos: Target foot position [3]
            phase: Progress through swing [0, 1]
            ground_height: Ground plane height

        Returns:
            Tuple of (position, velocity) both [3]
        """
        # Normalize phase
        phase = np.clip(phase, 0.0, 1.0)

        # Height offset curve - peaks at mid-stance
        height_phase = 4.0 * phase * (1.0 - phase)  # Parabolic curve
        height_offset = self.foot_height * height_phase

        # Horizontal interpolation (5th order polynomial for smooth velocity)
        h_phase = cubic_bezier(phase)
        pos_xy = start_pos[:2] + h_phase * (end_pos[:2] - start_pos[:2])

        # Z interpolation
        z_start = ground_height + self.toe_clearance
        z_end = ground_height + self.toe_clearance
        z_max = ground_height + self.foot_height

        # Bezier curve for height
        if phase < 0.5:
            local_phase = phase * 2.0
            z = z_start + (z_max - z_start) * cubic_bezier(local_phase)
        else:
            local_phase = (phase - 0.5) * 2.0
            z = z_max - (z_max - z_end) * cubic_bezier(local_phase)

        # Compute velocity analytically (derivative of cubic Bezier)
        # dh_phase/dphase = derivative of cubic_bezier
        # cubic_bezier(t) = 3t^2(1-t), derivative = 6t(1-t) - 3t^2 = 6t - 9t^2
        dh_phase = 6.0 * phase * (1.0 - phase) - 3.0 * phase**2  # = 6*phase - 9*phase**2
        # For height: h = h_max * 4*t*(1-t), dh/dt = h_max * 4*(1-2t)
        dh = self.foot_height * 4.0 * (1.0 - 2.0 * phase)

        vel_xy = dh_phase * (end_pos[:2] - start_pos[:2])

        # Z velocity
        if phase < 0.5:
            local_phase = phase * 2.0
            # derivative of cubic_bezier(local_phase) * 2
            dbez = 6.0 * local_phase * (1.0 - local_phase) - 3.0 * local_phase**2
            dz = (z_max - z_start) * dbez * 2.0
        else:
            local_phase = (phase - 0.5) * 2.0
            dbez = 6.0 * local_phase * (1.0 - local_phase) - 3.0 * local_phase**2
            dz = -(z_max - z_end) * dbez * 2.0

        pos = np.array([pos_xy[0], pos_xy[1], z])
        vel = np.array([vel_xy[0], vel_xy[1], dz])

        return pos, vel

    def generate_bezier(
        self,
        start_pos: np.ndarray,
        mid_pos: np.ndarray,
        end_pos: np.ndarray,
        phase: float,
    ) -> np.ndarray:
        """
        Generate swing trajectory using cubic Bezier curve.

        Args:
            start_pos: Start position [3]
            mid_pos: Control point (peak) position [3]
            end_pos: End position [3]
            phase: Progress through swing [0, 1]

        Returns:
            Position [3] on the curve
        """
        t = phase
        t2 = t * t
        t3 = t2 * t
        mt = 1.0 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        return (mt3 * start_pos +
                3 * mt2 * t * mid_pos +
                3 * mt * t2 * end_pos +
                t3 * end_pos)


class FootTrajectory:
    """
    Manages foot trajectories including swing and stance phases.

    Provides complete foot trajectory including:
    - Swing trajectory with foot clearance
    - Stance phase with penetration control
    - Velocity and acceleration profiles
    """

    def __init__(self, gait_params: Optional[GaitParams] = None):
        """
        Initialize foot trajectory manager.

        Args:
            gait_params: Gait parameters
        """
        self.params = gait_params if gait_params is not None else GaitParams()
        self.swing_gen = SwingTrajectoryGenerator(
            foot_height=self.params.foot_height,
            toe_clearance=self.params.foot_depth
        )

        # Stance phase target (foot desired position during stance)
        self.stance_target = np.zeros(3)
        self.stance_vel = np.zeros(3)

    def get_foot_state(
        self,
        leg_id: int,
        phase: float,
        contact_state: bool,
        foot_pos: np.ndarray,
        com_state: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get desired foot position and velocity.

        Args:
            leg_id: Leg index (0-3)
            phase: Current phase in gait cycle [0, 1]
            contact_state: Whether foot is in contact
            foot_pos: Current foot position [3]
            com_state: (com_pos, com_vel) tuple

        Returns:
            Tuple of (desired_pos, desired_vel) both [3]
        """
        if contact_state:
            # In stance: maintain current contact point with velocity
            return self.stance_target.copy(), self.stance_vel.copy()
        else:
            # In swing: follow swing trajectory
            return self.swing_gen.generate(
                foot_pos,
                self.stance_target,
                phase,
                self.params.ground_height
            )

    def update_stance_target(
        self,
        leg_id: int,
        desired_pos: np.ndarray,
        desired_vel: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update the stance target position for a foot.

        Args:
            leg_id: Leg index
            desired_pos: Desired foot position during stance [3]
            desired_vel: Desired foot velocity during stance [3]
        """
        self.stance_target = desired_pos.copy()
        if desired_vel is not None:
            self.stance_vel = desired_vel.copy()


class GaitScheduler:
    """
    Gait schedule manager for multiple legs.

    Handles the timing of stance and swing phases for all legs
    based on the current gait type and phase.
    """

    def __init__(self, gait_params: Optional[GaitParams] = None):
        """
        Initialize gait scheduler.

        Args:
            gait_params: Gait parameters
        """
        self.params = gait_params if gait_params is not None else GaitParams()
        self.leg_count = 4
        self._phase = 0.0  # Current phase [0, 1]

    def reset(self) -> None:
        """Reset gait phase to zero."""
        self._phase = 0.0

    def update(self, dt: float) -> float:
        """
        Update gait phase.

        Args:
            dt: Time step [s]

        Returns:
            Current phase [0, 1]
        """
        self._phase += dt / self.params.step_period
        self._phase %= 1.0
        return self._phase

    def get_contact_schedule(self, phase: Optional[float] = None) -> np.ndarray:
        """
        Get contact state for all legs at given phase.

        Args:
            phase: Gait phase (uses current if None) [0, 1]

        Returns:
            Contact states [4] as boolean array
        """
        if phase is None:
            phase = self._phase

        contacts = np.zeros(self.leg_count, dtype=bool)
        for i in range(self.leg_count):
            # Normalize phase for this leg
            leg_phase = (phase + self.params.contact_offsets[i]) % 1.0

            # In stance if phase is within stance duration
            contacts[i] = leg_phase < self.params.phase_shift

        return contacts

    def get_swing_phase(self, leg_id: int, phase: Optional[float] = None) -> float:
        """
        Get swing phase progress for a specific leg.

        Args:
            leg_id: Leg index
            phase: Gait phase

        Returns:
            Swing phase [0, 1] if in swing, -1 if in stance
        """
        if phase is None:
            phase = self._phase

        contact = self.get_contact_schedule(phase)

        if contact[leg_id]:
            return -1.0  # In stance

        # Compute swing phase
        leg_phase = (phase + self.params.contact_offsets[leg_id]) % 1.0
        return leg_phase / self.params.phase_shift

    def get_stance_phase(self, leg_id: int, phase: Optional[float] = None) -> float:
        """
        Get stance phase progress for a specific leg.

        Args:
            leg_id: Leg index
            phase: Gait phase

        Returns:
            Stance phase [0, 1] if in stance, -1 if in swing
        """
        if phase is None:
            phase = self._phase

        contact = self.get_contact_schedule(phase)

        if not contact[leg_id]:
            return -1.0  # In swing

        leg_phase = (phase + self.params.contact_offsets[leg_id]) % 1.0
        return leg_phase / self.params.phase_shift


class GaitGenerator:
    """
    Main gait generation interface.

    Combines the scheduler and trajectory generator to provide
    a complete interface for foot trajectory generation.
    """

    def __init__(self, gait_params: Optional[GaitParams] = None):
        """
        Initialize gait generator.

        Args:
            gait_params: Gait parameters
        """
        self.params = gait_params if gait_params is not None else GaitParams()
        self.scheduler = GaitScheduler(self.params)
        self.trajectory = FootTrajectory(self.params)
        self.swing_gen = SwingTrajectoryGenerator(
            foot_height=self.params.foot_height
        )

        # Current state tracking
        self._phase = 0.0
        self._contact_states = np.zeros(4, dtype=bool)
        self._foot_positions = np.zeros((4, 3))
        self._foot_velocities = np.zeros((4, 3))

    def reset(self) -> None:
        """Reset gait generator to initial state."""
        self.scheduler.reset()
        self._phase = 0.0

    def step(self, dt: float) -> None:
        """
        Advance gait by one time step.

        Args:
            dt: Time step [s]
        """
        self._phase = self.scheduler.update(dt)
        self._contact_states = self.scheduler.get_contact_schedule()

    def update_foot_positions(self, foot_positions: np.ndarray) -> None:
        """
        Update current foot positions from sensor data.

        Args:
            foot_positions: Current foot positions [4, 3]
        """
        self._foot_positions = foot_positions.copy()

    def compute_desired_foot_state(
        self,
        com_state: Tuple[np.ndarray, np.ndarray],
        base_vel: np.ndarray,
        com_acc: Optional[np.ndarray] = None,
        foot_positions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute desired foot states for all legs.

        Uses current foot positions as reference, with simple
        Raibert-like velocity compensation for swing legs.

        Args:
            com_state: (com_pos, com_vel) tuple
            base_vel: Base linear velocity in world frame [3]
            com_acc: Optional CoM acceleration [3]
            foot_positions: Current foot positions [4,3]

        Returns:
            Tuple of (foot_pos_desired [4,3], foot_vel_desired [4,3], contacts [4])
        """
        com_pos, com_vel = com_state
        foot_pos_des = np.zeros((4, 3))
        foot_vel_des = np.zeros((4, 3))

        # Default standing foot offset (simple - use actual foot positions)
        for i in range(4):
            if foot_positions is not None:
                # Use current foot position as desired (stance)
                foot_pos_des[i] = foot_positions[i].copy()
            else:
                # Fallback: fixed foot position below hip
                foot_pos_des[i] = com_pos + np.array([0.2 if i < 2 else -0.2,
                                                       0.08 if i % 2 == 0 else -0.08,
                                                       -0.35])
            foot_vel_des[i] = np.zeros(3)

        return foot_pos_des, foot_vel_des, self._contact_states

    def get_swing_trajectory(
        self,
        leg_id: int,
        phase: float,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get swing trajectory for specific leg.

        Args:
            leg_id: Leg index
            phase: Swing phase [0, 1]
            start_pos: Swing start position
            end_pos: Swing end position

        Returns:
            Tuple of (position, velocity)
        """
        return self.swing_gen.generate(
            start_pos,
            end_pos,
            phase,
            self.params.ground_height
        )

    @property
    def phase(self) -> float:
        """Current gait phase [0, 1]."""
        return self._phase

    @property
    def contact_states(self) -> np.ndarray:
        """Current contact states for all legs."""
        return self._contact_states

    @property
    def has_flight_phase(self) -> bool:
        """Whether current gait has a flight phase."""
        return self.params.flight_factor > 0

    @property
    def stance_time(self) -> float:
        """Duration of stance phase [s]."""
        return self.params.step_period * self.params.duty_factor

    @property
    def swing_time(self) -> float:
        """Duration of swing phase [s]."""
        return self.params.step_period * (1.0 - self.params.duty_factor)
