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
    """Parameters defining gait characteristics.

    Defaults match Quadruped-PyMPC trot: duty_factor=0.65, step_freq=1.4 Hz.
    step_period = 1/step_freq = 0.714s; stance = 0.65*0.714 = 0.464s.
    """
    # Gait type
    gait_type: GaitType = GaitType.TROT

    # Primary timing control (step_freq is preferred)
    step_freq: float = 1.4        # Step frequency [Hz] (primary control)
    step_period: float = None     # Full gait cycle period [s] = 1/step_freq
    duty_factor: float = 0.65     # Fraction of cycle in stance (stable trot)
    stance_duration: float = None  # [s], derived if None
    swing_duration: float = None   # [s], derived if None

    # Footstep parameters
    foot_height: float = 0.08      # Maximum foot clearance during swing [m]
    foot_depth: float = 0.015      # Foot penetration during stance [m]

    # Terrain interaction
    ground_height: float = 0.0     # Ground plane height [m]
    friction: float = 0.7          # Ground friction coefficient

    # Contact schedule offsets for each leg
    contact_offsets: Tuple[float, float, float, float] = None

    def __post_init__(self):
        """Initialize default offsets and derive timing."""
        if self.contact_offsets is None:
            self.contact_offsets = self._default_offsets()

        # Derive step_period from step_freq if not explicitly set
        if self.step_period is None:
            self.step_period = 1.0 / self.step_freq

        # Derive stance/swing durations
        if self.stance_duration is None:
            self.stance_duration = self.step_period * self.duty_factor
        if self.swing_duration is None:
            self.swing_duration = self.step_period * (1.0 - self.duty_factor)

        self.phase_shift = self.duty_factor
        self.swing_shift = 1.0 - self.duty_factor

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
    def duty_factor_prop(self) -> float:
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
        Generate swing trajectory from start_pos to end_pos.

        XY: cubic Bezier interpolation (zero velocity at endpoints)
        Z:  parabolic arc from start_z, peaking at max(start_z,end_z)+foot_height,
            landing at end_z.

        Args:
            start_pos: Starting foot position [3] (current foot pos)
            end_pos: Target foot position [3] (Raibert landing pos)
            phase: Progress through swing [0, 1]
            ground_height: Ground plane height

        Returns:
            Tuple of (position, velocity) both [3]
        """
        phase = np.clip(phase, 0.0, 1.0)

        # XY: cubic Bezier (smooth, zero endpoint velocity)
        h_phase = cubic_bezier(phase)
        pos_xy = start_pos[:2] + h_phase * (end_pos[:2] - start_pos[:2])
        dh_phase = 6.0 * phase - 6.0 * phase**2
        vel_xy = dh_phase * (end_pos[:2] - start_pos[:2])

        # Z: arc from start_z -> peak -> end_z
        z_start = start_pos[2]
        z_end = end_pos[2]
        z_peak = max(z_start, z_end, ground_height) + self.foot_height

        z = z_start + (z_end - z_start) * phase + \
            (z_peak - (z_start + z_end) / 2.0) * 4.0 * phase * (1.0 - phase)

        dz = (z_end - z_start) + \
             (z_peak - (z_start + z_end) / 2.0) * 4.0 * (1.0 - 2.0 * phase)

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
        self.params = gait_params if gait_params is not None else GaitParams()
        self.swing_gen = SwingTrajectoryGenerator(
            foot_height=self.params.foot_height,
            toe_clearance=self.params.foot_depth
        )
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
        """Get desired foot position and velocity."""
        if contact_state:
            return self.stance_target.copy(), self.stance_vel.copy()
        else:
            return self.swing_gen.generate(
                foot_pos, self.stance_target, phase, self.params.ground_height
            )

    def update_stance_target(
        self,
        leg_id: int,
        desired_pos: np.ndarray,
        desired_vel: Optional[np.ndarray] = None,
    ) -> None:
        """Update the stance target position for a foot."""
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
        self.params = gait_params if gait_params is not None else GaitParams()
        self.leg_count = 4
        self._phase = 0.0

    def reset(self) -> None:
        self._phase = 0.0

    def update(self, dt: float) -> float:
        """Advance gait phase by dt using step_freq."""
        self._phase += dt * self.params.step_freq
        self._phase %= 1.0
        return self._phase

    def get_contact_schedule(self, phase: Optional[float] = None) -> np.ndarray:
        """Get contact state for all legs at given phase."""
        if phase is None:
            phase = self._phase

        contacts = np.zeros(self.leg_count, dtype=bool)
        for i in range(self.leg_count):
            leg_phase = (phase + self.params.contact_offsets[i]) % 1.0
            contacts[i] = leg_phase < self.params.duty_factor

        return contacts

    def get_swing_phase(self, leg_id: int, phase: Optional[float] = None) -> float:
        """Get swing phase progress for a specific leg. Returns -1 if in stance."""
        if phase is None:
            phase = self._phase

        contact = self.get_contact_schedule(phase)
        if contact[leg_id]:
            return -1.0

        leg_phase = (phase + self.params.contact_offsets[leg_id]) % 1.0
        swing_duration = 1.0 - self.params.duty_factor
        return (leg_phase - self.params.duty_factor) / swing_duration

    def get_stance_phase(self, leg_id: int, phase: Optional[float] = None) -> float:
        """Get stance phase progress for a specific leg. Returns -1 if in swing."""
        if phase is None:
            phase = self._phase

        contact = self.get_contact_schedule(phase)
        if not contact[leg_id]:
            return -1.0

        leg_phase = (phase + self.params.contact_offsets[leg_id]) % 1.0
        return leg_phase / self.params.duty_factor


class GaitGenerator:
    """
    Main gait generation interface.

    Combines the scheduler and trajectory generator to provide
    a complete interface for foot trajectory generation.
    """

    def __init__(self, gait_params: Optional[GaitParams] = None):
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
        self.scheduler.reset()
        self._phase = 0.0
        self._contact_states = np.zeros(4, dtype=bool)

    def step(self, dt: float) -> None:
        """Advance gait by one time step."""
        self._phase = self.scheduler.update(dt)
        self._contact_states = self.scheduler.get_contact_schedule()

    def update_foot_positions(self, foot_positions: np.ndarray) -> None:
        self._foot_positions = foot_positions.copy()

    def compute_desired_foot_state(
        self,
        com_state: Tuple[np.ndarray, np.ndarray],
        base_vel: np.ndarray,
        com_acc: Optional[np.ndarray] = None,
        foot_positions: Optional[np.ndarray] = None,
        command: Optional[np.ndarray] = None,
        base_pos: Optional[np.ndarray] = None,
        base_rpy: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute desired foot states for all legs using Raibert heuristic.

        p_foot_des_xy = p_hip_xy + v_cmd * T_stance / 2 + k_raibert * (v_actual - v_cmd)
        """
        com_pos, com_vel = com_state
        foot_pos_des = np.zeros((4, 3))
        foot_vel_des = np.zeros((4, 3))

        # Hip positions in base frame
        hip_offsets = np.array([
            [ 0.25944,  0.075113, 0.0],   # FL
            [ 0.25944, -0.075113, 0.0],   # FR
            [-0.25944,  0.075113, 0.0],   # RL
            [-0.25944, -0.075113, 0.0],   # RR
        ])

        # Default foot offsets from hip in standing pose (base frame)
        default_foot_offsets = np.array([
            [ 0.0,  0.13, -0.35],   # FL
            [ 0.0, -0.13, -0.35],   # FR
            [ 0.0,  0.13, -0.35],   # RL
            [ 0.0, -0.13, -0.35],   # RR
        ])

        T_stance = self.params.stance_duration
        k_raibert = 0.03   # mild feedback for velocity correction

        if command is None:
            cmd_vx, cmd_vy = 0.0, 0.0
        else:
            cmd_vx, cmd_vy = command[0], command[1]

        actual_vx = base_vel[0]
        actual_vy = base_vel[1]

        # Raibert in body frame
        # Raibert offset: place foot ahead by half stance travel + feedback velocity correction
        raibert_body_x = cmd_vx * T_stance / 2.0 + k_raibert * (actual_vx - cmd_vx)
        raibert_body_y = cmd_vy * T_stance / 2.0 + k_raibert * (actual_vy - cmd_vy)

        # Kinematic safety clipping: prevent target from exceeding leg workspace limit (max 0.15m)
        raibert_norm = np.sqrt(raibert_body_x**2 + raibert_body_y**2)
        max_raibert = 0.15
        if raibert_norm > max_raibert:
            scale = max_raibert / raibert_norm
            raibert_body_x *= scale
            raibert_body_y *= scale

        # body->world rotation
        if base_rpy is not None:
            roll, pitch, yaw = base_rpy
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            R_body_to_world = np.array([
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp,   cp*sr,            cp*cr],
            ])
        else:
            R_body_to_world = np.eye(3)

        if base_pos is None:
            base_pos = com_pos

        raibert_body = np.array([raibert_body_x, raibert_body_y, 0.0])
        raibert_world = R_body_to_world @ raibert_body

        for i in range(4):
            hip_world = base_pos + R_body_to_world @ hip_offsets[i]
            default_foot_world = base_pos + R_body_to_world @ (hip_offsets[i] + default_foot_offsets[i])

            foot_pos_des[i, 0] = default_foot_world[0] + raibert_world[0]
            foot_pos_des[i, 1] = default_foot_world[1] + raibert_world[1]
            foot_pos_des[i, 2] = self.params.ground_height + 0.005

            # Blend current pos for stance legs to avoid jumps
            if self._contact_states[i] and foot_positions is not None:
                foot_pos_des[i, 0] = 0.9 * foot_positions[i, 0] + 0.1 * foot_pos_des[i, 0]
                foot_pos_des[i, 1] = 0.9 * foot_positions[i, 1] + 0.1 * foot_pos_des[i, 1]
                foot_pos_des[i, 2] = foot_positions[i, 2]

            if not self._contact_states[i]:
                foot_vel_des[i, 0] = cmd_vx
                foot_vel_des[i, 1] = cmd_vy
            foot_vel_des[i, 2] = 0.0

        return foot_pos_des, foot_vel_des, self._contact_states

    def get_swing_trajectory(
        self,
        leg_id: int,
        phase: float,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get swing trajectory for specific leg."""
        return self.swing_gen.generate(
            start_pos, end_pos, phase, self.params.ground_height
        )

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def contact_states(self) -> np.ndarray:
        return self._contact_states

    @property
    def has_flight_phase(self) -> bool:
        return self.params.flight_factor > 0

    @property
    def stance_time(self) -> float:
        return self.params.step_period * self.params.duty_factor

    @property
    def swing_time(self) -> float:
        return self.params.step_period * (1.0 - self.params.duty_factor)
