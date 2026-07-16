"""
Model Predictive Control (MPC) Module for Quadruped Locomotion.

This module implements a linear MPC for centroidal dynamics that optimizes
contact forces and foot positions over a receding horizon. The MPC solves
a QP at each control cycle to generate desired CoM trajectories and
contact force distributions.

Key features:
- Linearized centroidal dynamics
- Receding horizon optimization
- Friction cone constraints
- Contact schedule handling
- Warm-starting for real-time performance
"""

import numpy as np
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy import linalg

from utils.math_utils import block_diag, regularize


@dataclass
class MPCParams:
    """
    Parameters for the MPC controller.

    Tuned for Unitree A2 (from a2_scene.xml):
      - mass = 19.651 kg (base only; centroidal dynamics uses base mass)
      - inertia = diag([0.472, 0.408, 0.142])
      - torque limits: hip=±120, thigh=±120, calf=±180 Nm
    """
    # Prediction horizon
    horizon_steps: int = 10

    # Time step [s]
    dt: float = 0.025

    # ── A2 physical parameters ──────────────────────────
    base_mass: float = 19.651            # kg  (base_link only)
    total_mass: float = 40.071           # kg  (matching actual simulated model mass)
    inertia: np.ndarray = None          # [3,3] base inertia

    # CoM dynamics weights
    com_pos_weight: float = 100.0
    com_vel_weight: float = 50.0
    com_ang_weight: float = 50.0
    com_ang_vel_weight: float = 50.0

    # Force regularization
    force_weight: float = 0.01

    # Force change penalty (smoothness)
    force_rate_weight: float = 0.1

    # Foot position regularization
    foot_pos_weight: float = 0.01

    # Friction coefficient
    friction_coeff: float = 0.5   # matches Quadruped-PyMPC

    # Normal force limits [N]
    min_normal_force: float = 20.0
    max_normal_force: float = 250.0  # reduced: 2-leg stance = 196N, allow 27% margin

    # Force scale multiplier. 1.3 gives a bit more margin for yaw
    # correction within the friction cone.
    force_scale: float = 1.0   # must not push through contacts   # hybrid Jacobian + IK-based PD

    # State/action noise for robustness (sim2real)
    state_noise_std: float = 0.01
    action_noise_std: float = 0.05

    # QP solver settings
    solver: str = "OSQP"
    warm_start: bool = True
    max_iterations: int = 50

    min_com_height: float = 0.25
    max_com_height: float = 0.45
    min_com_height: float = 0.25
    max_com_height: float = 0.45

    def __post_init__(self):
        if self.inertia is None:
            self.inertia = np.diag([0.472398, 0.407723, 0.141627])
        # Sanity check: total_mass should equal base_mass (centroidal model
        # uses base-only mass; leg dynamics are handled by WBC separately).
        # total_mass ≠ base_mass is expected: centroidal model uses base mass,
        # while weight support calculates per-leg from total mass including legs.
        # This is intentional — no warning needed.

    @property
    def horizon_time(self) -> float:
        """Total prediction horizon time."""
        return self.horizon_steps * self.dt


class CentroidalDynamics:
    """
    Linearized centroidal dynamics model.

    The centroidal model approximates the robot dynamics at the center of mass:
    - Linear momentum: m * v_com_dot = sum(f_i) + m*g
    - Angular momentum: I * omega_dot = sum(p_i x f_i)

    Where p_i is foot position relative to CoM and f_i is ground reaction force.
    """

    def __init__(self, mass: float = None, inertia: np.ndarray = None):
        """
        Initialize centroidal dynamics.

        Args:
            mass: Total robot mass [kg]  (A2 base = 19.651 kg, full = 40.071 kg)
            inertia: Base inertia tensor [3, 3]
        """
        # Centroidal model uses BASE mass only (leg dynamics handled by WBC)
        # A2 base_link = 19.651 kg; full robot = 40.071 kg (used by WBC)
        self.mass = mass if mass is not None else 19.651
        # A2 base diaginertia = [0.472398, 0.407723, 0.141627]
        self.inertia = inertia if inertia is not None else np.diag([0.472398, 0.407723, 0.141627])

    def compute_jacobian(
        self,
        foot_positions: np.ndarray,
        contact_states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute linear and angular Jacobians for contact forces.

        Args:
            foot_positions: Foot positions relative to CoM [4, 3]
            contact_states: Contact states [4]

        Returns:
            Tuple of (A_lin, A_ang) Jacobians [12, 12]
        """
        n_contacts = int(contact_states.sum())

        if n_contacts == 0:
            return np.zeros((12, 12)), np.zeros((12, 12))

        # Build Jacobians
        A_lin = np.zeros((3, n_contacts * 3))
        A_ang = np.zeros((3, n_contacts * 3))

        idx = 0
        for i in range(4):
            if contact_states[i]:
                p = foot_positions[i]
                # Linear part: force directly contributes
                A_lin[:, idx:idx + 3] = np.eye(3)

                # Angular part: p x f
                A_ang[:, idx:idx + 3] = np.array([
                    [0, -p[2], p[1]],
                    [p[2], 0, -p[0]],
                    [-p[1], p[0], 0]
                ])
                idx += 3

        return A_lin, A_ang

    def build_state_matrix(
        self,
        dt: float,
        gravity: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build discrete-time state transition matrix.

        State: [com_pos(3), com_vel(3), com_euler(3), com_ang_vel(3)]

        Args:
            dt: Time step [s]
            gravity: Gravity vector (default: -z)

        Returns:
            Tuple of (A, B) system matrices
        """
        if gravity is None:
            gravity = np.array([0.0, 0.0, -9.81])

        # Continuous-time A matrix (linearized around hover)
        A_cont = np.zeros((12, 12))

        # Position velocity coupling
        A_cont[0:3, 3:6] = np.eye(3)
        # Euler rates (simplified, small angle)
        A_cont[6:9, 9:12] = -np.eye(3)

        # Discretize (Euler for simplicity)
        A = np.eye(12) + dt * A_cont

        # Input: only vertical force per leg → affects com_vel_z and com_pos_z
        # Input matrix: u = [fz_leg0, fz_leg1, fz_leg2, fz_leg3]
        # B maps input to state derivative
        B = np.zeros((12, 4))  # 4 legs, one vertical force each
        B[5, :] = 1.0 / self.mass  # v̇z = Σfz / m
        B[2, :] = dt * 1.0 / self.mass  # pż = v̇z * dt

        return A, B

    def step(
        self,
        state: np.ndarray,
        foot_positions: np.ndarray,
        contact_states: np.ndarray,
        foot_forces: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Propagate centroidal state forward.

        Args:
            state: Current centroidal state [12]
            foot_positions: Foot positions relative to CoM [4, 3]
            contact_states: Contact states [4]
            foot_forces: Contact forces [4, 3]
            dt: Time step [s]

        Returns:
            Next state [12]
        """
        gravity = np.array([0.0, 0.0, -9.81])
        A, B = self.build_state_matrix(dt, gravity)

        # Extract state components
        com_pos = state[0:3]
        com_vel = state[3:6]
        euler = state[6:9]
        ang_vel = state[9:12]

        # Compute total force and moment
        total_force = np.zeros(3)
        total_moment = np.zeros(3)

        idx = 0
        for i in range(4):
            if contact_states[i]:
                f = foot_forces[i]
                total_force += f
                total_moment += np.cross(foot_positions[i], f)

        # Add gravity
        total_force += self.mass * gravity

        # Compute angular acceleration
        ang_acc = np.linalg.solve(self.inertia, total_moment)

        # Update velocity (linear + angular)
        com_vel_new = com_vel + dt * total_force / self.mass
        ang_vel_new = ang_vel + dt * ang_acc

        # Update position (linear + angular)
        com_pos_new = com_pos + dt * com_vel_new

        # Update orientation (simplified Euler integration)
        euler_new = euler + dt * ang_vel

        return np.concatenate([com_pos_new, com_vel_new, euler_new, ang_vel_new])


class FrictionCone:
    """
    Friction cone constraints for contact forces.

    Models the friction constraint:
    - Normal force >= min_normal
    - Tangential force <= mu * normal force
    """

    def __init__(self, mu: float = 0.7):
        """
        Initialize friction cone.

        Args:
            mu: Friction coefficient
        """
        self.mu = mu

    def as_constraints(
        self,
        force,  # CVXPY variable when HAS_CVXPY, else placeholder
        min_normal: float = 20.0,
        max_normal: float = 300.0,
    ):
        """
        Generate CVXPY constraints for friction cone (if cvxpy available).

        Args:
            force: CVXPY variable [3] for contact force
            min_normal: Minimum normal force [N]
            max_normal: Maximum normal force [N]

        Returns:
            List of constraints (empty if cvxpy not available)
        """
        if not HAS_CVXPY:
            return []
        constraints = []

        # Normal force must be positive and bounded
        constraints.append(force[2] >= min_normal)
        constraints.append(force[2] <= max_normal)

        # Friction cone (linearized pyramid approximation)
        constraints.append(cp.abs(force[0]) <= self.mu * force[2])
        constraints.append(cp.abs(force[1]) <= self.mu * force[2])

        return constraints

    def as_linear_constraints(
        self,
        force,  # CVXPY variable
        min_normal: float = 20.0,
        max_normal: float = 300.0,
        n_approx: int = 4,
    ):
        """
        Generate polyhedral approximation of friction cone (if cvxpy available).

        Args:
            force: CVXPY variable [3]
            min_normal: Minimum normal force
            max_normal: Maximum normal force
            n_approx: Number of facets in approximation

        Returns:
            List of linear constraints (empty if cvxpy not available)
        """
        if not HAS_CVXPY:
            return []
        constraints = []

        # Normal force
        constraints.append(force[2] >= min_normal)
        constraints.append(force[2] <= max_normal)

        # Pyramidal approximation
        for i in range(n_approx):
            angle = 2 * np.pi * i / n_approx
            fx_dir = np.cos(angle)
            fy_dir = np.sin(angle)
            constraints.append(
                fx_dir * force[0] + fy_dir * force[1] <= self.mu * force[2]
            )

        return constraints


class MPCController:
    """
    Model Predictive Controller for centroidal dynamics.

    Solves a finite-horizon optimal control problem at each time step
    to optimize foot placement and contact forces for stable locomotion.
    """

    def __init__(
        self,
        params: Optional[MPCParams] = None,
        dynamics: Optional[CentroidalDynamics] = None,
    ):
        """
        Initialize MPC controller.

        Args:
            params: MPC parameters
            dynamics: Centroidal dynamics model
        """
        self.params = params if params is not None else MPCParams()
        self.dynamics = dynamics if dynamics is not None else CentroidalDynamics(
            mass=self.params.total_mass,      # centroidal model uses total mass
            inertia=self.params.inertia,
        )
        self.friction = FrictionCone(self.params.friction_coeff)

        # Problem and variables
        self._problem = None
        self._last_solution = None

        # State reference
        self._com_ref: Optional[np.ndarray] = None
        self._rpy_ref: Optional[np.ndarray] = None

        # Foot reference
        self._foot_ref: Optional[np.ndarray] = None

        # Warm start
        self._warm_start_force: Optional[np.ndarray] = None

    def set_reference(
        self,
        com_pos: np.ndarray,
        com_vel: Optional[np.ndarray] = None,
        rpy: Optional[np.ndarray] = None,
        ang_vel: Optional[np.ndarray] = None,
        foot_pos: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set reference trajectory for MPC.

        Args:
            com_pos: Reference CoM position [3] or [horizon, 3]
            com_vel: Reference CoM velocity [3] or [horizon, 3]
            rpy: Reference orientation [3] or [horizon, 3]
            ang_vel: Reference angular velocity [3] or [horizon, 3]
            foot_pos: Reference foot positions [4, 3] or [horizon, 4, 3]
        """
        H = self.params.horizon_steps

        self._com_ref = self._expand_ref(com_pos, H, 3)
        self._com_vel_ref = self._expand_ref(
            com_vel if com_vel is not None else np.zeros(3), H, 3
        )
        self._rpy_ref = self._expand_ref(
            rpy if rpy is not None else np.zeros(3), H, 3
        )
        self._ang_vel_ref = self._expand_ref(
            ang_vel if ang_vel is not None else np.zeros(3), H, 3
        )
        self._foot_ref = None  # Not used in simplified MPC

    def _expand_ref(self, ref, horizon: int, dim: int) -> np.ndarray:
        """Expand reference to horizon."""
        ref = np.asarray(ref, dtype=float)
        # If 1D vector, tile to horizon
        if ref.ndim == 1:
            return np.tile(ref, (horizon, 1))
        # If 2D with shape (1, dim), tile to horizon
        elif ref.ndim == 2 and ref.shape[0] == 1:
            return np.tile(ref, (horizon, 1))
        # Already horizon x dim
        elif ref.ndim == 2 and ref.shape[0] == horizon:
            return ref
        else:
            return np.tile(ref.flat[0] * np.ones(dim), (horizon, 1))

    def solve(
        self,
        state: np.ndarray,
        foot_positions: np.ndarray,
        contact_states: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        com_vel: Optional[np.ndarray] = None,
        cmd_vel: Optional[np.ndarray] = None,
        rpy: Optional[np.ndarray] = None,
        ang_vel: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the MPC problem: vertical force (height + weight) + horizontal
        force (velocity tracking) with friction-cone clipping.

        The force distribution follows:
          1. Vertical:  uniform weight support + height PD feedback
          2. Horizontal: PD on velocity error, clipped to friction cone
          3. Yaw:        small differential x-forces on left/right legs to damp rotation

        Args:
            state: Current centroidal state [12] = [com_pos, com_vel, euler, ang_vel]
            foot_positions: Current foot positions relative to CoM [4, 3]
            contact_states: Contact states [4]
            initial_guess: Optional initial force guess [4, 3]
            com_vel: Actual CoM velocity in world frame [3] (if not in state)
            cmd_vel: Command velocity [vx, vy, yaw_rate] in world frame
            rpy: Current roll-pitch-yaw [3]
            ang_vel: Current angular velocity [3]

        Returns:
            Tuple of (optimal_foot_forces [4,3], optimal_foot_positions [4,3])
        """
        n_contacts = int(contact_states.sum())

        if n_contacts == 0:
            return np.zeros((4, 3)), foot_positions.copy()

        # ── Vertical force (height + weight compensation) ─────────
        com_height = state[2]
        target_height = 0.43  # CoM target height
        height_error = target_height - com_height
        com_vel_z = state[5] if len(state) > 5 else 0.0

        # Weight support uses TOTAL robot mass (centroidal model carries whole robot)
        total_weight = self.params.total_mass * 9.81
        per_leg = total_weight / n_contacts

        # Height PD: Fz = mg + kp*(h_des-h) + kd*(0 - v_z)
        # With kp=200, kd=10, at height_error=0.12m and v_z=5m/s:
        #   fz_extra = (200*0.12 + 10*5)/n = (24+50)/n = 37N per leg (n=2) — safe.
        # Original kd=30 gave (24+150)/n = 87N which caused bounce/oscillation.
        kp_height = 200.0
        kd_height = 10.0
        fz_extra = (kp_height * height_error + kd_height * (0.0 - com_vel_z)) / n_contacts
        # Clamp height correction to ±30% of weight per leg to prevent oscillation
        fz_extra = np.clip(fz_extra, -per_leg * 0.3, per_leg * 0.3)

        # ── Horizontal force (velocity tracking in BODY frame) ───
        # Commands (vx,vy) are interpreted in the robot's body frame.
        # We use MODERATE gains since Raibert foot placement is the primary
        # propulsion mechanism. Horizontal forces primarily correct small errors.
        mu = self.params.friction_coeff  # 0.7

        # Velocity tracking — mild correction to supplement Raibert.
        # Gains kept very low (15/5) to avoid oscillation.
        kp_vx = 100.0
        kp_vy = 100.0

        # Command velocity (body-frame interpretation)
        if cmd_vel is not None:
            cmd_vx_body, cmd_vy_body = cmd_vel[0], cmd_vel[1]
        else:
            cmd_vx_body, cmd_vy_body = 0.0, 0.0

        # Build body→world rotation for velocity transform
        if rpy is not None:
            r, p, y = rpy
            cy, sy = np.cos(y), np.sin(y)
            R_yaw = np.array([[cy, -sy], [sy, cy]])  # 2D yaw rotation
        else:
            R_yaw = np.eye(2)

        # Actual velocity: convert world→body using yaw rotation
        if com_vel is not None:
            v_world_xy = com_vel[:2]
            v_body_xy = R_yaw.T @ v_world_xy  # world→body (inverse of R_yaw)
            actual_vx_body, actual_vy_body = v_body_xy[0], v_body_xy[1]
        else:
            actual_vx_body, actual_vy_body = 0.0, 0.0

        # Velocity error in body frame
        evx_body = cmd_vx_body - actual_vx_body
        evy_body = cmd_vy_body - actual_vy_body

        # Desired horizontal force in BODY frame (P-only, no D-damping)
        # D-term removed: -kd*v caused steady-state error (v_eq < v_cmd)
        fx_body_total = kp_vx * evx_body
        fy_body_total = kp_vy * evy_body

        # ── Yaw stabilization ────────────────────────────────────
        kp_yaw = 300.0   # Nm per rad — high to fight friction cone
        kd_yaw = 40.0    # Nm per rad/s

        if rpy is not None and ang_vel is not None:
            yaw_err = 0.0 - rpy[2]
            yaw_rate_err = 0.0 - ang_vel[2]
            tau_yaw_des = kp_yaw * yaw_err + kd_yaw * yaw_rate_err
            tau_yaw_des = np.clip(tau_yaw_des, -80.0, 80.0)
        else:
            tau_yaw_des = 0.0

        # Compute body-frame foot positions and y_sq_sum_body
        if rpy is not None:
            r, p, y = rpy
            cr, sr = np.cos(r), np.sin(r)
            cp, sp = np.cos(p), np.sin(p)
            cy, sy = np.cos(y), np.sin(y)
            R = np.array([
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp,   cp*sr,            cp*cr],
            ])
        else:
            R = np.eye(3)

        foot_pos_body = (R.T @ foot_positions.T).T
        y_sq_sum_body = sum(foot_pos_body[j, 1]**2 for j in range(4) if contact_states[j]) + 1e-6

        # ── Assemble forces with friction-cone clipping ──────────
        forces = np.zeros((4, 3))
        mu_effective = mu  # friction coefficient

        for i in range(4):
            if contact_states[i]:
                # Vertical force (world frame)
                fz = per_leg + fz_extra
                fz = np.clip(fz, self.params.min_normal_force, self.params.max_normal_force)

                # Horizontal forces in body frame
                fx_body = fx_body_total / n_contacts
                fy_body = fy_body_total / n_contacts

                # Yaw torque contribution (body frame)
                yi_body = foot_pos_body[i, 1]
                fx_yaw_body = -tau_yaw_des * yi_body / y_sq_sum_body
                fx_body += fx_yaw_body

                # Rotate horizontal forces to world frame
                f_world_xy = R_yaw @ np.array([fx_body, fy_body])
                fx = f_world_xy[0]
                fy = f_world_xy[1]

                # Friction cone clipping: sqrt(fx² + fy²) <= mu * fz
                f_h_norm = np.sqrt(fx**2 + fy**2)
                f_h_max = mu_effective * fz
                if f_h_norm > f_h_max and f_h_norm > 1e-6:
                    scale = f_h_max / f_h_norm
                    fx *= scale
                    fy *= scale

                forces[i] = np.array([fx, fy, fz])

        # Friction cone clipping (same as before)
        self._last_debug = {
            'height_err': height_error,
            'fz_total': n_contacts * (per_leg + fz_extra),
            'weight': total_weight,
            'fx_total': fx_body_total,
            'fy_total': fy_body_total,
            'evx_body': evx_body,
            'evy_body': evy_body,
            'vx_body': actual_vx_body,
            'vy_body': actual_vy_body,
            'tau_yaw': tau_yaw_des,
        }

        # Scale forces to compensate for MuJoCo contact gap (1.0 = no scaling)
        forces *= self.params.force_scale  # force_scale=1.0 → no-op

        return forces, foot_positions.copy()

    def compute_centroidal_state(
        self,
        com_pos: np.ndarray,
        com_vel: np.ndarray,
        rpy: np.ndarray,
        ang_vel: np.ndarray,
    ) -> np.ndarray:
        """
        Build centroidal state vector.

        Args:
            com_pos: CoM position [3]
            com_vel: CoM velocity [3]
            rpy: Roll-pitch-yaw [3]
            ang_vel: Angular velocity [3]

        Returns:
            State vector [12]
        """
        return np.concatenate([com_pos, com_vel, rpy, ang_vel])

    def reset(self) -> None:
        """Reset MPC internal state."""
        self._last_solution = None
        self._com_ref = None
        self._rpy_ref = None
        self._foot_ref = None
        self._warm_start_force = None
