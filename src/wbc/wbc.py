"""
Whole-Body Control (WBC) for Unitree A2 in MuJoCo.

Implements a physically-correct floating-base WBC that accounts for:
  - MuJoCo model uses Z-up world frame
  - All four legs share the same joint axis definitions
    (hip=[1,0,0], thigh=[0,1,0], calf=[0,1,0])
  - Contact force is the primary vertical-force source via task weights
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from utils.math_utils import block_diag, hat, pinv_svd, weighted_pinv, regularize


@dataclass
class WBCParams:
    """Parameters for the whole-body controller.

    A2 physical parameters (from a2_scene.xml):
      - Hip torque limit: ±120 Nm
      - Thigh torque limit: ±120 Nm
      - Calf torque limit: ±180 Nm
    """
    # Control frequency
    control_freq: float = 500.0
    dt: float = 0.002

    # PD gains – height is the critical channel
    com_z_kp: float = 300.0     # vertical position gain
    com_z_kd: float = 30.0      # vertical velocity damping
    com_roll_kp: float = 150.0  # roll stabilisation
    com_roll_kd: float = 15.0
    com_pitch_kp: float = 150.0 # pitch stabilisation
    com_pitch_kd: float = 15.0
    com_yaw_kp: float = 50.0
    com_yaw_kd: float = 5.0

    # Foot task gains (swing legs only)
    foot_xy_kp: float = 400.0
    foot_xy_kd: float = 20.0
    foot_z_kp: float = 800.0
    foot_z_kd: float = 40.0

    # Joint regularisation
    joint_pos_kp: float = 2.0
    joint_vel_kd: float = 0.1

    # Torque limits per joint type [hip, thigh, calf]
    # A2: hip=±120, thigh=±120, calf=±180 Nm
    torque_limits: Tuple[float, float, float] = (120.0, 120.0, 180.0)

    # Stance contact: how hard to maintain foot height
    contact_kp: float = 2000.0

    def __post_init__(self):
        for t in self.torque_limits:
            if t <= 0:
                raise ValueError("torque_limits must all be positive")


class WBCHierarchy:
    """Null-space-priority task solver."""

    def __init__(self, params: WBCParams):
        self.p = params

    def solve(
        self,
        J_list: List[np.ndarray],
        xd_list: List[np.ndarray],
        weights: List[np.ndarray],
        qvel: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve tasks in priority order.

        Args:
            J_list:  list of Jacobians  [task_dim, 18]
            xd_list: list of desired task-space accelerations
            weights: list of diagonal weight matrices [task_dim, task_dim]
            qvel:    full 18-DOF velocity

        Returns:
            (tau_full [18], info)
        """
        N = np.eye(18)
        qacc = np.zeros(18)   # cumulative solution
        info = {}

        for idx, (J, xd) in enumerate(zip(J_list, xd_list)):
            dim = J.shape[0]
            if dim == 0:
                continue

            # Project into current null-space
            J_proj = J @ N

            # Task gain: xd_proj = W^{-1/2} xd  (weighted error)
            W = weights[idx]
            W_sqrt = np.sqrt(W)          # [dim, dim] diagonal
            W_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(W)))

            xd_proj = W_inv_sqrt @ xd    # [dim]

            # Null-space projection of weighted Jacobian
            J_w = W_sqrt @ J_proj        # [dim, 18]
            JT = J_w.T

            # Solve (J_w * J_w^T) * lambda = xd_proj
            try:
                JJT = J_w @ JT
                reg = np.eye(dim) * 1e-4
                lam = np.linalg.solve(JJT + reg, xd_proj)  # [dim]
            except np.linalg.LinAlgError:
                lam = np.zeros(dim)

            # Task-space force
            f = JT @ lam                  # [18]

            # Inject into null-space
            qacc += f

            # Update null-space projector
            try:
                J_pinv = np.linalg.pinv(J_proj, rcond=1e-2)
                N = N @ (np.eye(18) - J_pinv @ J_proj)
            except np.linalg.LinAlgError:
                pass

            info[f'task_{idx}_dim'] = dim

        return qacc, info


class WBCController:
    """Floating-base whole-body controller for A2."""

    def __init__(self, params: Optional[WBCParams] = None):
        self.p = params if params is not None else WBCParams()
        self._solver = WBCHierarchy(self.p)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def compute_joint_torques(
        self,
        com_pos: np.ndarray,        # [3] world position
        com_vel: np.ndarray,        # [3] world velocity
        com_rpy: np.ndarray,        # [3] roll, pitch, yaw [rad]
        com_ang_vel: np.ndarray,    # [3] world ang vel
        foot_positions: np.ndarray, # [4, 3] world frame
        foot_velocities: np.ndarray,# [4, 3]
        foot_desired_positions: np.ndarray,  # [4, 3]
        foot_desired_velocities: np.ndarray, # [4, 3]
        contact_states: np.ndarray, # [4] bool
        joint_positions: np.ndarray,# [12]
        joint_velocities: np.ndarray,# [12]
        mpc_forces: Optional[np.ndarray] = None,  # [4, 3]
    ) -> Tuple[np.ndarray, Dict]:
        """Return joint torques [12] and info dict."""
        info = {}

        # ── 1. Build full 18-DOF velocity ───────────────────────────────
        qvel = np.zeros(18)
        qvel[0:3]   = com_vel         # base lin vel
        qvel[3:6]   = com_ang_vel     # base ang vel
        qvel[6:18]  = joint_velocities

        # ── 2. Assemble tasks ───────────────────────────────────────────
        J_list, xd_list, W_list = [], [], []

        self._add_height_task(J_list, xd_list, W_list, com_pos, com_vel, com_rpy)
        self._add_attitude_task(J_list, xd_list, W_list,
                                com_rpy, com_ang_vel)
        self._add_foot_tasks(J_list, xd_list, W_list,
                             foot_positions, foot_velocities,
                             foot_desired_positions, foot_desired_velocities,
                             contact_states)
        self._add_joint_reg(J_list, xd_list, W_list,
                            joint_positions, joint_velocities,
                            contact_states)

        # ── 3. Solve ───────────────────────────────────────────────────
        qacc_full, info = self._solver.solve(J_list, xd_list, W_list, qvel)

        # ── 4. Extract joint torques (joint-space dynamics) ─────────────
        # tau = M_j * qacc_j + C_j + G_j  (simplified: identity M, zero C/G)
        tau = 1.0 * qacc_full[6:18].copy()

        # ── 5. Add MPC force contribution ───────────────────────────────
        if mpc_forces is not None:
            # Build body-to-world rotation matrix from RPY
            R_body = self._rpy_to_rot(com_rpy)  # [3,3] body → world
            R_world_to_body = R_body.T           # world → body
            for i in range(4):
                if contact_states[i]:
                    f_world = mpc_forces[i]           # [3] world frame
                    f_body = R_world_to_body @ f_world  # rotate to body frame
                    Jj = self._foot_jac_joints(i)     # [3x12] in body/hip frame
                    # Force → torque (joint DOFs only)
                    tau += Jj.T @ f_body

        # ── 6. Clip ────────────────────────────────────────────────────
        # A2 torque limits per joint type: hip=±120, thigh=±120, calf=±180 Nm
        for i in range(4):  # 4 legs
            # hip joint (index 0, 3, 6, 9)
            tau[i*3] = np.clip(tau[i*3], -self.p.torque_limits[0], self.p.torque_limits[0])
            # thigh joint (index 1, 4, 7, 10)
            tau[i*3 + 1] = np.clip(tau[i*3 + 1], -self.p.torque_limits[1], self.p.torque_limits[1])
            # calf joint (index 2, 5, 8, 11)
            tau[i*3 + 2] = np.clip(tau[i*3 + 2], -self.p.torque_limits[2], self.p.torque_limits[2])
        self._last_info = info  # 保存用于调试
        return tau, info

    # ------------------------------------------------------------------ #
    #  Individual task builders                                           #
    # ------------------------------------------------------------------ #

    def _add_height_task(self, J_list, xd_list, W_list,
                         com_pos, com_vel, com_rpy=None):
        """Task 1 – maintain body height (priority highest).

        Uses the WORLD-frame z-rows of each foot Jacobian to command
        vertical forces. The body-frame Jacobian is rotated to world
        frame so that height control works correctly even when the
        base is tilted.
        """
        z_err = 0.38 - com_pos[2]
        vel_err = -com_vel[2]
        desired_total_force = self.p.com_z_kp * z_err + self.p.com_z_kd * vel_err  # [N]

        # Build world-frame z-direction in body coordinates
        if com_rpy is not None:
            R_body = self._rpy_to_rot(com_rpy)       # body → world
            world_z_in_body = R_body[2, :]            # 3rd row = world-z in body frame
        else:
            world_z_in_body = np.array([0.0, 0.0, 1.0])

        # Stack world-frame z-rows of foot Jacobian for all 4 legs
        J_rows = []
        for i in range(4):
            Ji_body = self._foot_jac_joints(i)        # [3 × 12] in body frame
            Ji_world_z = world_z_in_body @ Ji_body    # [12] world-z component
            J_rows.append(Ji_world_z.reshape(1, -1))

        J = np.vstack(J_rows)               # [4 × 12]

        # Each leg gets equal share of total force
        per_leg = desired_total_force / 4.0
        xd = np.full(4, per_leg)

        W = np.eye(4) * self.p.com_z_kp

        # Extend to 18-DOF (base rows 0:6 are zero → only joint torques 6:18 matter)
        J18 = np.zeros((4, 18))
        J18[:, 6:18] = J
        J_list.append(J18)
        xd_list.append(xd)
        W_list.append(W)

    def _add_attitude_task(self, J_list, xd_list, W_list,
                           rpy, ang_vel):
        """Task 2 – stabilise roll and pitch."""
        # Roll  -> qvel[3]
        # Pitch -> qvel[4]
        # Yaw   -> qvel[5]
        J = np.zeros((3, 18))
        J[0, 3] = 1.0   # roll_dot  = qvel[3]
        J[1, 4] = 1.0   # pitch_dot = qvel[4]
        J[2, 5] = 1.0   # yaw_dot   = qvel[5]

        roll_d, pitch_d, yaw_d = rpy
        wx, wy, wz = ang_vel

        xd = np.array([
            self.p.com_roll_kp  * (0.0 - roll_d)  + self.p.com_roll_kd  * (0.0 - wx),
            self.p.com_pitch_kp * (0.0 - pitch_d) + self.p.com_pitch_kd * (0.0 - wy),
            self.p.com_yaw_kp   * (0.0 - yaw_d)   + self.p.com_yaw_kd   * (0.0 - wz),
        ])
        W = np.diag([self.p.com_roll_kp, self.p.com_pitch_kp, self.p.com_yaw_kp])

        J_list.append(J)
        xd_list.append(xd)
        W_list.append(W)

    def _add_foot_tasks(self, J_list, xd_list, W_list,
                        foot_pos, foot_vel,
                        foot_des_pos, foot_des_vel,
                        contact):
        """Task 3 – swing-foot tracking; stance foot z-maintenance."""
        for i in range(4):
            Ji = self._foot_jac_joints(i)   # [3x12]

            if not contact[i]:
                # Swing: full 3-D position tracking
                err = foot_des_pos[i] - foot_pos[i]
                verr = foot_des_vel[i] - foot_vel[i]
                xd_vec = (self.p.foot_xy_kp * err +
                          self.p.foot_xy_kd * verr)
                W = np.diag([self.p.foot_xy_kp,
                              self.p.foot_xy_kp,
                              self.p.foot_z_kp])
            else:
                # Stance: only z-direction compliance
                z_err  = foot_des_pos[i, 2] - foot_pos[i, 2]
                z_verr = foot_des_vel[i, 2] - foot_vel[i, 2]
                k = self.p.contact_kp
                d = 20.0
                xd_vec = np.array([0.0, 0.0,
                                   k * z_err + d * z_verr])
                # Low xy weight; moderate z so stance foot can yield to height task
                W = np.diag([0.01, 0.01, 200.0])

            # Extend to 18-DOF: only joints matter
            J18 = np.zeros((3, 18))
            J18[:, 6:18] = Ji

            J_list.append(J18)
            xd_list.append(xd_vec)
            W_list.append(W)

    def _add_joint_reg(self, J_list, xd_list, W_list,
                       joint_pos, joint_vel, contact):
        """Task 4 – joint regularisation (lowest priority)."""
        # Regularise swing-leg joints toward 0
        n_stance = int(contact.sum())
        n_swing  = 4 - n_stance
        if n_swing == 0:
            return

        # Only swing joints are regularised
        reg_weight = self.p.joint_pos_kp
        J_reg = np.zeros((n_swing * 3, 18))
        xd_reg = np.zeros(n_swing * 3)
        W_reg  = np.eye(n_swing * 3) * reg_weight

        row = 0
        for i in range(4):
            if not contact[i]:
                J_reg[row:row+3, 6 + i*3 : 6 + i*3 + 3] = np.eye(3)
                xd_reg[row]     = -self.p.joint_pos_kp * joint_pos[i*3]
                xd_reg[row + 1] = -self.p.joint_pos_kp * joint_pos[i*3 + 1]
                xd_reg[row + 2] = -self.p.joint_pos_kp * joint_pos[i*3 + 2]
                row += 3

        J_list.append(J_reg)
        xd_list.append(xd_reg)
        W_list.append(W_reg)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
        """Convert roll-pitch-yaw [rad] to rotation matrix (body → world)."""
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr],
        ])

    # ------------------------------------------------------------------ #
    #  Analytical foot Jacobian (Z-up model, joint axes=[1,0,0],[0,1,0],[0,1,0])  #
    # ------------------------------------------------------------------ #

    def _foot_jac_joints(self, leg_id: int) -> np.ndarray:
        """
        Return 3×12 analytical Jacobian for one leg in the Z-up model.

        Joint layout: [hip, thigh, calf]
          hip_joint   axis = [1,0,0]  → rotates around body x-axis
          thigh_joint axis = [0,1,0]  → rotates around body y-axis
          calf_joint  axis = [0,1,0]  → rotates around body y-axis
        Leg lengths: l1 = 0.213 m (hip-to-knee), l2 = 0.213 m (knee-to-foot)

        The analytical derivatives below are valid at the standing point
        (thigh ≈ 0.6 rad, calf ≈ –1.5 rad) and are close enough for use
        as a linearised Jacobian.
        """
        # A2 leg lengths [m] (from a2_scene.xml)
        l1 = 0.12779    # hip → knee (thigh link length)
        l2 = 0.275      # knee → foot (calf link length)
        # A2 standing pose: hip=0, thigh=-0.6, calf=-1.0 [rad]
        # (moderately bent for stability and control authority)
        T  = -0.6       # thigh angle [rad]
        C  = -1.0       # calf angle [rad]
        cs = T + C      # thigh + calf sum angle

        ct, st = np.cos(T), np.sin(T)
        cc, sc = np.cos(cs), np.sin(cs)

        J = np.zeros((3, 12))
        s = leg_id * 3  # column offset in 12-DOF joint Jacobian

        # ∂foot / ∂q_hip  (hip rotates about x; at hip=0 this is near-zero for z)
        J[0, s+0] = 0.0
        J[1, s+0] = -l2 * sc           # ∂y/∂hip  (small coupling)
        J[2, s+0] = 0.0

        # ∂foot / ∂q_thigh
        J[0, s+1] =  l2 * sc
        J[1, s+1] =  l1 * ct - l2 * cc
        J[2, s+1] = -l1 * st - l2 * sc

        # ∂foot / ∂q_calf
        J[0, s+2] =  l2 * sc
        J[1, s+2] = -l2 * cc
        J[2, s+2] = -l2 * sc

        return J
