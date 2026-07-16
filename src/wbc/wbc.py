"""
Whole-Body Control (WBC) for Unitree A2 — aligned with Quadruped-PyMPC.

Architecture (matching Quadruped-PyMPC compute_stance_and_swing_torque):
  1. tau_ff = -J^T * F_mpc          (stance, body-origin Jacobian)
  2. tau_ori                         (orientation via differential forces)
  3. tau_swing                       (Cartesian PD for swing legs)
  4. tau_joint_pd                    (IK-based joint PD, kp=10, kd=2)

NO separate tau_g — MPC forces include gravity support.
Joint PD targets are computed via per-leg IK, not fixed standing pose.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import mujoco


@dataclass
class WBCParams:
    """Parameters for the whole-body controller — matching Quadruped-PyMPC."""

    control_freq: float = 500.0
    dt: float = 0.002

    # Orientation gains (tuned for 40kg robot mass with strong damping)
    ori_roll_kp: float = 1200.0
    ori_roll_kd: float = 120.0
    ori_pitch_kp: float = 1200.0
    ori_pitch_kd: float = 120.0
    ori_yaw_kp: float = 600.0
    ori_yaw_kd: float = 60.0

    # Joint PD (matching PyMPC impedence_joint_position/velocity_gain)
    imped_kp: float = 10.0   # hip+thigh
    imped_kd: float = 2.0    # hip+thigh
    calf_kp: float = 60.0    # calf needs stronger PD — no MPC torque

    # Swing foot tracking (matching PyMPC swing_position/velocity_gain_fb)
    swing_pos_kp: float = 400.0
    swing_pos_kd: float = 15.0

    # Torque limits (A2: hip=120, thigh=120, calf=180)
    torque_limits: Tuple[float, float, float] = (120.0, 120.0, 180.0)


class WBCController:
    """Whole-Body Controller aligned with Quadruped-PyMPC."""

    def __init__(self, model, data, params: Optional[WBCParams] = None):
        self.model = model
        self.data = data
        self.p = params if params is not None else WBCParams()
        self._nv = model.nv
        self._nj = self._nv - 6
        self._step_count = 0

        # Foot body IDs (calf body = end-effector)
        self._foot_body_ids = []
        for leg in ['FL', 'FR', 'RL', 'RR']:
            self._foot_body_ids.append(model.body(f'{leg}_calf').id)

        # Standing joint pose (for initial reference only)
        self._q_stand = np.array([
            0.0, 0.596, -1.194, 0.0, 0.596, -1.194,
            0.0, 0.594, -1.195, 0.0, 0.594, -1.195,
        ])

    # ===================================================================
    # Public API
    # ===================================================================

    def compute_joint_torques(
        self,
        com_pos, com_vel, com_rpy, com_ang_vel,
        foot_positions, foot_velocities,
        foot_desired_positions, foot_desired_velocities,
        contact_states,
        joint_positions, joint_velocities,
        mpc_forces=None, com_ref=None, com_vel_ref=None,
        joint_pos_des=None, joint_vel_des=None,
        is_walking=False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        WBC matching Quadruped-PyMPC:
          tau = tau_ff + tau_ori + tau_swing + tau_joint_pd
        where tau_ff = SUM(-J_i^T * F_mpc,i) for stance legs.
        """
        self._step_count += 1
        info = {}

        # ---- 1. MPC force feedforward (stance) ----
        tau_ff = np.zeros(self._nj)
        if mpc_forces is not None:
            for i in range(4):
                if contact_states[i]:
                    Ji = self._compute_foot_jacobian(i)          # [3, 18] world frame
                    Ji_joints = Ji[:, 6:]                        # [3, 12] joint cols
                    tau_ff += -Ji_joints.T @ mpc_forces[i]       # -J^T * F

        # ---- 2. Orientation stabilisation ----
        tau_ori = self._compute_orientation_pd(
            com_rpy, com_ang_vel, contact_states, foot_positions)

        # ---- 3. Swing foot tracking ----
        tau_swing = np.zeros(self._nj)
        for i in range(4):
            if not contact_states[i]:
                s = i * 3
                if joint_pos_des is not None:
                    # Joint-space swing PD tracking (highly robust to joint friction and leg weight)
                    q_des = joint_pos_des[s:s+3]
                    q_err = q_des - joint_positions[s:s+3]
                    dq_err = -joint_velocities[s:s+3]
                    tau_swing[s]   = 250.0 * q_err[0] + 6.0 * dq_err[0]
                    tau_swing[s+1] = 250.0 * q_err[1] + 6.0 * dq_err[1]
                    tau_swing[s+2] = 300.0 * q_err[2] + 8.0 * dq_err[2]
                else:
                    # Fallback to Cartesian operational space tracking
                    Ji = self._compute_foot_jacobian(i)
                    Ji_joints = Ji[:, 6:]
                    err = foot_desired_positions[i] - foot_positions[i]
                    verr = foot_desired_velocities[i] - foot_velocities[i]
                    f_swing = self.p.swing_pos_kp * err + self.p.swing_pos_kd * verr
                    tau_swing += Ji_joints.T @ f_swing

        # ---- 4. Joint PD on STANDING POSE (fixed reference, not IK)
        #    IK-based PD follows foot error and reinforces collapse.
        #    Fixed standing pose gives a stable reference for the knee.
        # ----
        tau_joint_pd = np.zeros(self._nj)
        for i in range(4):
            s = i * 3
            if contact_states[i]:  # Apply ONLY to stance legs to avoid fighting swing tracking
                # Stance legs track the fixed nominal stance joints self._q_stand
                # to prevent positive feedback height drift (extension / collapse).
                q_err = self._q_stand[s:s+3] - joint_positions[s:s+3]
                dq_err = -joint_velocities[s:s+3]
                if is_walking:
                    # Moderate stance gains to track joint trajectory and stabilize orientation
                    tau_joint_pd[s]   = 40.0 * q_err[0] + 3.0 * dq_err[0]
                    tau_joint_pd[s+1] = 40.0 * q_err[1] + 3.0 * dq_err[1]
                    tau_joint_pd[s+2] = 50.0 * q_err[2] + 4.0 * dq_err[2]
                else:
                    # Strong joint PD during standing to resist collapse
                    tau_joint_pd[s]   = 10.0 * q_err[0] + 2.0 * dq_err[0]
                    tau_joint_pd[s+1] = 10.0 * q_err[1] + 2.0 * dq_err[1]
                    tau_joint_pd[s+2] = 80.0 * q_err[2] + 4.0 * dq_err[2]
            else:
                # Swing legs: no extra joint-space damping needed (tau_swing already has damping)
                tau_joint_pd[s]   = 0.0
                tau_joint_pd[s+1] = 0.0
                tau_joint_pd[s+2] = 0.0

        # Gravity bias torque (for logging and diagnostic prints only)
        tau_g = self.data.qfrc_bias[6:].copy()

        # Combine WBC terms matching Quadruped-PyMPC.
        # We do NOT add tau_g because MPC forces (tau_ff) already compensate for
        # the total gravity weight (mg).
        tau = tau_ff + tau_ori + tau_swing + tau_joint_pd

        # Limit joint torques to mechanical limits
        if self._step_count % 100 == 0:
            for i in range(4):
                if not contact_states[i]:
                    s = i * 3
                    print(f"[DEBUG SWING TORQUE] step={self._step_count} | Leg {i} | "
                          f"q_des={joint_pos_des[s:s+3] if joint_pos_des is not None else 'None'} | "
                          f"q_cur={joint_positions[s:s+3]} | "
                          f"tau_sw={tau_swing[s:s+3]} | "
                          f"tau_pd={tau_joint_pd[s:s+3]} | "
                          f"tau_g={tau_g[s:s+3]} | "
                          f"tau_tot={tau[s:s+3]}")

        for i in range(4):
            s = i * 3
            tau[s]   = np.clip(tau[s],   -self.p.torque_limits[0], self.p.torque_limits[0])
            tau[s+1] = np.clip(tau[s+1], -self.p.torque_limits[1], self.p.torque_limits[1])
            tau[s+2] = np.clip(tau[s+2], -self.p.torque_limits[2], self.p.torque_limits[2])

        info['tau_norm']       = float(np.linalg.norm(tau))
        info['tau_ff_norm']    = float(np.linalg.norm(tau_ff))
        info['tau_ori_norm']   = float(np.linalg.norm(tau_ori))
        info['tau_swing_norm'] = float(np.linalg.norm(tau_swing))
        info['tau_pd_norm']    = float(np.linalg.norm(tau_joint_pd))
        info['tau_g_norm']     = float(np.linalg.norm(tau_g))

        return tau, info

    # ===================================================================
    # Helpers
    # ===================================================================

    def _compute_foot_jacobian(self, leg_id: int) -> np.ndarray:
        """Compute mathematically consistent contact-point Jacobian."""
        body_id = self._foot_body_ids[leg_id]
        J_body = np.zeros((3, self._nv))
        J_rot  = np.zeros((3, self._nv))
        mujoco.mj_jacBody(self.model, self.data, J_body, J_rot, body_id)

        r_body   = np.array([0.0, 0.0, -0.275])
        xmat     = self.data.xmat[body_id].reshape(3, 3)
        r_world  = xmat @ r_body
        rx,ry,rz = r_world
        skew_r   = np.array([[0,-rz,ry],[rz,0,-rx],[-ry,rx,0]])

        # J_point = J_body - skew_r @ J_rot (applied to all DOFs)
        J = J_body - skew_r @ J_rot
        return J

    def _compute_orientation_pd(self, rpy, ang_vel, contact_states,
                                 foot_positions=None) -> np.ndarray:
        """Orientation stabilisation via differential foot forces -> -J^T * F."""
        roll, pitch, yaw = rpy
        wx, wy, wz = ang_vel
        tau_ori = np.zeros(self._nj)
        n_stance = max(2, int(contact_states.sum()))

        # Roll: differential Fz left/right
        # Positive roll -> left side up -> left legs (y_sign=1) must push less (Fz_corr < 0)
        fz_roll = - (self.p.ori_roll_kp * roll + self.p.ori_roll_kd * wx)
        for i in range(4):
            if contact_states[i]:
                Ji = self._compute_foot_jacobian(i)
                Ji_joints = Ji[:, 6:]
                y_sign = 1.0 if i in (0, 2) else -1.0
                f_vec = np.array([0.0, 0.0, y_sign * fz_roll / n_stance])
                tau_ori += -Ji_joints.T @ f_vec

        # Pitch: differential Fz front/rear
        # Positive pitch -> nose down -> front legs (x_sign=1) must push harder (Fz_corr > 0)
        # Due to the negative sign in the pitch torque relation (tau_y = -x * Fz),
        # we need fz_pitch to be positive to get restoring torque.
        fz_pitch = self.p.ori_pitch_kp * pitch + self.p.ori_pitch_kd * wy
        for i in range(4):
            if contact_states[i]:
                Ji = self._compute_foot_jacobian(i)
                Ji_joints = Ji[:, 6:]
                x_sign = 1.0 if i in (0, 1) else -1.0
                f_vec = np.array([0.0, 0.0, x_sign * fz_pitch / n_stance])
                tau_ori += -Ji_joints.T @ f_vec

        # Yaw: differential Fx left/right
        # Positive yaw -> rotated left -> left legs (y > 0) must push forward (Fx_corr > 0)
        # to generate restoring torque (tau_z = -y * Fx < 0).
        fx_yaw = self.p.ori_yaw_kp * yaw + self.p.ori_yaw_kd * wz
        if foot_positions is not None:
            y_sq_sum = sum(foot_positions[i, 1]**2 for i in range(4)
                          if contact_states[i]) + 1e-6
        else:
            y_sq_sum = 1e-6
        for i in range(4):
            if contact_states[i]:
                Ji = self._compute_foot_jacobian(i)
                Ji_joints = Ji[:, 6:]
                y_val = foot_positions[i, 1] if foot_positions is not None else 0.0
                # Force correction Fx = fx_yaw * y / sum(y^2)
                f_vec = np.array([fx_yaw * y_val / y_sq_sum, 0.0, 0.0])
                tau_ori += -Ji_joints.T @ f_vec

        if self._step_count % 100 == 0:
            print(f"[DEBUG ORI] step={self._step_count} | rpy=({roll:.3f},{pitch:.3f},{yaw:.3f}) | "
                  f"fz_roll={fz_roll:.1f} fz_pitch={fz_pitch:.1f} fx_yaw={fx_yaw:.1f} | "
                  f"tau_ori_norm={np.linalg.norm(tau_ori):.3f} | "
                  f"gains: roll_kp={self.p.ori_roll_kp:.1f} pitch_kp={self.p.ori_pitch_kp:.1f}")

        return tau_ori
