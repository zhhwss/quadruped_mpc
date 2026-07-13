"""Unitree A2 Robot Model for Quadruped MPC."""

import numpy as np
import mujoco
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


@dataclass
class RobotState:
    """Complete robot state container."""
    base_pos: np.ndarray = None
    base_quat: np.ndarray = None
    base_lin_vel: np.ndarray = None
    base_ang_vel: np.ndarray = None
    joint_pos: np.ndarray = None
    joint_vel: np.ndarray = None
    joint_torque: np.ndarray = None

    def __post_init__(self):
        if self.base_pos is None:
            self.base_pos = np.zeros(3)
        if self.base_quat is None:
            self.base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if self.base_lin_vel is None:
            self.base_lin_vel = np.zeros(3)
        if self.base_ang_vel is None:
            self.base_ang_vel = np.zeros(3)
        if self.joint_pos is None:
            self.joint_pos = np.zeros(12)
        if self.joint_vel is None:
            self.joint_vel = np.zeros(12)
        if self.joint_torque is None:
            self.joint_torque = np.zeros(12)


class A2RobotParams:
    """物理参数：从 a2_scene.xml 精确提取"""

    # ── 质量 & 惯量 ───────────────────────────────────────────
    base_mass: float = 19.651            # kg
    base_inertia: np.ndarray = None      # [3,3] diagonal
    hip_mass: float = 1.62               # kg  per hip link
    thigh_mass: float = 3.079            # kg  per thigh
    calf_mass: float = 0.407             # kg  per calf

    # ── 连杆几何（XML 实测值） ───────────────────────────────
    hip_positions: np.ndarray = None     # [4,3] FL/FR/RL/RR hip-to-base
    thigh_length: float = 0.12779        # hip→thigh offset (y方向, m)
    calf_length: float = 0.275           # thigh→calf offset (z方向, m)

    # ── 关节限位 & 传动 ─────────────────────────────────────
    hip_joint_limits: Tuple[float, float] = (-1.01, 1.01)
    thigh_joint_limits: Tuple[float, float] = (-2.34, 3.15)
    calf_joint_limits: Tuple[float, float] = (-2.77, -0.54)

    torque_limit_hip: float = 120.0
    torque_limit_thigh: float = 120.0
    torque_limit_calf: float = 180.0

    def __post_init__(self):
        if self.base_inertia is None:
            self.base_inertia = np.diag([0.472398, 0.407723, 0.141627])
        if self.hip_positions is None:
            self.hip_positions = np.array([
                [ 0.25944,  0.075113, 0.0],   # FL
                [ 0.25944, -0.075113, 0.0],   # FR
                [-0.25944,  0.075113, 0.0],   # RL
                [-0.25944, -0.075113, 0.0],   # RR
            ])

    @property
    def total_mass(self) -> float:
        return self.base_mass + 4 * (self.hip_mass + self.thigh_mass + self.calf_mass)

    @property
    def standing_base_z(self) -> float:
        return 0.50  # base_link pos="0 0 0.50"

    @property
    def nominal_com_height(self) -> float:
        return 0.45

    def torque_limit_for_joint(self, joint_name: str) -> float:
        if 'hip' in joint_name:
            return self.torque_limit_hip
        elif 'thigh' in joint_name:
            return self.torque_limit_thigh
        else:
            return self.torque_limit_calf


class A2Robot:
    """Unitree A2 robot interface for MuJoCo."""

    # Joint ordering matches A2 model: FL, FR, RL, RR
    # Each leg: hip_joint, thigh_joint, calf_joint
    JOINT_NAMES = [
        'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
        'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
        'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
        'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    ]

    # Actuator ordering matches MuJoCo model: FR, FL, RR, RL
    # Each leg: hip_joint, thigh_joint, calf_joint
    ACTUATOR_NAMES = [
        'FR_hip', 'FR_thigh', 'FR_calf',
        'FL_hip', 'FL_thigh', 'FL_calf',
        'RR_hip', 'RR_thigh', 'RR_calf',
        'RL_hip', 'RL_thigh', 'RL_calf',
    ]

    FOOT_NAMES = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    FOOT_BODY_NAMES = {
        'FL': 'FL_calf',  # Foot is attached to calf body
        'FR': 'FR_calf',
        'RL': 'RL_calf',
        'RR': 'RR_calf',
    }

    def __init__(self, model_path: str):
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Map joint/actuator names to IDs and qpos/dof addresses
        self.joint_ids = {}
        self.joint_qposadr = {}   # qpos address for each joint
        self.joint_dofadr = {}    # qvel address for each joint
        self.actuator_ids = {}
        self.foot_body_ids = {}

        for name in self.JOINT_NAMES:
            try:
                jid = self.mj_model.joint(name).id
                self.joint_ids[name] = jid
                self.joint_qposadr[name] = self.mj_model.joint(jid).qposadr[0]
                self.joint_dofadr[name] = self.mj_model.joint(jid).dofadr[0]
            except KeyError:
                raise KeyError(f"Joint '{name}' not found")

        for name in self.ACTUATOR_NAMES:
            try:
                self.actuator_ids[name] = self.mj_model.actuator(name).id
            except KeyError:
                raise KeyError(f"Actuator '{name}' not found")

        for leg, body_name in self.FOOT_BODY_NAMES.items():
            try:
                self.foot_body_ids[leg] = self.mj_model.body(body_name).id
            except KeyError:
                raise KeyError(f"Body '{body_name}' not found")

        self.state = RobotState()
        self._init_standing_pose()

    def _init_standing_pose(self):
        """Set symmetric standing pose with all four legs identical.

        A2 leg geometry from a2_scene.xml:
          l1 (hip→thigh) = 0.12779 m  (y-offset)
          l2 (thigh→calf) = 0.275 m   (z-offset)
          hip=[1,0,0], thigh=[0,1,0], calf=[0,1,0]

        Pose with per-leg angles for balanced, level support under CoM:
          Front: hip=0, thigh=-0.29 (elbow back)
          Rear:  hip=0, thigh=+0.29 (knee forward)
          All:   calf=-1.0 (moderate bend)
          Support center at x≈0, feet at y≈±0.20 (natural width)
          Within limits: hip=[-1.01,1.01], thigh=[-2.34,3.15], calf=[-2.77,-0.54]
        """
        # IK (MuJoCo FK): H=0.467, contact exactly at (±0.259,±0.203,0), sup center x=0
        self.stand_joint_pos = np.array([
             0.0,  0.596, -1.194,   # FL
             0.0,  0.596, -1.194,   # FR
             0.0,  0.594, -1.195,   # RL
             0.0,  0.594, -1.195,   # RR
        ])

    def reset(self, qpos: Optional[np.ndarray] = None) -> RobotState:
        """Reset simulation."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        if qpos is not None:
            self.mj_data.qpos[:] = qpos
        else:
            # Set standing pose: use stored qposadr
            for i, name in enumerate(self.JOINT_NAMES):
                self.mj_data.qpos[self.joint_qposadr[name]] = self.stand_joint_pos[i]

        # Adjust base height so feet just touch ground
        # At standing pose (T=-0.35): foot_body_z ≈ base_z - 0.258
        # Collision geometry lowest point ≈ foot_body_z - 0.185
        # → base_z ≈ 0.443 for feet at z=0
        self.mj_data.qpos[2] = 0.467

        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.update_state()
        return self.state

    def update_state(self) -> RobotState:
        """Extract state from MuJoCo."""
        # Base
        body_id = self.mj_model.body('base_link').id
        self.state.base_pos = self.mj_data.xpos[body_id].copy()
        self.state.base_quat = self.mj_data.xquat[body_id].copy()

        # Base velocities (world frame)
        lin_vel_world = self.mj_data.cvel[body_id][:3].copy()
        ang_vel_world = self.mj_data.cvel[body_id][3:].copy()

        # Transform to body frame
        R = self._quat_to_rot(self.state.base_quat)
        self.state.base_lin_vel = R.T @ lin_vel_world
        self.state.base_ang_vel = R.T @ ang_vel_world

        # Joints (use qposadr/dofadr, NOT joint ID as index!)
        for i, name in enumerate(self.JOINT_NAMES):
            self.state.joint_pos[i] = self.mj_data.qpos[self.joint_qposadr[name]]
            self.state.joint_vel[i] = self.mj_data.qvel[self.joint_dofadr[name]]

        return self.state

    def get_rpy(self) -> np.ndarray:
        """Return roll-pitch-yaw [rad] of the base in world frame."""
        from utils.math_utils import quaternion_to_rpy
        return quaternion_to_rpy(self.state.base_quat)

    def get_foot_positions(self) -> np.ndarray:
        """Get foot positions [4, 3] in world frame."""
        positions = np.zeros((4, 3))
        for i, (leg, body_id) in enumerate(self.foot_body_ids.items()):
            positions[i] = self.mj_data.xpos[body_id].copy()
        return positions

    def get_foot_velocities(self) -> np.ndarray:
        """Get foot velocities [4, 3] in world frame."""
        velocities = np.zeros((4, 3))
        for i, (leg, body_id) in enumerate(self.foot_body_ids.items()):
            velocities[i] = self.mj_data.cvel[body_id][:3].copy()
        return velocities

    def set_joint_torque(self, torques: np.ndarray) -> None:
        """Set joint torques.

        Torques are ordered as JOINT_NAMES (FL,FR,RL,RR), but actuators
        in the MuJoCo model are ordered (FR,FL,RR,RL).  Reorder before
        writing into ctrl[].
        """
        # WBC torque order  : [FLh,FLt,FLc, FRh,FRt,FRc, RLh,RLt,RLc, RRh,RRt,RRc]
        # MuJoCo actuator order: [FRh,FRt,FRc, FLh,FLt,FLc, RRh,RRt,RRc, RLh,RLt,RLc]
        perm = [3, 4, 5,  0, 1, 2,  9, 10, 11,  6, 7, 8]
        for i in range(12):
            self.mj_data.ctrl[self.actuator_ids[self.ACTUATOR_NAMES[i]]] = \
                np.clip(torques[perm[i]], -180.0, 180.0)

    def step(self, torques: Optional[np.ndarray] = None) -> RobotState:
        """Step simulation."""
        if torques is not None:
            self.set_joint_torque(torques)
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.update_state()
        return self.state

    @staticmethod
    def _quat_to_rot(quat: np.ndarray) -> np.ndarray:
        """Quaternion to rotation matrix."""
        w, x, y, z = quat
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y+2*z*w, 2*x*z-2*y*w],
            [2*x*y-2*z*w, 1-2*x*x-2*z*z, 2*y*z+2*x*w],
            [2*x*z+2*y*w, 2*y*z-2*x*w, 1-2*x*x-2*y*y]
        ])

    @property
    def n_joints(self) -> int:
        return 12

    @property
    def n_contacts(self) -> int:
        return 4
