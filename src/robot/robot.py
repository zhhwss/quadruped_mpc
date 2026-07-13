"""
Unitree A1 Quadruped Robot Model Module.

This module provides the robot interface for the Unitree A1 quadruped,
including kinematics, dynamics, state estimation, and MuJoCo integration.
"""

import numpy as np
import mujoco
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class GaitType(Enum):
    """Supported gait types for quadruped locomotion."""
    TROT = "trot"
    WALK = "walk"
    GALLOP = "gallop"
    BOUND = "bound"
    PRONK = "pronk"
    STAND = "stand"


@dataclass
class RobotState:
    """Complete robot state container."""
    # Base pose
    base_pos: np.ndarray = None  # [x, y, z] position of CoM
    base_quat: np.ndarray = None  # [w, x, y, z] orientation quaternion

    # Base velocities
    base_lin_vel: np.ndarray = None  # [vx, vy, vz] linear velocity in body frame
    base_ang_vel: np.ndarray = None  # [wx, wy, wz] angular velocity in body frame

    # Joint states
    joint_pos: np.ndarray = None  # [q1..q12] joint positions
    joint_vel: np.ndarray = None  # [dq1..dq12] joint velocities
    joint_torque: np.ndarray = None  # [tau1..tau12] joint torques

    # Foot states
    foot_pos: np.ndarray = None  # [4, 3] foot positions in world frame
    foot_vel: np.ndarray = None  # [4, 3] foot velocities in world frame
    foot_force: np.ndarray = None  # [4, 3] ground reaction forces
    contact_state: np.ndarray = None  # [4] boolean contact states

    def __post_init__(self):
        """Initialize default arrays if not provided."""
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
        if self.foot_pos is None:
            self.foot_pos = np.zeros((4, 3))
        if self.foot_vel is None:
            self.foot_vel = np.zeros((4, 3))
        if self.foot_force is None:
            self.foot_force = np.zeros((4, 3))
        if self.contact_state is None:
            self.contact_state = np.zeros(4, dtype=bool)


@dataclass
class RobotParams:
    """Physical parameters of Unitree A1 robot."""
    # Mass properties
    mass: float = 12.0  # kg
    inertia: np.ndarray = None  # [3, 3] base inertia tensor

    # Hip positions relative to CoM [m]
    hip_positions: np.ndarray = None  # [4, 3] FL, FR, RL, RR

    # Leg parameters
    upper_leg_length: float = 0.213  # m (hip to knee)
    lower_leg_length: float = 0.213  # m (knee to foot)
    leg_link_masses: Tuple[float, float] = (1.0, 0.5)  # kg

    # Joint limits [rad]
    hip_joint_limits: Tuple[float, float] = (-1.5, 4.0)
    thigh_joint_limits: Tuple[float, float] = (-1.5, 4.0)
    calf_joint_limits: Tuple[float, float] = (-2.5, -0.1)

    # Velocity limits [rad/s]
    joint_vel_limits: Tuple[float, float] = (-30.0, 30.0)

    # Torque limits [Nm]
    joint_torque_limits: Tuple[float, float] = (-35.0, 35.0)

    # Motor parameters
    motor_kt: float = 0.05  # Torque constant [Nm/A]
    gear_ratio: float = 6.0  # Quaternion drive gear ratio

    # Friction
    friction_coeff: float = 0.7  # Foot-ground friction coefficient

    def __post_init__(self):
        """Initialize default parameters."""
        if self.inertia is None:
            self.inertia = np.diag([0.014, 0.028, 0.039])
        if self.hip_positions is None:
            # FL, FR, RL, RR in [x, y, z]
            self.hip_positions = np.array([
                [ 0.183,  0.047, 0.0],   # FL
                [ 0.183, -0.047, 0.0],   # FR
                [-0.183,  0.047, 0.0],   # RL
                [-0.183, -0.047, 0.0],   # RR
            ])

        self.n_legs = 4
        self.n_joints = 12  # 3 joints per leg


class A1Robot:
    """
    Unitree A1 quadruped robot interface.

    This class provides the interface between the MuJoCo simulation
    and the control algorithms (MPC, WBC). It handles:
    - Robot state extraction and estimation
    - Forward/inverse kinematics
    - Jacobian computation
    - Contact force estimation
    - Simulation stepping

    Attributes:
        params (RobotParams): Physical robot parameters
        state (RobotState): Current robot state
        mj_model: MuJoCo model
        mj_data: MuJoCo data
    """

    def __init__(self, model_path: str, params: Optional[RobotParams] = None):
        """
        Initialize the A1 robot.

        Args:
            model_path: Path to the MJCF/URDF model file
            params: Optional custom robot parameters
        """
        self.params = params if params is not None else RobotParams()

        # Initialize MuJoCo
        self._load_model(model_path)
        self._initialize_buffers()

        # Cache for Jacobians
        self._jacobian_cache: Dict[str, np.ndarray] = {}

    def _load_model(self, model_path: str) -> None:
        """Load MuJoCo model from file."""
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Get joint and actuator IDs
        self._init_joint_ids()
        self._init_actuator_ids()
        self._init_contact_ids()

    def _init_joint_ids(self) -> None:
        """Initialize joint ID mappings."""
        # Expected joint names: hip_joint, thigh_joint, calf_joint for each leg
        self.joint_names = []
        for leg in ['FL', 'FR', 'RL', 'RR']:
            for joint in ['hip_joint', 'thigh_joint', 'calf_joint']:
                name = f"{leg}_{joint}"
                self.joint_names.append(name)

        self.joint_ids = []
        for name in self.joint_names:
            try:
                self.joint_ids.append(self.mj_model.joint(name).id)
            except KeyError:
                raise KeyError(f"Joint '{name}' not found in model")

    def _init_actuator_ids(self) -> None:
        """Initialize actuator ID mappings."""
        self.actuator_names = self.joint_names.copy()
        self.actuator_ids = []
        for name in self.actuator_names:
            try:
                self.actuator_ids.append(self.mj_model.actuator(name).id)
            except KeyError:
                raise KeyError(f"Actuator '{name}' not found in model")

    def _init_contact_ids(self) -> None:
        """Initialize foot contact sensor IDs."""
        self.foot_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self.foot_body_ids = []
        for name in self.foot_names:
            try:
                self.foot_body_ids.append(self.mj_model.body(name).id)
            except KeyError:
                raise KeyError(f"Foot body '{name}' not found in model")

    def _initialize_buffers(self) -> None:
        """Initialize internal buffer arrays."""
        self.state = RobotState()
        self._contact_force_buffer = np.zeros((len(self.foot_names), 3))
        self._foot_pos_buffer = np.zeros((len(self.foot_names), 3))
        self._foot_vel_buffer = np.zeros((len(self.foot_names), 6))

    def update_state(self) -> RobotState:
        """
        Extract and update robot state from MuJoCo simulation.

        Returns:
            Updated RobotState object
        """
        # Base pose
        self.state.base_pos = self.mj_data.xpos[self.mj_model.body('base').id].copy()
        self.state.base_quat = self.mj_data.xquat[self.mj_model.body('base').id].copy()

        # Base velocities (transform to body frame)
        lin_vel_world = self.mj_data.cvel[self.mj_model.body('base').id][:3].copy()
        ang_vel_world = self.mj_data.cvel[self.mj_model.body('base').id][3:].copy()

        R = self.rotation_matrix_from_quat(self.state.base_quat)
        self.state.base_lin_vel = R.T @ lin_vel_world
        self.state.base_ang_vel = R.T @ ang_vel_world

        # Joint states
        for i, joint_id in enumerate(self.joint_ids):
            self.state.joint_pos[i] = self.mj_data.qpos[joint_id].copy()
            self.state.joint_vel[i] = self.mj_data.qvel[joint_id].copy()

        # Foot states
        self._update_foot_state()

        return self.state

    def _update_foot_state(self) -> None:
        """Update foot position, velocity, and contact state."""
        for i, body_id in enumerate(self.foot_body_ids):
            # Position
            self.state.foot_pos[i] = self.mj_data.xpos[body_id].copy()

            # Velocity (linear + angular)
            self._foot_vel_buffer[i] = self.mj_data.cvel[body_id].copy()
            self.state.foot_vel[i] = self._foot_vel_buffer[i, :3].copy()

            # Contact detection and force estimation
            self._update_contact_force(i)

    def _update_contact_force(self, foot_idx: int) -> None:
        """Update contact force for a specific foot."""
        contact = np.zeros(3)
        for j in range(self.mj_data.ncon):
            c = self.mj_data.contact[j]
            foot_id = self.mj_model.body(self.foot_names[foot_idx]).id
            if c.geom1 == foot_id or c.geom2 == foot_id:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.mj_model, self.mj_data, j, f)
                contact += f[:3]

        self.state.foot_force[foot_idx] = contact.copy()
        self.state.contact_state[foot_idx] = np.linalg.norm(contact) > 1.0

    def get_foot_position_in_base_frame(self) -> np.ndarray:
        """
        Get foot positions expressed in base frame.

        Returns:
            Foot positions [4, 3] in base frame
        """
        foot_pos_world = self.state.foot_pos
        R = self.rotation_matrix_from_quat(self.state.base_quat)
        foot_pos_base = (R.T @ (foot_pos_world - self.state.base_pos[:, None]).T).T
        return foot_pos_base

    def get_foot_velocity_in_base_frame(self) -> np.ndarray:
        """
        Get foot velocities expressed in base frame.

        Returns:
            Foot velocities [4, 3] in base frame
        """
        R = self.rotation_matrix_from_quat(self.state.base_quat)
        foot_vel_base = (R.T @ self.state.foot_vel.T).T
        return foot_vel_base

    def compute_jacobian(self, leg_id: int) -> np.ndarray:
        """
        Compute foot Jacobian for specified leg.

        The Jacobian maps joint velocities to foot velocity in world frame:
        v_foot = J @ qdot

        Args:
            leg_id: Leg index (0=FL, 1=FR, 2=RL, 3=RR)

        Returns:
            Jacobian matrix [3, 3] for the leg
        """
        if f"leg_{leg_id}" in self._jacobian_cache:
            return self._jacobian_cache[f"leg_{leg_id}"].copy()

        # Get relevant joints for this leg
        joint_start = leg_id * 3
        joint_ids = self.joint_ids[joint_start:joint_start + 3]

        # Compute using MuJoCo's Jacobian
        foot_name = self.foot_names[leg_id]
        foot_id = self.mj_model.body(foot_name).id

        jacp = np.zeros((3, self.mj_model.nv))  # Position Jacobian
        jacr = np.zeros((3, self.mj_model.nv))  # Rotation Jacobian

        mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp, jacr, foot_id)

        # Extract only the 3 joints of this leg
        J = jacp[:, joint_ids].copy()
        self._jacobian_cache[f"leg_{leg_id}"] = J

        return J

    def inverse_kinematics(
        self,
        leg_id: int,
        target_pos: np.ndarray,
        base_pos: Optional[np.ndarray] = None,
        base_quat: Optional[np.ndarray] = None,
        max_iter: int = 10,
        tol: float = 1e-4,
    ) -> np.ndarray:
        """
        Compute joint positions for target foot position using IK.

        Uses numerical IK via MuJoCo's inverse kinematics.

        Args:
            leg_id: Leg index
            target_pos: Desired foot position [3] in base frame
            base_pos: Optional base position override
            base_quat: Optional base quaternion override
            max_iter: Maximum IK iterations
            tol: Position error tolerance

        Returns:
            Joint positions [3] for the leg
        """
        joint_start = leg_id * 3
        joint_ids = self.joint_ids[joint_start:joint_start + 3]

        # Save current state
        saved_qpos = self.mj_data.qpos.copy()

        # Target in world frame
        if base_pos is None:
            base_pos = self.state.base_pos
        if base_quat is None:
            base_quat = self.state.base_quat

        R = self.rotation_matrix_from_quat(base_quat)
        target_world = R @ target_pos + base_pos

        # Numerical IK using MuJoCo's inverse kinematics
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Don't constrain orientation

        for _ in range(max_iter):
            # Current foot position in world frame
            foot_id = self.mj_model.body(self.foot_names[leg_id]).id
            foot_pos = self.mj_data.xpos[foot_id].copy()

            error = target_world - foot_pos
            if np.linalg.norm(error) < tol:
                break

            # Get Jacobian
            J = self.compute_jacobian(leg_id)
            J_world = jacp[:, joint_ids]  # World frame Jacobian

            # Update joint positions (damped least squares)
            damping = 0.01
            JT = J_world.T
            dq = JT @ np.linalg.solve(J_world @ JT + damping * np.eye(3), error)

            # Apply update
            self.mj_data.qpos[joint_ids] += dq
            mujoco.mj_forward(self.mj_model, self.mj_data)

        # Extract result
        result = self.mj_data.qpos[joint_ids].copy()

        # Restore state
        self.mj_data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)

        return result

    def forward_kinematics(self, leg_id: int, joint_pos: np.ndarray) -> np.ndarray:
        """
        Compute foot position from joint angles using forward kinematics.

        Args:
            leg_id: Leg index
            joint_pos: Joint angles [3]

        Returns:
            Foot position [3] in base frame
        """
        hip_pos = self.params.hip_positions[leg_id]
        l1 = self.params.upper_leg_length
        l2 = self.params.lower_leg_length

        hip_joint, thigh_joint, calf_joint = joint_pos

        # Forward kinematics for 3-DOF leg
        # Hip joint rotation (around y-axis for FL/RL, -y-axis for FR/RR)
        if leg_id in [0, 2]:  # Left legs
            hip_angle = hip_joint
        else:  # Right legs
            hip_angle = -hip_joint

        cos_h = np.cos(hip_angle)
        sin_h = np.sin(hip_angle)

        # Thigh joint (around x-axis)
        cos_t = np.cos(thigh_joint)
        sin_t = np.sin(thigh_joint)

        # Calf joint (around x-axis)
        cos_c = np.cos(calf_joint)
        sin_c = np.sin(calf_joint)

        # Knee position (relative to hip)
        knee_pos_rel = np.array([0.0, l1 * sin_t, -l1 * cos_t])

        # Hip rotation of knee position
        knee_pos_rel_rot = np.array([
            knee_pos_rel[0],
            cos_h * knee_pos_rel[1] - sin_h * knee_pos_rel[2],
            sin_h * knee_pos_rel[1] + cos_h * knee_pos_rel[2]
        ])

        # Foot position relative to hip (including calf)
        calf_angle = thigh_joint + calf_joint
        cos_ca = np.cos(calf_angle)
        sin_ca = np.sin(calf_angle)

        foot_rel = np.array([
            -l2 * sin_ca,
            l1 * sin_t + l2 * sin_ca,
            -(l1 * cos_t + l2 * cos_ca)
        ])

        # Rotate by hip joint
        foot_rel_rot = np.array([
            foot_rel[0],
            cos_h * foot_rel[1] - sin_h * foot_rel[2],
            sin_h * foot_rel[1] + cos_h * foot_rel[2]
        ])

        foot_pos = hip_pos + foot_rel_rot

        return foot_pos

    def set_joint_torque(self, torques: np.ndarray) -> None:
        """
        Set joint torques to actuators.

        Args:
            torques: Joint torques [12]
        """
        torques = np.clip(
            torques,
            self.params.joint_torque_limits[0],
            self.params.joint_torque_limits[1]
        )

        for i, act_id in enumerate(self.actuator_ids):
            self.mj_data.ctrl[act_id] = torques[i]

    def set_joint_position(self, positions: np.ndarray) -> None:
        """
        Set joint position targets for position-controlled mode.

        Args:
            positions: Target joint positions [12]
        """
        for i, act_id in enumerate(self.actuator_ids):
            self.mj_data.ctrl[act_id] = positions[i]

    def step(self, action: Optional[np.ndarray] = None) -> RobotState:
        """
        Step the simulation forward.

        Args:
            action: Optional action to apply

        Returns:
            Updated robot state
        """
        if action is not None:
            self.set_joint_torque(action)

        mujoco.mj_step(self.mj_model, self.mj_data)
        self.update_state()

        return self.state

    def reset(self, qpos: Optional[np.ndarray] = None) -> RobotState:
        """
        Reset the simulation to initial state.

        Args:
            qpos: Optional custom initial joint positions

        Returns:
            Reset robot state
        """
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        if qpos is not None:
            self.mj_data.qpos[:] = qpos

        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.update_state()

        return self.state

    def get_com_velocity(self) -> np.ndarray:
        """
        Get center of mass velocity in world frame.

        Returns:
            CoM velocity [3]
        """
        return self.mj_data.qvel[0:3].copy()

    def get_com_acceleration(self) -> np.ndarray:
        """
        Get center of mass acceleration in world frame.

        Returns:
            CoM acceleration [3]
        """
        return self.mj_data.qacc[0:3].copy()

    def get_contact_info(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get detailed contact information for all feet.

        Returns:
            Dictionary mapping foot names to (position, force) tuples
        """
        contacts = {}
        for i, foot_name in enumerate(self.foot_names):
            if self.state.contact_state[i]:
                contacts[foot_name] = (
                    self.state.foot_pos[i].copy(),
                    self.state.foot_force[i].copy()
                )
        return contacts

    @staticmethod
    def rotation_matrix_from_quat(quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.

        Args:
            quat: Quaternion [w, x, y, z]

        Returns:
            Rotation matrix [3, 3]
        """
        w, x, y, z = quat
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

    @staticmethod
    def quat_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion.

        Args:
            R: Rotation matrix [3, 3]

        Returns:
            Quaternion [w, x, y, z]
        """
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])

    @staticmethod
    def rpy_from_quat(quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to roll-pitch-yaw angles.

        Args:
            quat: Quaternion [w, x, y, z]

        Returns:
            RPY angles [roll, pitch, yaw]
        """
        w, x, y, z = quat
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        return self.mj_model.nv

    @property
    def n_joints(self) -> int:
        """Number of actuated joints."""
        return len(self.joint_ids)

    @property
    def n_contacts(self) -> int:
        """Number of potential contact points (feet)."""
        return len(self.foot_names)
