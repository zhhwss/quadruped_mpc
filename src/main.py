"""
Main Quadruped MPC Controller - Unitree A2.

This is the main entry point that integrates MPC and WBC for
quadruped locomotion in MuJoCo simulation.
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# Force unbuffered stdout so logs appear in real-time even with viewer
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Add src directory to path so modules can be imported
src_dir = str(Path(__file__).parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import mujoco

from robot.a2_robot import A2Robot
from gait.gait import GaitGenerator, GaitParams, GaitType
from mpc.mpc import MPCController, MPCParams, CentroidalDynamics
from wbc.wbc import WBCController, WBCParams
from utils.sim_utils import DataLogger


class QuadrupedController:
    """
    Main controller combining MPC and WBC.

    Architecture:
    1. Gait Generator: Determines foot contact schedule and swing trajectories
    2. MPC: Optimizes CoM trajectory and contact forces over receding horizon
    3. WBC: Computes joint torques for task prioritization
    """

    def __init__(
        self,
        model_path: str,
        mpc_params: Optional[MPCParams] = None,
        wbc_params: Optional[WBCParams] = None,
        gait_params: Optional[GaitParams] = None,
    ):
        """Initialize the quadruped controller."""
        # Load robot
        self.robot = A2Robot(model_path)
        self.robot.reset()

        # Initialize controllers
        self.mpc = MPCController(
            params=mpc_params if mpc_params else MPCParams()
        )
        self.wbc = WBCController(
            params=wbc_params if wbc_params else WBCParams(),
        )
        self.gait = GaitGenerator(
            gait_params=gait_params if gait_params else GaitParams()
        )

        # State tracking
        self._step_count = 0
        self._last_time = 0.0

        # Data logging
        self.logger = DataLogger()

        # Command state
        self._command = np.zeros(3)  # [vx, vy, yaw_rate]

    def set_command(self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0) -> None:
        """
        Set velocity command.

        Args:
            vx: Forward velocity [m/s]
            vy: Lateral velocity [m/s]
            yaw_rate: Yaw rate [rad/s]
        """
        self._command = np.array([vx, vy, yaw_rate])

    def update(self, dt: float) -> np.ndarray:
        """
        Main control update.

        Args:
            dt: Time step [s]

        Returns:
            Joint torques [12]
        """
        # 1. Get robot state
        state = self.robot.update_state()

        # 2. Update gait (skip if standing still)
        cmd_speed = np.linalg.norm(self._command)
        is_standing = cmd_speed < 0.01

        if not is_standing:
            self.gait.step(dt)
            self.gait.update_foot_positions(self.robot.get_foot_positions())

        # 3. Compute CoM state
        com_pos = state.base_pos.copy()
        com_vel = state.base_lin_vel.copy()
        rpy = self.robot.get_rpy()
        ang_vel = state.base_ang_vel.copy()

        # 4. Compute desired foot positions (Raibert heuristic)
        com_state = (com_pos, com_vel)
        foot_positions = self.robot.get_foot_positions()
        foot_velocities = self.robot.get_foot_velocities()

        if is_standing:
            # All legs stay in stance, no swing
            contact_states = np.ones(4, dtype=bool)
            self.gait._contact_states = contact_states  # sync for debug display
            foot_des_pos = foot_positions.copy()
            foot_des_vel = np.zeros((4, 3))
        else:
            foot_des_pos, foot_des_vel, contact_states = \
                self.gait.compute_desired_foot_state(
                    com_state,
                    state.base_lin_vel,
                )

            # 5. Update swing foot trajectories
            for i in range(4):
                if not contact_states[i]:
                    swing_phase = self.gait.scheduler.get_swing_phase(i)
                    if swing_phase >= 0:
                        start = foot_positions[i]
                        end = foot_des_pos[i]
                        pos, vel = self.gait.get_swing_trajectory(
                            i, swing_phase, start, end
                        )
                        foot_des_pos[i] = pos
                        foot_des_vel[i] = vel

        # 6. Update MPC reference
        com_des = com_pos.copy()
        com_des[0] += self._command[0] * 0.5  # Simple lookahead
        com_des[1] += self._command[1] * 0.5
        com_des[2] = 0.47  # Target height for A2

        self.mpc.set_reference(com_des)

        # 7. Solve MPC (simple force distribution)
        foot_pos_rel = foot_positions - com_pos[:, None].T
        mpc_forces, _ = self.mpc.solve(
            np.concatenate([com_pos, com_vel, rpy, ang_vel]),
            foot_pos_rel,
            contact_states,
        )
        self._last_mpc_forces = mpc_forces.copy()  # 保存用于调试

        # 8. WBC → q_des, v_des, T_des → PD
        # WBC computes per-leg targets from MPC + gait
        q_des = np.zeros(12)
        v_des = np.zeros(12)
        T_des = np.zeros(12)

        # Gravity bias (G_j from mj_rne)
        G_j = self._compute_gravity_bias()

        for i in range(4):
            s = i * 3
            if contact_states[i]:
                # Stance: standing pose + MPC force feedforward
                q_des[s:s+3] = self.robot.stand_joint_pos[s:s+3]
                v_des[s:s+3] = 0.0
                # T_des = G_j - Jc^T * f_mpc (from floating-base EOM: h = Sᵀτ + Jᵀf)
                Jj = self._foot_jacobian(i)[:, s:s+3]  # [3,3] for this leg only
                T_des[s:s+3] = G_j[s:s+3] - Jj.T @ mpc_forces[i]
            else:
                # Swing: IK foot→joint + gravity bias (no contact force)
                q_swing = self._solve_leg_ik(i, foot_des_pos[i])
                q_des[s:s+3] = q_swing
                v_des[s:s+3] = 0.0
                T_des[s:s+3] = G_j[s:s+3]  # gravity compensation, no contact force

        # PD: tau = kp*(q_des - q) + kd*(v_des - v) + T_des
        kp_j, kd_j = 400.0, 12.0
        torques = kp_j * (q_des - state.joint_pos) + kd_j * (v_des - state.joint_vel) + T_des
        torques = np.clip(torques, -120, 120)
        self._last_torques = torques.copy()
        self._last_q_des = q_des.copy()

        info = {'Td_max': float(np.max(np.abs(T_des))),
                'q_err_max': float(np.max(np.abs(q_des - state.joint_pos)))}
        self._last_info = info

        # 11. Log data
        self.logger.log(
            self._last_time,
            step=self._step_count,
            com_pos=com_pos,
            com_vel=com_vel,
            contact_states=contact_states,
            foot_des_pos=foot_des_pos,
            mpc_forces=mpc_forces,
            joint_torques=torques,
        )

        self._step_count += 1
        self._last_time += dt

        return torques

    def run_simulation(
        self,
        duration: float = 10.0,
        dt: float = 0.002,
        render: bool = True,
        command: Optional[np.ndarray] = None,
        debug: bool = False,
        debug_interval: int = 100,
    ) -> None:
        """
        Run simulation for specified duration.

        Args:
            duration: Simulation duration [s]
            dt: Control timestep [s]
            render: Enable MuJoCo viewer rendering
            command: Optional velocity command [vx, vy, yaw_rate]
            debug: Print detailed debug logs
            debug_interval: Print debug every N steps
        """
        if command is not None:
            self.set_command(*command)

        # Setup MuJoCo viewer if requested
        viewer = None
        if render:
            try:
                from mujoco.viewer import launch_passive
                # launch_passive: viewer 在独立线程, 不阻塞, 主线程驱动物理
                viewer = launch_passive(self.robot.mj_model, self.robot.mj_data)
                print("Viewer launched")
                try:
                    viewer.cam.type = mujoco.mjtCam.mjCAM_TRACKING
                    viewer.cam.trackbodyid = self.robot.mj_model.body('base_link').id
                    viewer.cam.distance = 1.5
                    viewer.cam.elevation = -10.0
                except Exception:
                    pass
                print("Viewer enabled - close window to stop")
            except Exception as e:
                print(f"Could not create viewer: {e}")
                print("Running headless...")
                render = False
        print("start simulation")
        n_steps = int(duration / dt)
        print(f"Running simulation: {n_steps} steps, {duration}s")
        print(f"Model: Unitree A2 | Gait: {self.gait.params.gait_type.value}")
        print(f"Command: vx={self._command[0]:.2f}, vy={self._command[1]:.2f}, yaw={self._command[2]:.2f}")
        if debug:
            print(f"[DEBUG] Mode: ON | Interval: every {debug_interval} steps")
            print(f"[DEBUG] Controller: joint PD + gravity bias (mj_rne)")
            print(f"[DEBUG] MPC: height_target=0.48, friction={self.mpc.params.friction_coeff:.2f}")
            print(f"[DEBUG] Gait: period={self.gait.params.step_period:.2f}s, "
                  f"stance={self.gait.params.stance_duration:.2f}s, "
                  f"swing={self.gait.params.swing_duration:.2f}s")

        start_time = time.time()
        for step in range(n_steps):
            # Control
            torques = self.update(dt)

            # Step simulation
            self.robot.step(torques)

            # Sync viewer: 每物理步渲染一次, 与物理世界时间严格同步
            if render and viewer is not None:
                if viewer.is_running():
                    viewer.sync()
                else:
                    print("Viewer closed, stopping simulation.")
                    break

            # Flush stdout periodically so logs appear with viewer active
            if step % 50 == 0:
                sys.stdout.flush()

            # Print progress
            if step % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {step}/{n_steps} | "
                      f"CoM: ({self.robot.state.base_pos[0]:.2f}, "
                      f"{self.robot.state.base_pos[1]:.2f}, "
                      f"{self.robot.state.base_pos[2]:.2f}) | "
                      f"FPS: {step/max(elapsed,1e-6):.0f}", flush=True)

            # Detailed debug output
            if debug and step % debug_interval == 0 and step > 0:
                self._print_debug_info(step, dt)

        print("Simulation complete", flush=True)

    def _compute_gravity_bias(self) -> np.ndarray:
        """Use MuJoCo inverse dynamics to compute joint torques needed
        to hold the standing pose against gravity (qacc=0, qvel=0)."""
        m, d = self.robot.mj_model, self.robot.mj_data

        # Save current state
        qpos_save = d.qpos.copy()
        qvel_save = d.qvel.copy()

        # Set to standing pose with zero velocity
        for i, name in enumerate(self.robot.JOINT_NAMES):
            d.qpos[self.robot.joint_qposadr[name]] = self.robot.stand_joint_pos[i]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        # mj_rne with qacc=0 gives generalized force to hold pose against gravity
        tau_full = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, tau_full)  # qacc=0 → pure gravity compensation

        # Restore original state
        d.qpos[:] = qpos_save
        d.qvel[:] = qvel_save
        mujoco.mj_forward(m, d)

        return tau_full[6:18].copy()  # joint part only

    def _solve_leg_ik(self, leg_id: int, foot_target_world: np.ndarray) -> np.ndarray:
        """Quick IK for one leg: Newton step using MuJoCo numerical Jacobian.

        Args:
            leg_id: 0=FL, 1=FR, 2=RL, 3=RR
            foot_target_world: desired foot position in world frame [3]
        Returns:
            joint angles [3] for this leg
        """
        s = leg_id * 3
        q0 = self.robot.state.joint_pos[s:s+3].copy()
        leg_names = ['FL', 'FR', 'RL', 'RR']
        body_id = self.robot.mj_model.body(self.robot.FOOT_BODY_NAMES[leg_names[leg_id]]).id
        foot_current = self.robot.mj_data.xpos[body_id].copy()

        # Newton step: Δq = pinv(J) * (p_des - p_cur)
        J = self._foot_jacobian(leg_id)[:, s:s+3]  # [3,3] for this leg
        err = foot_target_world - foot_current
        try:
            JTJ = J.T @ J + np.eye(3) * 0.01
            dq = np.linalg.solve(JTJ, J.T @ err)
        except np.linalg.LinAlgError:
            dq = np.zeros(3)

        q_new = q0 + dq
        # Clip to joint limits
        limits = [(-1.01, 1.01), (-2.34, 3.15), (-2.77, -0.54)]
        return np.clip(q_new, [l[0] for l in limits], [l[1] for l in limits])

    def _foot_jacobian(self, leg_id: int) -> np.ndarray:
        """Compute 3×12 foot Jacobian using MuJoCo numerical Jacobian.

        Args:
            leg_id: 0=FL, 1=FR, 2=RL, 3=RR

        Returns:
            J [3, 12] mapping joint velocities to foot linear velocity (world frame)
        """
        leg_names = ['FL', 'FR', 'RL', 'RR']
        body_name = self.robot.FOOT_BODY_NAMES[leg_names[leg_id]]
        body_id = self.robot.mj_model.body(body_name).id

        jac_full = np.zeros((3, self.robot.mj_model.nv))
        mujoco.mj_jacBody(
            self.robot.mj_model, self.robot.mj_data,
            jac_full, None, body_id
        )
        # Return joint part (columns 6:18, skipping 6 floating-base DOFs)
        return jac_full[:, 6:18].copy()

    def save_data(self, path: str) -> None:
        """Save logged data."""
        self.logger.save(path)

    def _print_debug_info(self, step: int, dt: float) -> None:
        """Print detailed debug information."""
        state = self.robot.state
        print(f"\n[DEBUG] === Step {step} (t={step*dt:.2f}s) ===")
        print(f"[DEBUG] CoM  pos : ({state.base_pos[0]:.3f}, {state.base_pos[1]:.3f}, {state.base_pos[2]:.3f})")
        print(f"[DEBUG] CoM  vel : ({state.base_lin_vel[0]:.3f}, {state.base_lin_vel[1]:.3f}, {state.base_lin_vel[2]:.3f})")
        rpy = self.robot.get_rpy()
        print(f"[DEBUG] RPY      : roll={rpy[0]*180/np.pi:.1f}° pitch={rpy[1]*180/np.pi:.1f}° yaw={rpy[2]*180/np.pi:.1f}°")
        print(f"[DEBUG] Ang vel  : ({state.base_ang_vel[0]:.3f}, {state.base_ang_vel[1]:.3f}, {state.base_ang_vel[2]:.3f})")

        # Joint states
        print(f"[DEBUG] Joint pos: {np.array2string(state.joint_pos, precision=3)}")
        print(f"[DEBUG] Joint vel: {np.array2string(state.joint_vel, precision=3)}")

        # Contact states
        contact_names = ['FL', 'FR', 'RL', 'RR']
        contact_str = ' '.join(f"{n}:{int(self.gait.contact_states[i])}" for i, n in enumerate(contact_names))
        print(f"[DEBUG] Contacts  : {contact_str}")

        # Gait phase
        print(f"[DEBUG] Gait phase: {self.gait.phase:.3f}")

        # Foot positions
        foot_names = ['FL', 'FR', 'RL', 'RR']
        for i, name in enumerate(foot_names):
            fp = self.robot.get_foot_positions()[i]
            print(f"[DEBUG] Foot {name} pos: ({fp[0]:.3f}, {fp[1]:.3f}, {fp[2]:.3f})")

        # MPC forces (from log)
        if hasattr(self.logger, '_buffer') and 'mpc_forces' in self.logger._buffer:
            last_forces = self.logger._buffer['mpc_forces']
            if last_forces is not None and last_forces.size > 0:
                print(f"[DEBUG] MPC forces shape: {last_forces.shape}, ndim: {last_forces.ndim}")
                # Handle both array and scalar cases
                if last_forces.ndim == 2 and last_forces.shape[0] == 4 and last_forces.shape[1] == 3:
                    print(f"[DEBUG] MPC forces:")
                    for i, name in enumerate(foot_names):
                        f = last_forces[i]
                        print(f"[DEBUG]   {name}: ({f[0]:.1f}, {f[1]:.1f}, {f[2]:.1f}) N")
                elif last_forces.ndim == 1 and last_forces.size == 4:
                    # Old format: [fz_FL, fz_FR, fz_RL, fz_RR]
                    print(f"[DEBUG] MPC forces (Fz only): "
                          f"FL={last_forces[0]:.1f}, FR={last_forces[1]:.1f}, "
                          f"RL={last_forces[2]:.1f}, RR={last_forces[3]:.1f} N")
                elif last_forces.ndim == 1 and last_forces.size == 12:
                    # Flattened [f0x,f0y,f0z,f1x,f1y,f1z,...]
                    print(f"[DEBUG] MPC forces (flattened):")
                    for i, name in enumerate(foot_names):
                        idx = i * 3
                        print(f"[DEBUG]   {name}: ({last_forces[idx]:.1f}, {last_forces[idx+1]:.1f}, {last_forces[idx+2]:.1f}) N")
                else:
                    print(f"[DEBUG] MPC forces: {last_forces}")

        # Torques
        if hasattr(self, '_last_torques'):
            t = self._last_torques
            print(f"[DEBUG] Torques   : {np.array2string(t, precision=2)}")
            print(f"[DEBUG] Torque max: {np.max(np.abs(t)):.2f} Nm")

        # MPC force vs weight check
        mpc_total_fz = 0.0
        for i in range(4):
            if self.gait.contact_states[i]:
                if hasattr(self, '_last_mpc_forces'):
                    mpc_total_fz += self._last_mpc_forces[i, 2]
        robot_weight = self.mpc.dynamics.mass * 9.81
        n_stance = int(self.gait.contact_states.sum())
        print(f"[DEBUG] MPC total Fz: {mpc_total_fz:.1f} N | Robot weight: {robot_weight:.1f} N | Stance legs: {n_stance}")

        # Controller info
        if hasattr(self, '_last_info'):
            _info = self._last_info
            print(f"[DEBUG] Control   : bias={_info.get('bias_max',0):.0f} Nm, torque_max={_info.get('torque_max',0):.0f} Nm")


def main():
    parser = argparse.ArgumentParser(description='Quadruped MPC Controller')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Simulation duration (s)')
    parser.add_argument('--gait', type=str, default='trot',
                        choices=['trot', 'walk', 'gallop', 'bound', 'stand'],
                        help='Gait type')
    parser.add_argument('--command', type=float, nargs=3, default=[0.3, 0.0, 0.0],
                        help='Velocity command [vx, vy, yaw_rate]')
    parser.add_argument('--headless', action='store_true',
                        help='Run without rendering')
    parser.add_argument('--debug', action='store_true',
                        help='Enable detailed debug output')
    parser.add_argument('--debug-interval', type=int, default=100,
                        help='Debug print interval (steps)')
    args = parser.parse_args()

    # Paths
    model_dir = Path(__file__).parent.parent / 'models'
    model_path = str(model_dir / 'a2_scene.xml')

    # Configure gait
    gait_params = GaitParams(gait_type=GaitType(args.gait))

    # Create controller
    controller = QuadrupedController(
        model_path=model_path,
        gait_params=gait_params,
    )

    # Set command
    controller.set_command(*args.command)

    # Run
    controller.run_simulation(
        duration=args.duration,
        render=not args.headless,
        debug=args.debug,
        debug_interval=args.debug_interval,
    )

    # Save data
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    controller.save_data(str(output_dir / 'simulation_data.json'))
    print(f"Data saved to {output_dir / 'simulation_data.json'}")


if __name__ == '__main__':
    main()
