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
            model=self.robot.mj_model,
            data=self.robot.mj_data,
            params=wbc_params if wbc_params else WBCParams(),
        )
        self.gait = GaitGenerator(
            gait_params=gait_params if gait_params else GaitParams()
        )

        # State tracking
        self._step_count = 0
        self._last_time = 0.0
        self._swing_start_pos = np.zeros((4, 3))
        self._last_contact_states = np.ones(4, dtype=bool)
        self._stance_target_pos = self.robot.get_foot_positions()

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

        Pipeline:
          1. State estimation (from MuJoCo)
          2. Gait scheduler + Raibert foot placement
          3. Swing trajectory generation (Bezier)
          4. MPC: height + velocity-tracking forces with friction cone
          5. Joint PD + gravity compensation + force feedforward + yaw stabilisation

        Args:
            dt: Time step [s]

        Returns:
            Joint torques [12]
        """
        # ── 1. Get robot state ────────────────────────────────────
        state = self.robot.update_state()
        com_pos = state.base_pos.copy()
        rpy = self.robot.get_rpy()

        # A2Robot returns velocities in BODY frame (it already applies R^T).
        # We need WORLD-frame velocities for MPC (velocity tracking uses world
        # commands) and BODY-frame for Raibert (Raibert is body-centric).
        R_body_to_world = self._rpy_to_rot(rpy)

        # Linear: body → world (MPC uses world frame for velocity tracking)
        com_vel_body = state.base_lin_vel.copy()    # body frame from A2Robot
        com_vel = R_body_to_world @ com_vel_body    # → world frame

        # Angular: body → world  (A2Robot already converted to body, so R is needed)
        ang_vel_body = state.base_ang_vel.copy()    # body frame from A2Robot
        ang_vel = R_body_to_world @ ang_vel_body    # → world frame

        foot_positions = self.robot.get_foot_positions()
        foot_velocities = self.robot.get_foot_velocities()

        # ── 2. Update gait ────────────────────────────────────────
        cmd_speed = np.linalg.norm(self._command[:2])
        is_standing = cmd_speed < 0.01

        if not is_standing:
            self.gait.step(dt)
            self.gait.update_foot_positions(foot_positions)

        # ── 3. Compute desired foot positions (Raibert) ───────────
        com_state = (com_pos, com_vel)

        if is_standing:
            contact_states = np.ones(4, dtype=bool)
            self.gait._contact_states = contact_states
            foot_des_pos = self._stance_target_pos.copy()
            foot_des_vel = np.zeros((4, 3))
        else:
            foot_des_pos, foot_des_vel, contact_states = \
                self.gait.compute_desired_foot_state(
                    com_state,
                    com_vel_body,  # BODY-frame velocity for Raibert (cmd is body-frame)
                    foot_positions=foot_positions,
                    command=self._command,
                    base_pos=com_pos,
                    base_rpy=rpy,
                )

            # ── 4. Swing trajectory (Bezier interpolation) ────────
            for i in range(4):
                if not contact_states[i]:
                    # Detect liftoff: transitioned from stance to swing
                    if self._last_contact_states[i]:
                        self._swing_start_pos[i] = foot_positions[i].copy()
                    
                    swing_phase = self.gait.scheduler.get_swing_phase(i)
                    if swing_phase >= 0:
                        start = self._swing_start_pos[i]
                        end = foot_des_pos[i]
                        pos, vel = self.gait.get_swing_trajectory(
                            i, swing_phase, start, end
                        )
                        foot_des_pos[i] = pos
                        foot_des_vel[i] = vel
                else:
                    # Stance leg: lock touchdown position to avoid kinematics drift
                    # Detect touchdown: transitioned from swing to stance
                    if self._last_contact_states[i] == False: # transitioned from swing
                        self._stance_target_pos[i] = foot_des_pos[i].copy()
                    foot_des_pos[i] = self._stance_target_pos[i].copy()
                    foot_des_vel[i] = np.zeros(3)

        # ── 5. MPC: compute reference contact forces ──────────────
        # MPC outputs desired contact forces F_mpc for velocity tracking
        # and height control.  These are fed directly to WBC as J^T*F_mpc
        # feedforward for stance legs.
        # State: [com_pos(3), com_vel(3), rpy(3), ang_vel(3)] = [12]
        foot_pos_rel = foot_positions - com_pos[:, None].T
        centroidal_state = np.concatenate([com_pos, com_vel, rpy, ang_vel])

        mpc_forces, _ = self.mpc.solve(
            centroidal_state,
            foot_pos_rel,
            contact_states,
            com_vel=com_vel,
            cmd_vel=self._command,
            rpy=rpy,
            ang_vel=ang_vel,
        )
        self._last_mpc_forces = mpc_forces.copy()
        self._last_mpc_debug = getattr(self.mpc, '_last_debug', {})

        # Compute joint-space targets for all legs using IK
        joint_pos_des = np.zeros(12)
        for i in range(4):
            joint_pos_des[i*3:i*3+3] = self._solve_leg_ik(i, foot_des_pos[i])
            if not contact_states[i] and self._step_count % 100 == 0:
                leg_names = ['FL', 'FR', 'RL', 'RR']
                s = i * 3
                print(f"[DEBUG SWING] Leg {leg_names[i]}:")
                print(f"  foot_des: {foot_des_pos[i]}")
                print(f"  foot_cur: {foot_positions[i]}")
                print(f"  q_des   : {joint_pos_des[s:s+3]}")
                print(f"  q_cur   : {state.joint_pos[s:s+3]}")

        # ── 6. WBC: QP-based whole-body control ──────────────────
        # Decision variables: x = [qdd (18), tau (12), Fc (3·n_contacts)]
        #   Cost: COM tracking + orientation + swing foot + force ref + τ reg
        #   Eq:   M·qdd + h = Sᵀ·tau + Jcᵀ·Fc  (dynamics)
        #         Jc·qdd = 0                     (stance no-slip)
        #   Ineq: friction cone, torque limits
        if is_standing:
            com_ref = np.array([0.0, 0.0, 0.43])
        else:
            com_ref = np.array([com_pos[0], com_pos[1], 0.43])  # desired CoM height
        torques, wbc_info = self.wbc.compute_joint_torques(
            com_pos=com_pos,
            com_vel=com_vel,
            com_rpy=rpy,
            com_ang_vel=ang_vel,
            foot_positions=foot_positions,
            foot_velocities=foot_velocities,
            foot_desired_positions=foot_des_pos,
            foot_desired_velocities=foot_des_vel,
            contact_states=contact_states,
            joint_positions=state.joint_pos,
            joint_velocities=state.joint_vel,
            mpc_forces=mpc_forces,
            com_ref=com_ref,
            joint_pos_des=joint_pos_des,
            is_walking=not is_standing,
        )

        # Yaw stabilisation is handled by WBC (orientation PD) and MPC (yaw force).
        # No extra yaw torque is added here to avoid triple-stacking.

        # ── 7. Store for debug ────────────────────────────────────
        self._last_torques = torques.copy()
        self._last_wbc_info = wbc_info
        self._last_foot_des_pos = foot_des_pos.copy()
        self._last_foot_pos = foot_positions.copy()
        self._last_contact_states = contact_states.copy()
        self._last_com_vel = com_vel.copy()
        self._last_rpy = rpy.copy()

        # ── 9. Log data ───────────────────────────────────────────
        self.logger.log(
            self._last_time,
            step=self._step_count,
            com_pos=com_pos,
            com_vel=com_vel,
            rpy=rpy,
            ang_vel=ang_vel,
            contact_states=contact_states,
            foot_pos=foot_positions,
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
            print(f"[DEBUG] Controller: PD joints + Raibert foot placement + MPC force FF + yaw stab")
            print(f"[DEBUG] MPC: height PD + velocity-tracking horizontal force + friction cone (μ={self.mpc.params.friction_coeff:.2f})")
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
            if step % 500 == 0:
                elapsed = time.time() - start_time
                state = self.robot.state
                v = state.base_lin_vel
                rpy = self.robot.get_rpy()
                cmd = self._command
                ev = cmd[0] - v[0]
                print(f"  Step {step:5d}/{n_steps} | "
                      f"CoM: ({state.base_pos[0]:.2f}, {state.base_pos[1]:.2f}, {state.base_pos[2]:.3f}) | "
                      f"Vbody:({v[0]:.3f},{v[1]:.3f}) cmd=({cmd[0]:.2f},{cmd[1]:.2f}) | "
                      f"Yaw:{rpy[2]*180/np.pi:+.1f}° | "
                      f"FPS:{step/max(elapsed,1e-6):.0f}", flush=True)

            # Detailed debug output
            if debug and step % debug_interval == 0 and step > 0:
                self._print_debug_info(step, dt)

        print("Simulation complete", flush=True)

    @staticmethod
    def _rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
        """Convert roll-pitch-yaw [rad] → rotation matrix (body→world)."""
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr],
        ])

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
        """Numerical IK for one leg with convergence loop (Newton-Raphson).

        Args:
            leg_id: 0=FL, 1=FR, 2=RL, 3=RR
            foot_target_world: desired foot position in world frame [3]
        Returns:
            joint angles [3] for this leg
        """
        s = leg_id * 3
        limits = [(-1.01, 1.01), (-2.34, 3.15), (-2.77, -0.54)]
        q_min = np.array([l[0] for l in limits])
        q_max = np.array([l[1] for l in limits])

        # Backup current state of this leg to restore later
        q_backup = self.robot.mj_data.qpos[7+s : 7+s+3].copy()

        q_sol = q_backup.copy()
        for _ in range(5):
            self.robot.mj_data.qpos[7+s : 7+s+3] = q_sol
            mujoco.mj_forward(self.robot.mj_model, self.robot.mj_data)

            foot_current = self.robot.get_foot_positions()[leg_id]
            err = foot_target_world - foot_current
            if np.linalg.norm(err) < 1e-4:
                break

            J = self._foot_jacobian(leg_id)[:, s:s+3]
            try:
                JTJ = J.T @ J + np.eye(3) * 0.001
                dq = np.linalg.solve(JTJ, J.T @ err)
            except np.linalg.LinAlgError:
                break

            q_sol = np.clip(q_sol + dq, q_min, q_max)

        # Restore state
        self.robot.mj_data.qpos[7+s : 7+s+3] = q_backup
        mujoco.mj_forward(self.robot.mj_model, self.robot.mj_data)

        return q_sol

    def _foot_jacobian(self, leg_id: int) -> np.ndarray:
        """Compute corrected 3×12 contact-point foot Jacobian.

        Args:
            leg_id: 0=FL, 1=FR, 2=RL, 3=RR

        Returns:
            J [3, 12] mapping joint velocities to foot linear velocity (world frame)
        """
        leg_names = ['FL', 'FR', 'RL', 'RR']
        body_name = self.robot.FOOT_BODY_NAMES[leg_names[leg_id]]
        body_id = self.robot.mj_model.body(body_name).id

        J_body = np.zeros((3, self.robot.mj_model.nv))
        J_rot  = np.zeros((3, self.robot.mj_model.nv))
        mujoco.mj_jacBody(
            self.robot.mj_model, self.robot.mj_data,
            J_body, J_rot, body_id
        )

        r_body   = np.array([0.0, 0.0, -0.275])
        xmat     = self.robot.mj_data.xmat[body_id].reshape(3, 3)
        r_world  = xmat @ r_body
        rx,ry,rz = r_world
        skew_r   = np.array([[0,-rz,ry],[rz,0,-rx],[-ry,rx,0]])

        J_full = J_body - skew_r @ J_rot
        # Return joint part (columns 6:18, skipping 6 floating-base DOFs)
        return J_full[:, 6:18].copy()

    def save_data(self, path: str) -> None:
        """Save logged data."""
        self.logger.save(path)

    def _print_debug_info(self, step: int, dt: float) -> None:
        """Print comprehensive debug information for root-cause analysis."""
        state = self.robot.state
        rpy = self.robot.get_rpy()
        foot_names = ['FL', 'FR', 'RL', 'RR']
        contact_names = ['FL', 'FR', 'RL', 'RR']

        print(f"\n{'='*70}")
        print(f"[DEBUG] Step {step} (t={step*dt:.2f}s) | Gait phase={self.gait.phase:.3f}")
        print(f"{'='*70}")

        # ═══════════════════════════════════════════════════════════
        # 1. STATE — CoM position, velocity, orientation
        # ═══════════════════════════════════════════════════════════
        print(f"[STATE] CoM pos  : ({state.base_pos[0]:.4f}, {state.base_pos[1]:.4f}, {state.base_pos[2]:.4f}) m")
        v_body = state.base_lin_vel
        v_cmd = self._command
        evx_body = v_cmd[0] - v_body[0]
        evy_body = v_cmd[1] - v_body[1]
        print(f"[STATE] CoM vel (body): ({v_body[0]:.4f}, {v_body[1]:.4f}, {v_body[2]:.4f}) m/s  |  "
              f"CMD:(vx={v_cmd[0]:.2f},vy={v_cmd[1]:.2f})  "
              f"Δv_body=({evx_body:+.3f},{evy_body:+.3f})  "
              f"|v|={np.linalg.norm(v_body[:2]):.3f}/{np.linalg.norm(v_cmd[:2]):.3f} m/s")
        v_world = getattr(self, '_last_com_vel', v_body)
        print(f"[STATE] CoM vel (world):({v_world[0]:.4f}, {v_world[1]:.4f}, {v_world[2]:.4f}) m/s")
        print(f"[STATE] RPY      : roll={rpy[0]*180/np.pi:+.2f}°  pitch={rpy[1]*180/np.pi:+.2f}°  yaw={rpy[2]*180/np.pi:+.2f}°")
        ang = state.base_ang_vel
        print(f"[STATE] Ang vel  : ({ang[0]:.4f}, {ang[1]:.4f}, {ang[2]:.4f}) rad/s")

        # ═══════════════════════════════════════════════════════════
        # 2. GAIT — Contact schedule & foot placement
        # ═══════════════════════════════════════════════════════════
        contact_str = ' '.join(
            f"{n}:{'STANCE' if self.gait.contact_states[i] else 'SWING '}"
            for i, n in enumerate(contact_names))
        print(f"[GAIT] Contacts  : {contact_str}")

        fp = self.robot.get_foot_positions()
        fdp = getattr(self, '_last_foot_des_pos', fp)
        for i, name in enumerate(foot_names):
            is_swing = not self.gait.contact_states[i]
            tag = "🦶" if not is_swing else "✈️ "
            f_cur = fp[i]
            f_des = fdp[i] if fdp is not None else f_cur
            err_xy = np.linalg.norm(f_des[:2] - f_cur[:2])
            err_z = f_des[2] - f_cur[2]
            if is_swing:
                print(f"[GAIT] {tag} {name}: cur=({f_cur[0]:.4f},{f_cur[1]:.4f},{f_cur[2]:.4f})  "
                      f"des=({f_des[0]:.4f},{f_des[1]:.4f},{f_des[2]:.4f})  "
                      f"err_xy={err_xy:.4f} err_z={err_z:+.4f}")
            else:
                print(f"[GAIT] {tag} {name}: cur=({f_cur[0]:.4f},{f_cur[1]:.4f},{f_cur[2]:.4f})  "
                      f"Fz_des={self._last_mpc_forces[i,2]:.1f}N")

        # ═══════════════════════════════════════════════════════════
        # 3. MPC — Forces (horizontal + vertical) & friction check
        # ═══════════════════════════════════════════════════════════
        print(f"[MPC] ── Diagnostic Forces (world frame, NOT fed to WBC) ──")
        mpc_f = self._last_mpc_forces
        total_f = np.zeros(3)
        total_fz = 0.0
        for i, name in enumerate(foot_names):
            f = mpc_f[i]
            fh_norm = np.linalg.norm(f[:2])
            fz = f[2]
            mu_used = fh_norm / max(fz, 1e-6)
            contact = self.gait.contact_states[i]
            if contact:
                total_f += f
                total_fz += fz
                mu_flag = " ⚠️SLIP" if mu_used > 0.7 else ""
            else:
                mu_flag = ""
            tag = "STANCE" if contact else "SWING "
            print(f"[MPC]   {name} [{tag}]: fx={f[0]:+7.1f} fy={f[1]:+7.1f} fz={f[2]:+7.1f} N  "
                  f"|fh|={fh_norm:.1f} μ={mu_used:.3f}{mu_flag}")
        robot_weight = self.mpc.dynamics.mass * 9.81
        n_stance = int(self.gait.contact_states.sum())
        print(f"[MPC]   Total: Σfx={total_f[0]:+.1f} Σfy={total_f[1]:+.1f} Σfz={total_fz:.1f} N  "
              f"vs weight={robot_weight:.1f} N  stance_legs={n_stance}")
        # MPC internal debug
        mpc_dbg = getattr(self, '_last_mpc_debug', {})
        if mpc_dbg:
            print(f"[MPC]   Internal: height_err={mpc_dbg.get('height_err',0):+.4f}m  "
                  f"fx_total={mpc_dbg.get('fx_total',0):+.1f}N  "
                  f"fy_total={mpc_dbg.get('fy_total',0):+.1f}N  "
                  f"evx_body={mpc_dbg.get('evx_body',0):+.3f} evy_body={mpc_dbg.get('evy_body',0):+.3f}  "
                  f"v_body=({mpc_dbg.get('vx_body',0):.3f},{mpc_dbg.get('vy_body',0):.3f})  "
                  f"yaw_tau={mpc_dbg.get('tau_yaw',0):+.2f}Nm")

        # ═══════════════════════════════════════════════════════════
        # 4. CONTROL — Joint positions, torques, yaw stabilisation
        # ═══════════════════════════════════════════════════════════
        print(f"[CTRL] Joint pos : {np.array2string(state.joint_pos, precision=3, suppress_small=True)}")
        t = self._last_torques
        print(f"[CTRL] Torques   : {np.array2string(t, precision=1, suppress_small=True)}")
        print(f"[CTRL] Torque max: {np.max(np.abs(t)):.1f} Nm  |  "
              f"limit: hip=±120 thigh=±120 calf=±180")
        wbc_info = getattr(self, '_last_wbc_info', {})
        if wbc_info:
            print(f"[CTRL] WBC: |tau|={wbc_info.get('tau_norm',0):.0f}Nm  "
                  f"|g|={wbc_info.get('tau_g_norm',0):.0f}  "
                  f"|ff|={wbc_info.get('tau_ff_norm',0):.0f}  "
                  f"|ori|={wbc_info.get('tau_ori_norm',0):.0f}  "
                  f"|sw|={wbc_info.get('tau_swing_norm',0):.0f}")

        # ═══════════════════════════════════════════════════════════
        # 5. FORCE CHAIN — F→a→v→p diagnostics
        # ═══════════════════════════════════════════════════════════
        print(f"[CHAIN] ── F→a→v→p diagnostic chain ──")
        m = self.mpc.dynamics.mass  # 40.071 kg
        g_vec = np.array([0.0, 0.0, -9.81])

        # ── F: MPC planned forces ─────────────────────────────────
        f_mpc = getattr(self, '_last_mpc_forces', np.zeros((4,3)))
        total_f_mpc = np.sum(f_mpc, axis=0)
        print(f"[CHAIN]   F_mpc (Σ): ({total_f_mpc[0]:+.1f}, {total_f_mpc[1]:+.1f}, {total_f_mpc[2]:+.1f}) N")

        # ── a_exp: expected COM accel from F_mpc ──────────────────
        a_exp = total_f_mpc / m + g_vec
        print(f"[CHAIN]   a_exp = ΣF/m + g = ({a_exp[0]:+.3f}, {a_exp[1]:+.3f}, {a_exp[2]:+.3f}) m/s²")

        # ── a_act: actual COM accel (finite diff) ──────────────────
        v_now = getattr(self, '_last_com_vel', np.zeros(3))
        v_prev = getattr(self, '_prev_com_vel', v_now.copy())
        a_act = (v_now - v_prev) / dt
        self._prev_com_vel = v_now.copy()
        print(f"[CHAIN]   a_act (fdiff)   = ({a_act[0]:+.3f}, {a_act[1]:+.3f}, {a_act[2]:+.3f}) m/s²")
        print(f"[CHAIN]   a_err = a_exp-a_act = ({a_exp[0]-a_act[0]:+.3f}, {a_exp[1]-a_act[1]:+.3f}, {a_exp[2]-a_act[2]:+.3f}) m/s²")

        # ── v: velocity tracking ──────────────────────────────────
        v_body = state.base_lin_vel
        v_world = v_now
        v_cmd_body = np.array([self._command[0], self._command[1], 0.0])
        print(f"[CHAIN]   v_act(body) = ({v_body[0]:+.4f}, {v_body[1]:+.4f}, {v_body[2]:+.4f}) m/s")
        print(f"[CHAIN]   v_act(world)= ({v_world[0]:+.4f}, {v_world[1]:+.4f}, {v_world[2]:+.4f}) m/s")
        print(f"[CHAIN]   v_cmd(body) = ({v_cmd_body[0]:+.2f}, {v_cmd_body[1]:+.2f}, {v_cmd_body[2]:+.2f}) m/s")

        # ── p: position tracking ──────────────────────────────────
        print(f"[CHAIN]   p_act = ({state.base_pos[0]:+.4f}, {state.base_pos[1]:+.4f}, {state.base_pos[2]:+.4f}) m")
        print(f"[CHAIN]   p_des_z = 0.470 m")

        # ── WBC torque decomposition ──────────────────────────────
        wbc_info = getattr(self, '_last_wbc_info', {})
        if wbc_info:
            print(f"[CHAIN]   τ: |base|={wbc_info.get('tau_base_norm',0):.1f} "
                  f"|tot|={wbc_info.get('tau_norm',0):.1f} Nm  "
                  f"a_des_z={wbc_info.get('a_des_z',0):+.1f} dz={wbc_info.get('dz',0):+.3f}")

        # ═══════════════════════════════════════════════════════════
        # 6. SUMMARY — Key metrics for triage
        # ═══════════════════════════════════════════════════════════
        print(f"[SUMMARY] ─────────────────────────────────────────")
        evx_body = v_cmd[0] - v_body[0]
        evy_body = v_cmd[1] - v_body[1]
        issues = []
        if abs(evx_body) > 0.08:
            issues.append(f"VX_body err {evx_body:+.3f} m/s")
        if abs(evy_body) > 0.08:
            issues.append(f"VY_body err {evy_body:+.3f} m/s")
        if abs(rpy[2]) > 0.08:
            issues.append(f"Yaw drift {rpy[2]*180/np.pi:+.1f}°")
        if abs(ang[2]) > 0.15:
            issues.append(f"Yaw rate {ang[2]:+.3f} rad/s")
        if abs(0.47 - state.base_pos[2]) > 0.03:
            issues.append(f"Height err {state.base_pos[2]-0.47:+.3f}m")
        if abs(rpy[0]) > 0.10:
            issues.append(f"Roll {rpy[0]*180/np.pi:+.1f}°")
        if abs(rpy[1]) > 0.10:
            issues.append(f"Pitch {rpy[1]*180/np.pi:+.1f}°")

        if issues:
            print(f"[SUMMARY] ⚠️  Issues: {' | '.join(issues)}")
        else:
            print(f"[SUMMARY] ✅ All metrics nominal")


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
