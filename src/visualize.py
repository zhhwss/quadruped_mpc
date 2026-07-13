"""Visualize simulation using MuJoCo's viewer.launch API."""

import mujoco
import time
import numpy as np
from pathlib import Path


def replay_simulation(
    model_path: str = "models/a2_scene.xml",
    data_path: str = "results/simulation_data.json",
    speed: float = 1.0,
):
    """
    Replay simulation in MuJoCo viewer.

    Args:
        model_path: Path to MuJoCo model XML
        data_path: Path to simulation data JSON
        speed: Replay speed multiplier
    """
    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Load recorded data
    with open(data_path) as f:
        import json
        recorded = json.load(f)

    steps = recorded['data']['step']
    joint_torques = np.array(recorded['data']['joint_torques'])
    com_positions = np.array(recorded['data']['com_pos'])

    print(f"Loaded {len(steps)} steps of simulation data")
    print(f"Starting replay at {speed}x speed...")
    print("Close the viewer window to exit.")

    # Launch viewer
    from mujoco.viewer import launch
    with launch(model, data) as viewer:
        # Set camera to track base
        try:
            viewer.cam.type = mujoco.mjtCam.mjCAM_TRACKING
            viewer.cam.trackbodyid = model.body('base_link').id
            viewer.cam.distance = 1.5
            viewer.cam.elevation = -10.0
        except:
            pass

        # Replay
        dt = 0.002 / speed  # simulation timestep adjusted for replay speed
        idx = 0

        while viewer.is_running() and idx < len(steps):
            # Apply recorded torques
            torque = joint_torques[idx]
            # Map torques to actuators
            for i, act_name in enumerate([
                'FR_hip', 'FR_thigh', 'FR_calf',
                'FL_hip', 'FL_thigh', 'FL_calf',
                'RR_hip', 'RR_thigh', 'RR_calf',
                'RL_hip', 'RL_thigh', 'RL_calf',
            ]):
                try:
                    data.actuator(i)[:] = torque[i]
                except:
                    pass

            # Step simulation
            mujoco.mj_step(model, data)

            # Render
            viewer.render()

            idx += 1
            time.sleep(max(0, dt - (time.time() - time.time())))

        print(f"Replay complete: {idx} steps")


def replay_with_control(
    model_path: str = "models/a2_scene.xml",
    data_path: str = "results/simulation_data.json",
    gait_type: str = "trot",
    speed: float = 0.3,
):
    """
    Run live simulation with MPC/WBC control and viewer.

    Args:
        model_path: Path to MuJoCo model
        gait_type: Gait pattern
        speed: Forward speed command [m/s]
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from robot.a2_robot import A2Robot
    from gait.gait import GaitGenerator, GaitParams, GaitType
    from mpc.mpc import MPCController, MPCParams
    from wbc.wbc import WBCController, WBCParams

    # Initialize
    robot = A2Robot(model_path)
    robot.reset()

    gait = GaitGenerator(GaitParams(gait_type=GaitType(gait_type)))
    mpc = MPCController()
    wbc = WBCController(n_joints=12)

    print(f"Starting live simulation: gait={gait_type}, speed={speed} m/s")
    print("Close viewer window to exit.")

    # Launch viewer
    from mujoco.viewer import launch
    with launch(model, data) as viewer:
        try:
            viewer.cam.type = mujoco.mjtCam.mjCAM_TRACKING
            viewer.cam.trackbodyid = model.body('base_link').id
            viewer.cam.distance = 1.5
            viewer.cam.elevation = -10.0
        except:
            pass

        dt = 0.002
        step = 0

        while viewer.is_running():
            # Get state
            state = robot.update_state()
            com_pos = state.base_pos
            com_vel = state.base_lin_vel

            # Update gait
            gait.step(dt)
            contact = gait.get_contact_schedule()

            # Compute foot desired (simple: maintain current position)
            foot_pos = robot.get_foot_positions()
            foot_des = foot_pos.copy()
            foot_vel_des = np.zeros((4, 3))

            # Swing legs: move forward with body
            for i in range(4):
                if not contact[i]:
                    foot_des[i, 0] += speed * dt  # Move foot forward

            # Compute torques
            torques = wbc.compute_joint_torques(
                com_pos=com_pos, com_vel=com_vel, com_rpy=np.zeros(3),
                com_ang_vel=state.base_ang_vel,
                foot_positions=foot_pos,
                foot_velocities=np.zeros((4, 3)),
                foot_desired_positions=foot_des,
                foot_desired_velocities=foot_vel_des,
                contact_states=contact,
                joint_positions=state.joint_pos,
                joint_velocities=state.joint_vel,
            )
            torques = wbc.clip_torques(torques, state.joint_vel)

            # Step simulation
            robot.step(torques)

            # Render
            viewer.render()

            step += 1
            if step % 500 == 0:
                print(f"  Step {step} | CoM: ({com_pos[0]:.2f}, {com_pos[1]:.2f}, {com_pos[2]:.2f})")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--live':
        # Live simulation mode
        model_path = sys.argv[2] if len(sys.argv) > 2 else "models/a2_scene.xml"
        gait = sys.argv[3] if len(sys.argv) > 3 else "trot"
        speed = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
        replay_with_control(model_path, gait_type=gait, speed=speed)
    else:
        # Replay mode
        model_path = sys.argv[1] if len(sys.argv) > 1 else "models/a2_scene.xml"
        data_path = sys.argv[2] if len(sys.argv) > 2 else "results/simulation_data.json"
        speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        replay_simulation(model_path, data_path, speed)
