"""Final walking test"""
import sys, os, numpy as np
os.chdir('/Users/blackzhou/work/robot/mpc/test/quadruped_mpc')
sys.path.insert(0, 'src')
from main import QuadrupedController

ctrl = QuadrupedController('models/a2_scene.xml')
ctrl.set_command(0.3, 0.0, 0.0)  # forward command

print("=== Walking test (vx=0.3, 800 steps) ===\n")
for i in range(800):
    tau = ctrl.update(0.002)
    ctrl.robot.step(tau)
    if i % 100 == 0:
        s = ctrl.robot.state
        rpy = ctrl.robot.get_rpy()
        contacts = ctrl.gait.contact_states
        n_swing = int((~contacts).sum())
        print(f"  Step {i:4d}: z={s.base_pos[2]:.3f} x={s.base_pos[0]:.3f} "
              f"vx={s.base_lin_vel[0]:.3f} yaw={rpy[2]*180/np.pi:+.1f}deg "
              f"swing={n_swing}")

s = ctrl.robot.state
rpy = ctrl.robot.get_rpy()
print(f"\nFinal: z={s.base_pos[2]:.3f} x={s.base_pos[0]:.3f} vx={s.base_lin_vel[0]:.3f}")
print(f"  displacement={s.base_pos[0]:.3f}m  avg_vx={s.base_pos[0]/(800*0.002):.3f}m/s")
print(f"  yaw={rpy[2]*180/np.pi:+.1f}deg")
print(f"  height_ok={s.base_pos[2] > 0.30}  walking={abs(s.base_pos[0]) > 0.3}")
