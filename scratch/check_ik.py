import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('models/a2_scene.xml')
data = mujoco.MjData(model)

# Set base to the CoM pos and RPY from Step 500:
# CoM pos: (0.1589, 0.0238, 0.4684)
# RPY: roll=+5.12°, pitch=+9.05°, yaw=+0.82°
data.qpos[0:3] = [0.1589, 0.0238, 0.4684]
# Base orientation quaternion (RPY to Quat)
roll, pitch, yaw = np.radians([5.12, 9.05, 0.82])
cr, sr = np.cos(roll/2), np.sin(roll/2)
cp, sp = np.cos(pitch/2), np.sin(pitch/2)
cy, sy = np.cos(yaw/2), np.sin(yaw/2)
qw = cr*cp*cy + sr*sp*sy
qx = sr*cp*cy - cr*sp*sy
qy = cr*sp*cy + sr*cp*sy
qz = cr*cp*sy - sr*sp*cy
data.qpos[3:7] = [qw, qx, qy, qz]

# Set RL leg joint angles to q_des and q_cur
# Leg RL is leg 2 (joints indices: 6, 7, 8 in qpos[7:19])
q_des = [-0.11898368,  0.67512802, -1.40163474]
q_cur = [-0.06981695,  0.78527649, -1.45243489]

def get_foot_pos_for_joints(joints):
    data.qpos[7:19] = 0.0  # reset other legs
    data.qpos[7+6 : 7+9] = joints  # set RL
    mujoco.mj_forward(model, data)
    
    body_id = model.body('RL_calf').id
    calf_pos = data.xpos[body_id].copy()
    R_calf = data.xmat[body_id].reshape(3, 3)
    foot_offset_calf = np.array([0.0, 0.0, -0.275])
    foot_pos = calf_pos + R_calf @ foot_offset_calf
    return foot_pos

pos_from_q_des = get_foot_pos_for_joints(q_des)
pos_from_q_cur = get_foot_pos_for_joints(q_cur)

print("RL Foot Pos from q_des:")
print("  FK pos  :", pos_from_q_des)
print("  Target  :", [-0.1555632, 0.20996926, 0.08649081])

print("\nRL Foot Pos from q_cur:")
print("  FK pos  :", pos_from_q_cur)
print("  Actual  :", [-0.18719147, 0.22863123, 0.11584124])
