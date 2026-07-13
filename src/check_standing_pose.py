#!/usr/bin/env python3
"""
分析 A2 机器人实际站立姿态。
读取 a2_scene.xml 中的真实连杆长度和关节偏移，
使用 MuJoCo mj_jacBody 计算精确的正运动学。
"""
import sys
import os
import numpy as np

sys.path.insert(0, 'src')
import mujoco

# ── 加载模型 ─────────────────────────────────────────────────────
model_path = 'models/a2_scene.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# ── 提取关节和 body ID ───────────────────────────────────────────
joint_names = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
]
joint_qposadr = {n: model.joint(n).qposadr for n in joint_names}
joint_body    = {n: model.joint(n).bodyid for n in joint_names}

base_body = model.body('base_link').id
foot_bodies = {
    'FL': model.body('FL_calf').id,
    'FR': model.body('FR_calf').id,
    'RL': model.body('RL_calf').id,
    'RR': model.body('RR_calf').id,
}
hip_bodies = {
    'FL': model.body('FL_hip').id,
    'FR': model.body('FR_hip').id,
    'RL': model.body('RL_hip').id,
    'RR': model.body('RR_hip').id,
}

print("=" * 60)
print("A2 腿连杆几何分析 (来自 MuJoCo FK)")
print("=" * 60)

# 在零位 (q=0) 下测量各段长度
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print("\n[零位几何] (所有关节角度 = 0)")
for leg, foot_bid in foot_bodies.items():
    hip_bid = hip_bodies[leg]
    hip_pos = data.xpos[hip_bid].copy()
    foot_pos = data.xpos[foot_bid].copy()
    vec = foot_pos - hip_pos
    length = np.linalg.norm(vec)
    print(f"  {leg}: hip→foot = {vec}  |  长度 = {length:.4f} m")

# ── 网格搜索寻找站立姿态 ───────────────────────────────────────
print("\n[网格搜索] 寻找 base_z≈0.50 且 feet_z≈0 的站立姿态")

def set_pose(data, q_thigh, q_calf, q_hip=0.0):
    """设置对称站姿（4条腿相同）"""
    for leg in ['FL', 'FR', 'RL', 'RR']:
        data.qpos[joint_qposadr[f'{leg}_hip_joint']] = q_hip
        data.qpos[joint_qposadr[f'{leg}_thigh_joint']] = q_thigh
        data.qpos[joint_qposadr[f'{leg}_calf_joint']] = q_calf

def score_pose(data, target_base_z=0.50, target_foot_z=0.0):
    """评价姿态：base_z误差 + 脚z误差"""
    base_z = data.xpos[base_body][2]
    foot_err = sum((data.xpos[bid][2] - target_foot_z)**2 for bid in foot_bodies.values())
    base_err = abs(base_z - target_base_z)
    return base_err + 0.1 * foot_err, base_z, {leg: data.xpos[bid][2] for leg, bid in foot_bodies.items()}

best_score = 1e9
best_info = None

# 搜索范围：thigh=[-2.5,-0.1], calf=[-2.77,-0.54]
for qt in np.arange(-2.5, -0.1, 0.1):
    for qc in np.arange(-2.77, -0.54, 0.1):
        set_pose(data, qt, qc)
        mujoco.mj_forward(model, data)
        score, bz, fz = score_pose(data)
        if score < best_score:
            best_score = score
            best_info = (qt, qc, bz, fz)

if best_info:
    qt, qc, bz, fz = best_info
    print(f"\n最佳姿态: q_thigh={qt:.2f}, q_calf={qc:.2f}")
    print(f"  base_z = {bz:.4f} m")
    print(f"  脚部z坐标:")
    for leg, z in fz.items():
        print(f"    {leg}: {z:.4f} m")
    print(f"  得分 (越低越好): {best_score:.6f}")

    # 也输出关节角（度）
    print(f"\n  关节角 (度):")
    print(f"    hip   = 0.0°  (范围 ±{np.rad2deg(1.01):.1f}°)")
    print(f"    thigh = {np.rad2deg(qt):.1f}°  (范围 [{np.rad2deg(-2.34):.1f}°, {np.rad2deg(3.15):.1f}°])")
    print(f"    calf  = {np.rad2deg(qc):.1f}°  (范围 [{np.rad2deg(-2.77):.1f}°, {np.rad2deg(-0.54):.1f}°])")
else:
    print("未找到合适姿态")
