# 四足机器人MPC控制设计文档

**项目名称**: Unitree A1 四足机器人模型预测控制与全身控制  
**版本**: v0.1.0  
**适用机器人**: Unitree A1  
**仿真平台**: MuJoCo 3.x  
**编程语言**: Python 3.12

---

## 目录

1. [系统概述](#1-系统概述)
2. [整体架构](#2-整体架构)
3. [机器人模型模块](#3-机器人模型模块)
4. [步态生成器模块](#4-步态生成器模块)
5. [MPC模块](#5-mpc模块)
6. [WBC模块](#6-wbc模块)
7. [控制循环流程](#7-控制循环流程)
8. [Sim2Real鲁棒性设计](#8-sim2real鲁棒性设计)
9. [参数配置](#9-参数配置)
10. [使用说明](#10-使用说明)

---

## 1. 系统概述

### 1.1 问题背景

四足机器人的稳定行走控制是一个复杂的控制问题，需要同时考虑：
- **质心稳定性**: 维持机器人整体平衡
- **足端轨迹**: 控制四条腿的协调运动
- **接触力优化**: 在摩擦约束下合理分配地面反作用力
- **关节力矩**: 将高层的力/位置指令映射到关节力矩

本系统实现了一个**分层递阶控制系统**，将上述问题分解为：
1. **步态层**: 决定每条腿的支撑/摆动时序和足端轨迹
2. **MPC层**: 在预测时域内优化质心轨迹和足端力分布
3. **WBC层**: 将质心/足端指令转化为关节力矩

### 1.2 设计目标

| 目标 | 说明 |
|------|------|
| **模块化** | 各层独立，可单独替换和优化 |
| **稳定性** | 支持trot、walk等多种步态 |
| **实时性** | MPC在20-50ms内求解 |
| **Sim2Real** | 低gap设计，直接迁移真实机器人 |
| **可扩展性** | 支持添加新任务和约束 |

### 1.3 适用场景

- MuJoCo仿真环境下的四足机器人控制研究
- MPC/WBC算法验证与调参
- Sim2Real迁移研究
- 教学与算法演示

---

## 2. 整体架构

### 2.1 系统框图

```
┌─────────────────────────────────────────────────────────────┐
│                        主控制循环                             │
│  (50Hz MPC + 500Hz WBC, dt=2ms)                            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   Gait层      │     │   MPC层       │     │   WBC层       │
│   (步态生成)   │────▶│  (质心预测)   │────▶│  (关节控制)   │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ - 步态时序     │     │ - 重心轨迹     │     │ - 关节力矩    │
│ - 足端轨迹     │     │ - 足端力优化   │     │ - 优先级任务   │
│ - 触地检测     │     │ - 摩擦锥约束   │     │ - 力矩限制    │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌───────────────────┐
                    │   机器人仿真        │
                    │   (MuJoCo)         │
                    └───────────────────┘
```

### 2.2 控制频率与延迟

| 层级 | 控制频率 | 延迟 | 说明 |
|------|---------|------|------|
| Gait | 50Hz | 20ms | 步态时序更新 |
| MPC | 50Hz | 20-40ms | QP求解 |
| WBC | 500Hz | 2ms | 关节力矩计算 |
| 仿真 | 500Hz | 2ms | MuJoCo步进 |

### 2.3 数据流

```
状态估计 → Gait生成 → MPC优化 → WBC计算 → 关节力矩 → 仿真步进
   ↑                                              │
   └──────────── 传感器反馈 ←──────────────────────┘
```

---

## 3. 机器人模型模块

### 3.1 模块概述

`robot/robot.py` 负责机器人与MuJoCo仿真环境的交互，提供：
- 机器人状态提取与估计
- 运动学计算（正运动学/逆运动学）
- 雅可比矩阵计算
- 接触力估计
- 仿真步进控制

### 3.2 机器人参数

#### Unitree A1 物理参数

| 参数 | 数值 | 单位 |
|------|------|------|
| 总质量 | 12.0 | kg |
| 质心高度 | 0.28 | m |
| 上腿长度 | 0.213 | m |
| 下腿长度 | 0.213 | m |
| 髋关节间距 | 0.183 | m |
| 最大关节力矩 | 35.0 | Nm |
| 关节角度限制 | ±2.7~4.0 | rad |
| 摩擦系数 | 0.7 | - |

#### 关节配置

```
        FL(0)    FR(1)
          \      /
           [base]
          /      \
        RL(2)    RR(3)

每条腿3个关节: hip_joint, thigh_joint, calf_joint
总关节数: 12
```

### 3.3 核心类设计

#### RobotParams (机器人参数)

```python
@dataclass
class RobotParams:
    mass: float              # 机器人总质量 [kg]
    inertia: np.ndarray      # 惯量张量 [3,3]
    hip_positions: np.ndarray  # 髋关节相对质心位置 [4,3]
    upper_leg_length: float  # 上腿长度 [m]
    lower_leg_length: float  # 下腿长度 [m]
    joint_limits: Tuple     # 关节角度限制 [rad]
    torque_limits: Tuple    # 关节力矩限制 [Nm]
    friction_coeff: float   # 摩擦系数
```

#### RobotState (机器人状态)

```python
@dataclass
class RobotState:
    base_pos: np.ndarray       # 质心位置 [x,y,z]
    base_quat: np.ndarray      # 质心姿态四元数 [w,x,y,z]
    base_lin_vel: np.ndarray   # 质心线速度 [vx,vy,vz] (body frame)
    base_ang_vel: np.ndarray   # 质心角速度 [wx,wy,wz] (body frame)
    joint_pos: np.ndarray      # 关节角度 [q1..q12]
    joint_vel: np.ndarray      # 关节速度 [dq1..dq12]
    foot_pos: np.ndarray       # 足端位置 [4,3]
    foot_vel: np.ndarray       # 足端速度 [4,3]
    foot_force: np.ndarray     # 地面反作用力 [4,3]
    contact_state: np.ndarray  # 触地状态 [4] (bool)
```

### 3.4 运动学计算

#### 正运动学 (Forward Kinematics)

使用**解析法**计算足端位置：

```
q = [q_hip, q_thigh, q_calf]  # 单腿关节角

p_foot = p_hip + R_hip(q_hip) * [0, l_thigh*sin(q_thigh), -l_thigh*cos(q_thigh)]
                                 + [0, 0, -l_calf*cos(q_thigh + q_calf)]
```

其中 `R_hip` 是髋关节旋转矩阵。

#### 雅可比矩阵 (Jacobian)

足端速度与关节速度的关系：

```
v_foot = J(q) * q_dot
```

雅可比矩阵 `J ∈ ℝ^(3×3)` 通过MuJoCo API计算：
```python
mujoco.mj_jacBody(model, data, jacp, jacr, foot_body_id)
```

#### 逆运动学 (Inverse Kinematics)

使用**阻尼最小二乘法**：

```
q_dot = J^T * (J * J^T + λ*I)^(-1) * (p_des - p_cur)
```

其中 λ 是阻尼系数 (λ=0.01)，用于处理雅可比矩阵奇异问题。

### 3.5 状态估计

#### 坐标变换

```python
# 世界坐标系 → 机体坐标系
v_body = R_wb^T * v_world

# 机体坐标系 → 世界坐标系  
p_world = R_wb * p_body + p_com
```

#### 姿态表示

使用四元数 `q = [w, x, y, z]` 表示姿态，避免欧拉角的万向锁问题。

---

## 4. 步态生成器模块

### 4.1 模块概述

`gait/gait.py` 实现步态时序调度和足端轨迹生成：

```
┌──────────────────────────────────────────────────┐
│                  GaitGenerator                     │
├──────────────────────────────────────────────────┤
│  GaitScheduler                                    │
│  ├── 步态时序调度                                  │
│  ├── 支撑/摆动相位判断                              │
│  └── 接触状态预测                                   │
├──────────────────────────────────────────────────┤
│  FootTrajectory                                    │
│  ├── 支撑相足端位置保持                              │
│  └── 摆动相轨迹跟踪                                  │
├──────────────────────────────────────────────────┤
│  SwingTrajectoryGenerator                          │
│  ├── Bezier曲线足端轨迹                             │
│  ├── 足端高度控制                                   │
│  └── 速度平滑插值                                   │
└──────────────────────────────────────────────────┘
```

### 4.2 步态类型

| 步态 | 支撑模式 | 占空比 | 飞行相 | 适用场景 |
|------|---------|--------|--------|---------|
| **Trot** | 对角线足对 | 62.5% | 无 | 快速行走 |
| **Walk** | 单腿支撑 | 75% | 无 | 慢速/稳定 |
| **Gallop** | 不对称 | ~50% | 有 | 快速奔跑 |
| **Bound** | 前后同步 | 50% | 有 | 跳跃 |
| **Pronk** | 四腿同时 | 50% | 有 | 实验用 |

### 4.3 步态时序

#### 步态周期

```
T_step = T_stance + T_swing

占空比 = T_stance / T_step
```

#### Trot步态时序图

```
Leg:  FL    FR    RL    RR
     [=====]      [=====]     ← T_stance
            [=====]      [=====]
              ↑                    ↑
            T_step/2相移
```

### 4.4 摆动轨迹生成

#### 三次Bezier曲线

摆动相使用修改的三次Bezier曲线生成平滑轨迹：

```
轨迹点 = B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
```

其中：
- `P₀`: 起跳足端位置
- `P₁`: 起跳点控制点 (抬腿)
- `P₂`: 落地点控制点 (落腿)
- `P₃`: 目标足端位置

#### 足端高度曲线

```
h(t) = h_max * 4 * t * (1 - t)  // 抛物线型高度曲线
```

在 `t=0.5` 时达到最大高度 `h_max`。

### 4.5 Raibert启发式足端位置

**Raibert heuristic** 预测足端落点：

```
p_foot_des = p_com + ω × r_hip + v_com * T_stance
```

其中：
- `p_com`: 质心位置
- `ω × r_hip`: 角速度补偿
- `v_com * T_stance`: 速度前瞻补偿

---

## 5. MPC模块

### 5.1 模块概述

`mpc/mpc.py` 实现基于线性化 centroidal 动力学的模型预测控制：

```
┌─────────────────────────────────────────────────┐
│                  MPCController                    │
├─────────────────────────────────────────────────┤
│  CentroidalDynamics                              │
│  ├── 线性化质心动力学模型                          │
│  ├── 状态空间表示 (A, B矩阵)                      │
│  └── 状态传播                                     │
├─────────────────────────────────────────────────┤
│  FrictionCone                                    │
│  ├── 摩擦锥约束 (线性化金字塔近似)                  │
│  ├── 法向力上下限                                  │
│  └── 切向力限制                                    │
├─────────────────────────────────────────────────┤
│  QP求解器 (CVXPY/CVXOPT)                         │
│  ├── 重启动热启动                                   │
│  └── 预测时域优化                                   │
└─────────────────────────────────────────────────┘
```

### 5.2 质心动力学

#### 连续时间模型

**线动量**:
```
m * v̇_com = Σf_i + m * g
```

**角动量**:
```
I * ω̇_com = Σ(p_i - p_com) × f_i
```

其中：
- `m`: 机器人总质量
- `v_com`: 质心速度
- `f_i`: 第i条腿的地面反作用力
- `p_i`: 第i条腿足端位置
- `I`: 质心惯量张量
- `g`: 重力加速度

#### 离散化状态空间

状态向量:
```
x = [p_com, v_com, φ, θ, ψ, ω_com] ∈ ℝ^12
```

其中 `φ, θ, ψ` 是RPY欧拉角。

系统矩阵:
```
x_{k+1} = A * x_k + B * u_k
```

其中输入 `u` 是质心加速度和角加速度。

**A矩阵** (离散化):
```
A = I + dt * A_cont

A_cont = [ 0    I    0    0
           0    0    0    0
           0    0    0   -I
           0    0    0    0 ]
```

**B矩阵**:
```
B = dt * [ I/m     0
          0    I⁻¹ ]
```

### 5.3 QP问题 formulation

#### 优化变量

```
X = [x_0, x_1, ..., x_H]     # 状态轨迹
F = [f_0, f_1, ..., f_H]     # 接触力轨迹
```

#### 目标函数

```
min Σ_{k=0}^{H} ||x_k - x_ref||²_Q + ||f_k||²_R
    + ||f_k - f_{k-1}||²_R_rate
```

权重矩阵：
- `Q`: 状态跟踪权重（位置>速度>姿态）
- `R`: 力控制权重（正则化）
- `R_rate`: 力变化率权重（平滑性）

#### 约束条件

**动力学约束**:
```
x_{k+1} = A * x_k + B * C_k * f_k
```

其中 `C_k` 是接触力选择矩阵（仅支撑腿有力）。

**摩擦锥约束** (线性化金字塔近似):
```
|f_x| ≤ μ * f_z
|f_y| ≤ μ * f_z
f_z ≥ f_min
f_z ≤ f_max
```

**质心高度约束**:
```
h_min ≤ p_com,z ≤ h_max
```

### 5.4 求解策略

#### 热启动 (Warm Start)

使用上一时刻的求解结果初始化当前QP：
```python
if self._last_solution is not None:
    prob.solve(warm_start=True)
```

这显著加速了QP求解（通常减少30-50%求解时间）。

#### QP求解器选择

| 求解器 | 优点 | 缺点 | 适用场景 |
|--------|------|------|---------|
| OSQP | 快速、鲁棒 | 精度一般 | 实时控制 |
| CVXOPT | 精确 | 较慢 | 离线优化 |
| ECOS | 平衡 | 中等 | 通用 |
| SCS | 并行 | 精度低 | 大规模问题 |

#### 备用方案

当QP求解失败时，退回到均匀力分配：
```python
f_per_leg = (m * g) / n_stance
```

---

## 6. WBC模块

### 6.1 模块概述

`wbc/wbc.py` 实现基于**操作空间控制 (OSC)** 的分层全身控制：

```
┌─────────────────────────────────────────────────┐
│               WBCController                      │
├─────────────────────────────────────────────────┤
│  Task Priority Level 1: CoM Position Control    │
│  └── 维持质心高度和水平位置                        │
├─────────────────────────────────────────────────┤
│  Task Priority Level 2: CoM Orientation Control │
│  └── 维持机体姿态水平                              │
├─────────────────────────────────────────────────┤
│  Task Priority Level 3: Swing Leg Tracking      │
│  └── 摆动腿足端轨迹跟踪                           │
├─────────────────────────────────────────────────┤
│  Task Priority Level 4: Stance Leg Contact      │
│  └── 支撑腿接触力维持                             │
├─────────────────────────────────────────────────┤
│  Task Priority Level 5: Joint Regularization    │
│  └── 关节位置/速度正则化                          │
└─────────────────────────────────────────────────┘
```

### 6.2 操作空间控制 (OSC)

#### 基本公式

关节力矩由任务空间误差和动力学补偿组成：

```
τ = J^T * Λ * (ä_des - ȧ_dot) + J^T * f + C + G
```

其中：
- `τ`: 关节力矩
- `J`: 任务雅可比矩阵
- `Λ`: 任务惯性矩阵 = `(J * M⁻¹ * J^T)⁻¹`
- `ä_des`: 期望任务加速度
- `ȧ_dot`: 任务加速度（雅可比导数×速度）
- `f`: 接触力
- `C`: 科里奥利力
- `G`: 重力力矩

#### PD控制律

任务加速度由PD控制律产生：

```
ä_des = Kp * (x_des - x) + Kd * (ẋ_des - ẋ)
```

### 6.3 任务优先级

#### 层次化QP求解

高优先级任务先求解，低优先级任务在高优先级任务的**零空间**中求解：

```
N_k = I - J_k^+ * J_k    # 零空间投影矩阵

τ_total = τ_1 + N_1 * τ_2 + N_1*N_2 * τ_3 + ...
```

#### 优先级权重

| 任务 | 权重 | 控制带宽 |
|------|------|---------|
| CoM位置 | 1000 | 10Hz |
| CoM姿态 | 800 | 10Hz |
| 足端位置 | 500 | 20Hz |
| 关节正则 | 0.1 | - |

### 6.4 接触约束

#### 支撑腿约束

支撑腿的足端速度必须为零：
```
J_c * q̇ = 0
```

这等价于要求支撑腿足端保持静止。

#### 力分布

支撑腿的接触力由MPC提供，WBC负责实现：
```python
tau += J^T * f_contact
```

### 6.5 力矩限制处理

#### 硬限制

```python
tau_clipped = np.clip(tau, tau_min, tau_max)
```

#### 速度相关限制

当关节速度接近极限时，限制力矩方向：
```python
if qdot > 0.95 * qdot_max:
    tau = min(tau, 0)  # 只能减速
elif qdot < 0.95 * qdot_min:
    tau = max(tau, 0)  # 只能加速
```

---

## 7. 控制循环流程

### 7.1 主循环伪代码

```
Initialize:
    robot = A1Robot(model_path)
    gait = GaitGenerator()
    mpc = MPCController()
    wbc = WBCController()

Control Loop (dt = 2ms):
    ┌─────────────────────────────────────────┐
    │ 1. 状态估计                              │
    │    state = robot.update_state()         │
    │    com_pos, com_vel = extract_com()     │
    │    foot_pos, contact = extract_feet()   │
    └─────────────────┬───────────────────────┘
                      ▼
    ┌─────────────────────────────────────────┐
    │ 2. 步态更新 (每20ms)                     │
    │    gait.update(dt)                      │
    │    foot_des = gait.compute_desired()    │
    └─────────────────┬───────────────────────┘
                      ▼
    ┌─────────────────────────────────────────┐
    │ 3. MPC求解 (每20ms)                      │
    │    mpc.set_reference(com_des)           │
    │    forces = mpc.solve(state, feet)      │
    └─────────────────┬───────────────────────┘
                      ▼
    ┌─────────────────────────────────────────┐
    │ 4. WBC求解 (每2ms)                       │
    │    tau = wbc.compute_torques(...)       │
    │    tau = clip(tau, limits)              │
    └─────────────────┬───────────────────────┘
                      ▼
    ┌─────────────────────────────────────────┐
    │ 5. 执行与仿真                            │
    │    robot.set_joint_torque(tau)          │
    │    mujoco.mj_step(model, data)          │
    └─────────────────────────────────────────┘
```

### 7.2 时间同步

```python
# MPC层：20ms更新一次
mpc_dt = 0.02
mpc_counter = 0

# WBC层：2ms更新一次
wbc_dt = 0.002

# 仿真步进
sim_dt = 0.002
mujoco.mj_step(model, data)
```

---

## 8. Sim2Real鲁棒性设计

### 8.1 关键设计原则

#### 8.1.1 低仿真复杂度

| 策略 | 说明 |
|------|------|
| **线性化模型** | MPC使用线性化模型，不依赖精确非线性 |
| **简化雅可比** | WBC使用解析雅可比，避免数值微分噪声 |
| **无模型依赖** | WBC包含动力学补偿，不依赖精确模型 |

#### 8.1.2 鲁棒性增强

**力平滑**:
```python
force_rate_weight = 0.1  # 惩罚力变化率
```

**状态噪声注入**:
```python
state_noise = np.random.normal(0, 0.01, size=12)
state_robust = state + state_noise
```

**边界层软化**:
```python
# 摩擦锥使用线性化近似而非精确圆锥
# 关节力矩限制留有余量
tau_limit_real = 0.8 * tau_limit_sim
```

#### 8.1.3 现实世界适配

**执行器延迟补偿**:
```python
# 使用低通滤波模拟执行器响应
tau_filtered = low_pass_filter(tau_des, tau_prev, cutoff=50Hz, dt=0.002)
```

**接触检测延迟**:
```python
# 使用力传感器阈值而非位置判断
contact = foot_force_z > 5.0  # N
```

### 8.2 Sim2Real迁移检查清单

- [ ] MPC预测时域与真实延迟匹配
- [ ] WBC控制频率与真实控制器一致
- [ ] 关节力矩限制留20%余量
- [ ] 摩擦系数略低于仿真值
- [ ] 状态估计延迟建模
- [ ] 执行器带宽建模

---

## 9. 参数配置

### 9.1 配置文件

所有参数存储在 `config/default.yaml`：

```yaml
robot:
  mass: 12.0           # 机器人质量 [kg]
  inertia: [0.014, 0.028, 0.039]  # 惯量张量
  friction_coeff: 0.7  # 摩擦系数

gait:
  type: "trot"         # 步态类型
  step_period: 0.4     # 步态周期 [s]
  stance_duration: 0.25  # 支撑相时长 [s]
  foot_height: 0.08    # 抬腿高度 [m]

mpc:
  horizon_steps: 10    # 预测时域步数
  dt: 0.025            # MPC时间步长 [s]
  com_pos_weight: 100  # 质心位置权重
  friction_coeff: 0.7  # 摩擦系数
  min_normal_force: 20 # 最小法向力 [N]
  max_normal_force: 300 # 最大法向力 [N]

wbc:
  control_freq: 500    # 控制频率 [Hz]
  com_pos_kp: 200      # CoM位置KP
  com_pos_kd: 20       # CoM位置KD
  foot_kp: 800         # 足端位置KP
  foot_kd: 40          # 足端位置KD
```

### 9.2 参数调优指南

#### MPC参数调优

| 参数 | 增大效果 | 减小效果 | 推荐范围 |
|------|---------|---------|---------|
| `com_pos_weight` | 更精确的CoM跟踪 | 更柔和的控制 | 50-500 |
| `force_weight` | 更小的力 | 更平滑的力 | 0.001-0.1 |
| `force_rate_weight` | 更平滑的力变化 | 更快的力响应 | 0.01-1.0 |
| `horizon_steps` | 更远的预测 | 更快的求解 | 5-20 |

#### WBC参数调优

| 参数 | 增大效果 | 推荐值 |
|------|---------|--------|
| `com_pos_kp` | 更快的CoM响应 | 100-300 |
| `foot_kp` | 更硬的足端跟踪 | 500-1500 |
| `com_ori_weight` | 更平的姿态 | 500-1000 |

---

## 10. 使用说明

### 10.1 快速开始

```bash
# 1. 创建环境
conda env create -f environment.yml
conda activate quadruped_mpc

# 2. 运行仿真
python src/main.py --gait trot --duration 10 --command 0.3 0.0 0.0

# 3. 查看结果
ls results/
```

### 10.2 命令参数

```bash
python src/main.py \
    --gait trot          # 步态类型: trot/walk/gallop/bound
    --duration 10.0      # 仿真时长 [s]
    --command 0.3 0.0 0.0  # 速度指令 [vx, vy, yaw_rate]
    --headless           # 无界面模式
```

### 10.3 代码示例

```python
from src.robot import A1Robot
from src.gait import GaitGenerator, GaitParams
from src.mpc import MPCController, MPCParams
from src.wbc import WBCController, WBCParams

# 初始化
robot = A1Robot("models/unitree_a1.xml")
gait = GaitGenerator(GaitParams(gait_type="trot"))
mpc = MPCController(MPCParams())
wbc = WBCController(WBCParams())

# 控制循环
for _ in range(5000):
    # 状态估计
    state = robot.update_state()
    
    # 步态更新
    gait.update(dt=0.002)
    
    # MPC求解
    forces = mpc.solve(centroidal_state, foot_pos, contact)
    
    # WBC求解
    torques = wbc.compute_torques(state, gait_desired, contact)
    
    # 执行
    robot.set_joint_torque(torques)
    mujoco.mj_step(model, data)
```

---

## 附录

### A. 符号说明

| 符号 | 含义 |
|------|------|
| `p_com` | 质心位置 |
| `v_com` | 质心速度 |
| `ω_com` | 质心角速度 |
| `f_i` | 第i条腿的地面反作用力 |
| `μ` | 摩擦系数 |
| `J` | 雅可比矩阵 |
| `Λ` | 任务惯性矩阵 |
| `τ` | 关节力矩 |
| `q` | 关节位置 |
| `q̇` | 关节速度 |

### B. 参考论文

1. **MPC for Quadrupedal Locomotion** - Kim et al., ICRA 2019
2. **Whole-Body MPC** - Herzog et al., Humanoids 2015
3. **MIT Cheetah MPC** - Bledt et al., IROS 2018
4. **Raibert Hopping** - Raibert, 1986

### C. 常见问题

**Q: MPC求解失败怎么办？**  
A: 检查初始状态是否合理，增大 `regularization`，或使用备用力分布。

**Q: 机器人容易摔倒？**  
A: 增大 `com_pos_weight`，减小 `dt`，检查摩擦系数设置。

**Q: 如何提高仿真速度？**  
A: 使用 `--headless` 模式，减小 `horizon_steps`。

---

*文档版本: 0.1.0 | 最后更新: 2025-07-11*
