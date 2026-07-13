# 四足机器人MPC控制项目

基于MuJoCo仿真的宇树A1四足机器人模型预测控制（MPC）与全身控制（WBC）实现。

## 项目特性

- 🦾 **宇树A1机器人模型**: 完整的URDF/MJCF模型和运动学/动力学接口
- 🚶 **多种步态模式**: 步态生成器支持trot、walk、gallop等多种步态
- 🧮 **MPC轨迹优化**: 线性MPC实现重心轨迹规划与足端力分配
- 🤖 **WBC全身控制**: 任务空间优先级控制，支持位置/速度/力矩混合控制
- 🔄 **Sim2Real迁移**: 低sim2real gap设计，支持直接部署到真实机器人

## 目录结构

```
quadruped_mpc/
├── config/          # YAML配置文件
├── src/
│   ├── robot/       # 机器人模型与运动学
│   ├── gait/        # 步态生成
│   ├── mpc/         # 模型预测控制
│   ├── wbc/         # 全身控制
│   └── utils/       # 工具函数
├── models/          # 机器人URDF/MJCF模型
├── tests/           # 测试脚本
└── docs/            # 详细设计文档
```

## 快速开始

### 环境配置

```bash
conda env create -f environment.yml
conda activate quadruped_mpc
```

### 运行仿真

```bash
cd /Users/blackzhou/work/robot/mpc/test/quadruped_mpc
python src/main.py
```

### 单元测试

```bash
pytest tests/
```

## 核心模块说明

| 模块 | 功能 | 关键文件 |
|------|------|----------|
| Robot | 机器人状态、运动学、动力学 | `src/robot/` |
| Gait | 足端轨迹、步态时序 | `src/gait/` |
| MPC | 重心轨迹预测、足端力规划 | `src/mpc/` |
| WBC | 任务优先级控制、关节力矩计算 | `src/wbc/` |
| Controller | 主控制循环 | `src/main.py` |

## 依赖

- Python >= 3.12
- mujoco >= 3.0
- numpy >= 1.24
- scipy >= 1.11
- PyYAML >= 6.0
- matplotlib >= 3.7

## 许可证

MIT License
# quadruped_mpc
