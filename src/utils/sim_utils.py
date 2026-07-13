from __future__ import annotations

"""
Simulation Utilities Module.

Provides utilities for MuJoCo simulation setup, visualization,
and data logging.
"""

import numpy as np
import mujoco
from typing import Optional, Tuple, List
from pathlib import Path


class SimConfig:
    """Simulation configuration."""

    def __init__(
        self,
        timestep: float = 0.002,
        n_substeps: int = 1,
        solver_iterations: int = 50,
        enable_perturbation: bool = False,
        perturb_force_range: Tuple[float, float] = (-50, 50),
    ):
        self.timestep = timestep
        self.n_substeps = n_substeps
        self.solver_iterations = solver_iterations
        self.enable_perturbation = enable_perturbation
        self.perturb_force_range = perturb_force_range


def setup_mujoco_simulation(model: mujoco.MjModel, config: SimConfig) -> mujoco.MjData:
    """
    Configure MuJoCo simulation parameters.

    Args:
        model: MuJoCo model
        config: Simulation configuration

    Returns:
        MuJoCo data object
    """
    # Set timestep
    model.opt.timestep = config.timestep

    # Solver settings for accuracy
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = config.solver_iterations
    model.opt.tolerance = 1e-6

    # Enable accurate contact
    model.opt.contype = 1
    model.opt.conaffinity = 1

    # Integration
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    return data


def add_viewer_controls(
    viewer: object,
    config: SimConfig,
) -> None:
    """Add perturbation controls to viewer."""
    if config.enable_perturbation:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True


def compute_com_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    com_desired: np.ndarray,
    rpy_desired: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute CoM tracking metrics.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        com_desired: Desired CoM position [3]
        rpy_desired: Desired RPY orientation [3]

    Returns:
        Tuple of (position_error, orientation_error)
    """
    com_pos = data.xpos[model.body('base').id].copy()

    # Position error
    pos_error = np.linalg.norm(com_pos[:2] - com_desired[:2])

    # Orientation error
    quat_actual = data.xquat[model.body('base').id].copy()
    from utils.math_utils import quaternion_to_rpy
    rpy_actual = quaternion_to_rpy(quat_actual)
    ori_error = np.linalg.norm(rpy_actual - rpy_desired)

    return pos_error, ori_error


class DataLogger:
    """Log simulation data for analysis."""

    def __init__(self):
        self.timestamps: List[float] = []
        self.data: dict = {}
        self._buffer: dict = {}  # 最新数据缓存（用于调试）

    def log(self, timestamp: float, **kwargs) -> None:
        """Log data at timestamp."""
        self.timestamps.append(timestamp)
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(np.asarray(value).copy())
            # 更新最新缓存
            self._buffer[key] = np.asarray(value).copy()

    def get_array(self, key: str) -> np.ndarray:
        """Get logged data as array."""
        return np.array(self.data[key])

    def save(self, path: str) -> None:
        """Save logged data to file."""
        import json

        output = {
            'timestamps': self.timestamps,
            'data': {k: [v.tolist() if isinstance(v, np.ndarray) else v
                         for v in vals] for k, vals in self.data.items()}
        }
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)

    def load(self, path: str) -> None:
        """Load logged data from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.timestamps = data['timestamps']
        self.data = {k: [np.array(v) for v in vals] for k, vals in data['data'].items()}


def save_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
) -> None:
    """Save list of frames as video."""
    import imageio
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def render_to_array(viewer) -> np.ndarray:
    """Render viewer frame to numpy array."""
    return np.array(viewer.read_pixels(
        viewer.viewport.width,
        viewer.viewport.height,
        depth=False
    ))
