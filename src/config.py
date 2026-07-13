"""
Configuration Module.

YAML-based configuration for all controller parameters.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(path: str = "config/default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# Default configuration
DEFAULT_CONFIG = {
    'robot': {
        'mass': 12.0,
        'inertia': [0.014, 0.028, 0.039],
        'friction_coeff': 0.7,
    },
    'gait': {
        'type': 'trot',
        'step_period': 0.4,
        'stance_duration': 0.25,
        'swing_duration': 0.15,
        'foot_height': 0.08,
    },
    'mpc': {
        'horizon_steps': 10,
        'dt': 0.025,
        'com_pos_weight': 100.0,
        'com_vel_weight': 50.0,
        'force_weight': 0.01,
        'force_rate_weight': 0.1,
        'friction_coeff': 0.7,
        'min_normal_force': 20.0,
        'max_normal_force': 300.0,
    },
    'wbc': {
        'control_freq': 500.0,
        'com_pos_kp': 200.0,
        'com_pos_kd': 20.0,
        'foot_kp': 800.0,
        'foot_kd': 40.0,
    },
    'simulation': {
        'timestep': 0.002,
        'duration': 10.0,
    }
}
