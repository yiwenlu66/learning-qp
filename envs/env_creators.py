import numpy as np
from .linear_system import LinearSystem

sys_param = {
    "tank": {
        "n": 4,
        "m": 2,
        "A": np.array([
            [0.984,  0.0,      0.0422029,  0.0],
            [0.0,    0.98895,  0.0,        0.0326014],
            [0.0,    0.0,      0.957453,   0.0],
            [0.0,    0.0,      0.0,        0.967216],
        ]),
        "B": np.array([
            [0.825822,    0.0101995],
            [0.00512673,  0.624648], 
            [0.0,         0.468317],
            [0.307042,    0.0],
        ]),
        "Q": np.eye(4),
        "R": 0.1 * np.eye(2),
        "x_min": 0,
        "x_max": 20,
        "u_min": -1,
        "u_max": 8,
        "u_eq_min": 0.,
        "u_eq_max": 0.3,
    }
}

env_creators = {
    "tank": lambda **kwargs: LinearSystem(
        A=sys_param["tank"]["A"],
        B=sys_param["tank"]["B"],
        Q=sys_param["tank"]["Q"],
        R=sys_param["tank"]["R"],
        sqrt_W=kwargs["noise_level"] * np.eye(4),
        x_min=sys_param["tank"]["x_min"] * np.ones(4),
        x_max=sys_param["tank"]["x_max"] * np.ones(4),
        u_min=sys_param["tank"]["u_min"] * np.ones(2),
        u_max=sys_param["tank"]["u_max"] * np.ones(2),
        u_eq_min=sys_param["tank"]["u_eq_min"] * np.ones(2),
        u_eq_max=sys_param["tank"]["u_eq_max"] * np.ones(2),
        bs=kwargs["bs"],
        barrier_thresh=1.,
        max_steps=kwargs["max_steps"],
        keep_stats=kwargs["keep_stats"],
        run_name=kwargs["run_name"],
        exp_name=kwargs["exp_name"],
    ),
}

