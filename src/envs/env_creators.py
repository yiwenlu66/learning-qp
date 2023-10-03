import numpy as np
from .linear_system import LinearSystem
from .cartpole import CartPole

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
    },
    "cartpole": {
        "n": 4,
        "m": 1,
        "m_cart": [0.7, 1.3],
        "m_pole": [0.07, 0.13],
        "l": [0.4, 0.7],
        "m_cart_nom": 1.0,
        "m_pole_nom": 0.1,
        "l_nom": 0.55,
        "Q": np.diag([1., 1e-4, 1., 1e-4]),
        "R": np.array([[1e-4]]),
        "x_min": -2,
        "x_max": 2,
        "u_min": -10,
        "u_max": 10,
        "dt": 0.1,
    },
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
        barrier_thresh=1.,
        randomize_std=(0.001 if kwargs["randomize"] else 0.),
        **kwargs
    ),
    "cartpole": lambda **kwargs: CartPole(
        parameters={
            "m_cart": [sys_param["cartpole"]["m_cart_nom"], sys_param["cartpole"]["m_cart_nom"]] if not kwargs["randomize"] else sys_param["cartpole"]["m_cart"],
            "m_pole": [sys_param["cartpole"]["m_pole_nom"], sys_param["cartpole"]["m_pole_nom"]] if not kwargs["randomize"] else sys_param["cartpole"]["m_pole"],
            "l": [sys_param["cartpole"]["l_nom"], sys_param["cartpole"]["l_nom"]] if not kwargs["randomize"] else sys_param["cartpole"]["l"],
            "dt": sys_param["cartpole"]["dt"],
        },
        Q=sys_param["cartpole"]["Q"],
        R=sys_param["cartpole"]["R"],
        noise_std=kwargs["noise_level"],
        x_min=sys_param["cartpole"]["x_min"],
        x_max=sys_param["cartpole"]["x_max"],
        u_min=sys_param["cartpole"]["u_min"],
        u_max=sys_param["cartpole"]["u_max"],
        bs=kwargs["bs"],
        barrier_thresh=0.1,
        max_steps=kwargs["max_steps"],
        keep_stats=kwargs["keep_stats"],
        run_name=kwargs["run_name"],
        exp_name=kwargs["exp_name"],
    ),
}