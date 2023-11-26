import numpy as np
import torch
from .linear_system import LinearSystem
from .cartpole import CartPole

sys_param = {
    "double_integrator": {
        "n": 2,
        "m": 1,
        "A": np.array([
            [1.0, 1.0],
            [0.0, 1.0],
        ]),
        "B": np.array([
            [0.0],
            [1.0],
        ]),
        "Q": np.eye(2),
        "R": np.array([[100.0]]),
        "x_min": -5.,
        "x_max": 5.,
        "u_min": -0.5,
        "u_max": 0.5,
    },
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
        "u_min": 0,
        "u_max": 8,
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

def tank_initial_generator(size, device, rng):
    """
    Generate initial states for the tank environment.
    State components are sampled in [0, 16] to ensure that the initial state stays within the maximal contraint invariant set.
    """
    x0 = 16. * torch.rand((size, 4), generator=rng, device=device)
    return x0

def tank_ref_generator(size, device, rng):
    """
    Generate reference states for the tank environment.
    Sampled across the entire state space.
    """
    x_ref = 20. * torch.rand((size, 4), generator=rng, device=device)
    return x_ref

def tank_randomizer(size, device, rng):
    """
    Generate \Delta A, \Delta B for the tank environment.
    """
    Delta_A11 = 0.002 * (2. * torch.rand((size,), generator=rng, device=device) - 1.)   # Leakage of tank 1
    Delta_A22 = 0.002 * (2. * torch.rand((size,), generator=rng, device=device) - 1.)   # Leakage of tank 2
    Delta_A13 = 0.002 * (2. * torch.rand((size,), generator=rng, device=device) - 1.)   # Leakage from tank 3 to tank 1
    Delta_A33 = -Delta_A13  # Conservation of tank 3
    Delta_A24 = 0.002 * (2. * torch.rand((size,), generator=rng, device=device) - 1.)   # Leakage from tank 4 to tank 2
    Delta_A44 = -Delta_A24  # Conservation of tank 4
    zeros = torch.zeros((size,), device=device)   # Other elements are not perturbed
    # A = [A11 0 A13 0; 0 A22 0 A24; 0 0 A33 0; 0 0 0 A44]
    Delta_A = torch.stack([
        torch.stack([Delta_A11, zeros, Delta_A13, zeros], dim=1),
        torch.stack([zeros, Delta_A22, zeros, Delta_A24], dim=1),
        torch.stack([zeros, zeros, Delta_A33, zeros], dim=1),
        torch.stack([zeros, zeros, zeros, Delta_A44], dim=1)
    ], dim=1)

    multiplier_B1 = 0.02 * (2. * torch.rand((size,), generator=rng, device=device) - 1.)   # Voltage perturbation on pump 1
    multiplier_B2 = 0.02 * (2. * torch.rand((size,), generator=rng, device=device) - 1.)   # Voltage perturbation on pump 2
    B = torch.tensor(sys_param["tank"]["B"], device=device, dtype=torch.float).unsqueeze(0)
    Delta_B1 = multiplier_B1.unsqueeze(-1) * B[:, :, 0]
    Delta_B2 = multiplier_B2.unsqueeze(-1) * B[:, :, 1]
    Delta_B = torch.stack([Delta_B1, Delta_B2], dim=2)

    return Delta_A, Delta_B


env_creators = {
    "double_integrator": lambda **kwargs: LinearSystem(
        A=sys_param["double_integrator"]["A"],
        B=sys_param["double_integrator"]["B"],
        Q=sys_param["double_integrator"]["Q"],
        R=sys_param["double_integrator"]["R"],
        sqrt_W=kwargs["noise_level"] * np.eye(2),
        x_min=sys_param["double_integrator"]["x_min"] * np.ones(2),
        x_max=sys_param["double_integrator"]["x_max"] * np.ones(2),
        u_min=sys_param["double_integrator"]["u_min"] * np.ones(1),
        u_max=sys_param["double_integrator"]["u_max"] * np.ones(1),
        barrier_thresh=0.1,
        randomize_std=(0.001 if kwargs["randomize"] else 0.),
        stabilization_only=True,
        **kwargs
    ),
    "tank": lambda **kwargs: LinearSystem(
        A=sys_param["tank"]["A"],
        B=sys_param["tank"]["B"],
        Q=sys_param["tank"]["Q"],
        R=sys_param["tank"]["R"],
        sqrt_W=kwargs["noise_level"] * np.eye(4),
        x_min=sys_param["tank"]["x_min"] * np.ones(4),
        x_max=sys_param["tank"]["x_max"] * np.ones(4),
        u_min=sys_param["tank"]["u_min"] * np.ones(2),
        u_max=sys_param["tank"]["u_max"] * np.ones(2) if not kwargs.get("skip_to_steady_state", False) else 1.0 * np.ones(2),
        barrier_thresh=1.,
        randomizer=(tank_randomizer if kwargs["randomize"] else None),
        reward_shaping_parameters={
            "steady_c1": kwargs["reward_shaping"][0],
            "steady_c2": kwargs["reward_shaping"][1],
            "steady_c3": kwargs["reward_shaping"][2],
        } if "reward_shaping" in kwargs else {},
        initial_generator=tank_initial_generator,
        ref_generator=tank_ref_generator,
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
