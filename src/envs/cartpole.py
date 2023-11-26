import numpy as np
import pandas as pd
import torch
import gym
import os
import random
from datetime import datetime
from icecream import ic
from ..utils.torch_utils import conditional_fork_rng, bsolve, bqf, get_rng


class CartPole():
    def __init__(self, parameters, Q, R, noise_std, x_min, x_max, u_min, u_max, bs, barrier_thresh, max_steps, device="cuda:0", random_seed=None, quiet=False, keep_stats=False,
    reward_shaping_parameters={},
    run_name="",
    **kwargs):
        """
        Class to model the Cart-Pole environment for control theory and reinforcement learning experiments.

        Parameters
        ----------
        parameters : dict
            Dictionary containing environment parameters like time-step (dt), mass of pole and car, length of pole.
        Q : ndarray
            State cost matrix.
        R : ndarray
            Control input cost matrix. (Note: this is 1x1 matrix)
        noise_std : float
            Standard deviation of noise for the environment dynamics.
        x_min, x_max, u_min, u_max : float
            Bounds for position and control input.
        bs : int
            Batch size for parallel environments.
        barrier_thresh : float
            Threshold for barrier functions.
        max_steps : int
            Maximum number of steps for an episode.
        device : str, optional
            Device for PyTorch tensors.
        random_seed : int, optional
            Random seed for reproducibility.
        quiet : bool, optional
            Suppress output if True.
        keep_stats : bool, optional
            Keep statistics for debugging if True.
        run_name : str, optional
            Name of the experiment run.
        reward_shaping_parameters (dict, optional): Parameters for reward shaping.
        """
        # Set random seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Random number generators for different part of randomness
        self.rng_initial = get_rng(device, random_seed)
        self.rng_process = get_rng(device, random_seed)
        self.rng_dynamics = get_rng(device, random_seed)

        # Unpack parameters
        self.dt = parameters["dt"]
        self.m_pole_min = parameters["m_pole"][0]
        self.m_pole_max = parameters["m_pole"][1]
        self.m_cart_min = parameters["m_cart"][0]
        self.m_cart_max = parameters["m_cart"][1]
        self.l_min = parameters["l"][0]
        self.l_max = parameters["l"][1]
        self.noise_std = noise_std
        t = lambda arr: torch.tensor(arr, device=device, dtype=torch.float).unsqueeze(0)
        self.Q = t(Q)
        self.R = t(R)

        # Dynamics
        batch_ones = lambda shape: torch.zeros((bs,) + shape, dtype=torch.float32, device=device)
        single_ones = lambda shape: torch.zeros((1,) + shape, dtype=torch.float32, device=device)
        self.m_pole = self.m_pole_min * batch_ones(tuple())
        self.m_cart = self.m_cart_min * batch_ones(tuple())
        self.l = self.l_min * batch_ones(tuple())
        self.m_pole_nom = 0.5 * (self.m_pole_min + self.m_pole_max) * single_ones(tuple())
        self.m_cart_nom = 0.5 * (self.m_cart_min + self.m_cart_max) * single_ones(tuple())
        self.l_nom = 0.5 * (self.l_min + self.l_max) * single_ones(tuple())

        # State and input bounds
        self.x_min = x_min
        self.x_max = x_max
        self.theta_min = -0.5
        self.theta_max = 0.5
        self.u_min = u_min
        self.u_max = u_max
        self.bs = bs

        # States, references, inputs
        batch_zeros = lambda shape: torch.zeros((bs,) + shape, dtype=torch.float32, device=device)
        self.initial_state = batch_zeros((4,))
        self.x = batch_zeros(tuple())
        self.x_ref = batch_zeros(tuple())
        self.x_dot = batch_zeros(tuple())
        self.theta = batch_zeros(tuple())
        self.theta_dot = batch_zeros(tuple())
        self.u = batch_zeros((1,))

        # Episode information
        self.is_done = torch.zeros((bs,), dtype=torch.uint8, device=device)
        self.step_count = torch.zeros((bs,), dtype=torch.int32, device=device)
        self.cumulative_cost = torch.zeros((bs,), dtype=torch.float32, device=device)

        # Gym environment settings
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.state_space = self.observation_space
        self.action_space = gym.spaces.Box(low=self.u_min, high=self.u_max, shape=(1,), dtype=np.float32)
        self.num_states = 6
        self.num_actions = 1

        # Other parameters
        self.barrier_thresh = barrier_thresh
        self.max_steps = max_steps
        self.device = device
        self.quiet = quiet
        self.run_name = run_name
        self.reward_shaping_parameters = reward_shaping_parameters

        # Statistics for testing
        self.keep_stats = keep_stats
        self.already_on_stats = torch.zeros((bs,), dtype=torch.uint8, device=device)   # Each worker can only contribute once to the statistics, to avoid bias towards shorter episodes
        self.stats = pd.DataFrame(columns=['initial_state', 'x_ref', 'episode_length', 'cumulative_cost', 'constraint_violated'])

        self.info_dict = {}

        self.reset()

    def obs(self):
        """Returns the observation from the environment in the format (x, x_dot, theta, theta_dot, x_ref)."""
        return torch.cat([t.unsqueeze(-1) for t in [self.x, self.x_dot, self.theta, self.theta_dot, self.x_ref]], dim=-1)

    def cost(self, state_error=None, u=None):
        """Computes and returns the cost based on the state and control input.

        state_error : torch.Tensor, optional. Defaults to the current state error.
        u: torch.Tensor, optional. Defaults to the current control input.
        """
        if state_error is None:
            state_error = torch.stack([self.x - self.x_ref, self.x_dot, self.theta, self.theta_dot], dim=-1)
        if u is None:
            u = self.u
        return bqf(state_error, self.Q) + bqf(u, self.R)

    def reward(self):
        """Computes and returns the reward based on the cost and episode termination status."""
        cost = self.cost()
        rew_main = -cost
        rew_done = -1.0 * (self.is_done == 1)

        # Reward shaping for address steady-state error: c1 * exp(-c2 * (cost - c3))
        c1 = self.reward_shaping_parameters.get("steady_c1", 10.)
        c2 = self.reward_shaping_parameters.get("steady_c2", 0.1)
        c3 = self.reward_shaping_parameters.get("steady_c3", 0.)
        rew_steady = c1 * torch.exp(-c2 * (cost - c3))

        coef_main = 1.
        coef_done = 1000.
        coef_steady = 1.

        rew_total = coef_main * rew_main + coef_done * rew_done + coef_steady * rew_steady

        self.info_dict["actual_costs"] = cost + coef_done * (self.is_done == 1)

        if not self.quiet:
            avg_rew_main = coef_main * rew_main.mean().item()
            avg_rew_done = coef_done * rew_done.mean().item()
            avg_rew_steady = coef_steady * rew_steady.mean().item()
            avg_rew_total = rew_total.mean().item()
            ic(avg_rew_main, avg_rew_done, avg_rew_steady, avg_rew_total)

        return rew_total

    def done(self):
        """Returns whether the episode has terminated."""
        return self.is_done.bool()

    def info(self):
        """Returns additional information about the environment."""
        self.info_dict["already_on_stats"] = self.already_on_stats
        return self.info_dict

    def get_number_of_agents(self):
        """Returns the number of agents in the environment. Always returns 1 for this environment."""
        return 1

    def get_num_parallel(self):
        """Returns the batch size for parallel environments."""
        return self.bs

    def generate_ref(self, size):
        """Generates and returns reference positions of given size."""
        x_ref = self.x_min + (self.x_max - self.x_min) * torch.rand((size,), device=self.device, generator=self.rng_initial)
        return x_ref

    def generate_initial(self, size):
        """Generates initial state in the form (x, x_dot, theta, theta_dot)."""
        x_min = self.x_min + 0.2
        x_max = self.x_max - 0.2
        x = x_min + (x_max - x_min) * torch.rand((size,), device=self.device, generator=self.rng_initial)
        x_dot_min = -1
        x_dot_max = 1
        x_dot = x_dot_min + (x_dot_max - x_dot_min) * torch.rand((size,), device=self.device, generator=self.rng_initial)
        theta_min = -0.1
        theta_max = 0.1
        theta = theta_min + (theta_max - theta_min) * torch.rand((size,), device=self.device, generator=self.rng_initial)
        theta_dot_min = -0.1
        theta_dot_max = 0.1
        theta_dot = theta_dot_min + (theta_dot_max - theta_dot_min) * torch.rand((size,), device=self.device, generator=self.rng_initial)
        return x, x_dot, theta, theta_dot


    def reset_done_envs(self, need_reset=None, x=None, x_ref=None, randomize_seed=None):
        """
        Resets the state and parameters of environments that have reached termination conditions.

        Parameters
        ----------
        need_reset : torch.Tensor, optional
            Boolean tensor of shape (bs,) specifying which environments need to be reset.
            If None, it will use the current 'is_done' status.
        x : torch.Tensor, optional
            Initial position for the cart in reset environments. If None, it is randomly generated.
        x_ref : torch.Tensor, optional
            Reference position for the cart in reset environments. If None, it is randomly generated.
        randomize_seed : int, optional
            Seed for random number generation if parameters like mass and length need to be randomized.

        Notes
        -----
        - The function updates the internal state variables like position (x), velocity (x_dot), angle (theta),
        angular velocity (theta_dot), and other episode-specific counters and flags.
        - If randomize_seed is not None, the function will randomize the mass of the pole, cart, and the length of the pole.
        - Resets the cumulative cost and step count for environments that need resetting.
        """
        is_done = self.is_done.bool() if need_reset is None else need_reset.bool()
        size = is_done.sum().item()
        self.step_count[is_done] = 0
        self.cumulative_cost[is_done] = 0.
        self.x_ref[is_done] = self.generate_ref(size) if x_ref is None else x_ref
        if x is None:
            x, x_dot, theta, theta_dot = self.generate_initial(size)
        else:
            x_dot = torch.zeros((size,), device=self.device)
            theta = torch.zeros((size,), device=self.device)
            theta_dot = torch.zeros((size,), device=self.device)
        self.initial_state[is_done, 0] = x
        self.initial_state[is_done, 1] = x_dot
        self.initial_state[is_done, 2] = theta
        self.initial_state[is_done, 3] = theta_dot
        self.x[is_done] = x
        self.x_dot[is_done] = x_dot
        # Initialize theta with a small perturbation from 0 (upright position)
        self.theta[is_done] = theta
        self.theta_dot[is_done] = theta_dot
        self.is_done[is_done] = 0
        if randomize_seed is not None:
            # Seed for randomization of dynamics is specified in function call; use it directly
            with torch.random.fork_rng():
                torch.manual_seed(randomize_seed)
                self.m_pole[is_done] = self.m_pole_min + (self.m_pole_max - self.m_pole_min) * torch.rand((size,), device=self.device)
                self.m_cart[is_done] = self.m_cart_min + (self.m_cart_max - self.m_cart_min) * torch.rand((size,), device=self.device)
                self.l[is_done] = self.l_min + (self.l_max - self.l_min) * torch.rand((size,), device=self.device)
        else:
            # No seed specified; use predefined random number generator for randomization of dynamics
            self.m_pole[is_done] = self.m_pole_min + (self.m_pole_max - self.m_pole_min) * torch.rand((size,), device=self.device, generator=self.rng_dynamics)
            self.m_cart[is_done] = self.m_cart_min + (self.m_cart_max - self.m_cart_min) * torch.rand((size,), device=self.device, generator=self.rng_dynamics)
            self.l[is_done] = self.l_min + (self.l_max - self.l_min) * torch.rand((size,), device=self.device, generator=self.rng_dynamics)


    def reset(self, x=None, x_ref=None, randomize_seed=None):
        """Resets the environment and returns the initial observation."""
        # All environments need to be reset
        need_reset = torch.ones((self.bs,), dtype=torch.uint8, device=self.device)
        self.reset_done_envs(need_reset=need_reset, x=x, x_ref=x_ref, randomize_seed=randomize_seed)
        return self.obs()

    def check_constraints(self):
        """Checks and returns whether the state constraints are violated."""
        return (self.x >= self.x_min) & (self.x <= self.x_max) & (self.theta >= self.theta_min) & (self.theta <= self.theta_max)

    def write_episode_stats(self, i):
        """Writes statistics for the i-th episode."""
        self.already_on_stats[i] = 1
        initial_state = self.initial_state[i, :].cpu().numpy()
        x_ref = self.x_ref[i].item()
        episode_length = self.step_count[i].item()
        cumulative_cost = self.cumulative_cost[i].item()
        constraint_violated = (self.is_done[i] == 1).item()
        self.stats.loc[len(self.stats)] = [initial_state, x_ref, episode_length, cumulative_cost, constraint_violated]

    def dump_stats(self, filename=None):
        """Dumps statistics to a CSV file."""
        if filename is None:
            directory = "test_results"
            if not os.path.exists(directory):
                os.makedirs(directory)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = self.run_name
            filename = os.path.join(directory, f"{tag}_{timestamp}.csv")
        self.stats.to_csv(filename, index=False)

    def step(self, u):
        """Takes a step in the environment with control input u and returns the new observation, reward, done flag, and info."""
        self.reset_done_envs()
        u = torch.clamp(u, self.u_min, self.u_max)
        self.u = u
        self.cumulative_cost += self.cost()
        self.step_count += 1

        # Construct batch of matrices, each being [m_cart + m_pole, m_pole * l * cos(theta); m_pole * L * cos(theta), m_pole * l ^ 2]
        lhs_mat = torch.stack([
            torch.stack([self.m_cart + self.m_pole, self.m_pole * self.l * torch.cos(self.theta)], dim=-1),
            torch.stack([self.m_pole * self.l * torch.cos(self.theta), self.m_pole * self.l ** 2], dim=-1),
        ], dim=-2)
        # Construct batch of vectors, each being [u + m_pole * l * theta_dot ^ 2 * sin(theta); m_pole * g * l * sin(theta)]
        u_scalar = self.u.squeeze(-1)   # Remove singleton dimension for control input
        rhs_vec = torch.stack([
            u_scalar + self.m_pole * self.l * self.theta_dot ** 2 * torch.sin(self.theta),
            self.m_pole * 9.8 * self.l * torch.sin(self.theta),
        ], dim=-1)
        # Solve for [x_ddot; theta_ddot]
        acc = bsolve(lhs_mat, rhs_vec)
        # Add noise to acceleration
        acc += self.noise_std * torch.randn(acc.shape, device=self.device, generator=self.rng_process)
        x_ddot = acc[..., 0]
        theta_ddot = acc[..., 1]

        # Update states
        self.x += self.x_dot * self.dt
        self.theta += self.theta_dot * self.dt
        # Wrap to [-pi, pi]
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self.x_dot += x_ddot * self.dt
        self.theta_dot += theta_ddot * self.dt

        # Check constraints
        self.is_done[self.check_constraints() == False] = 1   # 1 for failure
        self.is_done[self.step_count >= self.max_steps] = 2   # 2 for timeout

        # Write episode stats
        if self.keep_stats:
            done_indices = torch.nonzero(self.is_done.to(dtype=torch.bool) & torch.logical_not(self.already_on_stats), as_tuple=False)
            for i in done_indices:
                self.write_episode_stats(i)

        # Return observation, reward, done, info
        return self.obs(), self.reward(), self.done(), self.info()

    def render(self, **kwargs):
        """Renders the environment. Currently, it just prints out state variables and average cost."""
        ic(self.x, self.x_ref, self.xdot, self.theta, self.theta_dot)
        avg_cost = (self.cumulative_cost / self.step_count).cpu().numpy()
        ic(avg_cost)
