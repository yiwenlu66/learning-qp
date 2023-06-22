import argparse
import os
import sys
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "rl_games"))
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from utils.rlgame_utils import RLGPUEnv, RLGPUAlgoObserver
from envs.linear_system import LinearSystem
import yaml
import torch
import glob
import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import numpy as np


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

parser = argparse.ArgumentParser()
parser.add_argument("train_or_test", type=str, help="Train or test")
parser.add_argument("env", type=str)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--exp-name", type=str, default="default")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--num-parallel", type=int, default=100000)
parser.add_argument("--mini-epochs", type=int, default=5)
parser.add_argument("--mlp-size-last", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--horizon", type=int, default=200)
parser.add_argument("--score-to-win", type=int, default=20000)
parser.add_argument("--save-freq", type=int, default=10)
parser.add_argument("--epoch-index", type=int, default=-1, help="For test only, -1 for using latest")
parser.add_argument("--quiet", action='store_true')
parser.add_argument("--device", type=str, default='cuda:0')
args = parser.parse_args()


def get_num_parallel():
    if args.train_or_test == "train":
        return args.num_parallel
    elif args.train_or_test == "test":
        return 1

envs = {
    "tank": lambda **kwargs: LinearSystem(
        A=np.array([
            [0.984,  0.0,      0.0422029,  0.0],
            [0.0,    0.98895,  0.0,        0.0326014],
            [0.0,    0.0,      0.957453,   0.0],
            [0.0,    0.0,      0.0,        0.967216],
        ]),
        B=np.array([
            [0.825822,    0.0101995],
            [0.00512673,  0.624648], 
            [0.0,         0.468317],
            [0.307042,    0.0],
        ]),
        Q=np.eye(4),
        R=0.1 * np.eye(2),
        sqrt_W=np.eye(4),
        x_min=np.zeros(4),
        x_max=20 * np.ones(4),
        u_min=np.zeros(2),
        u_max=8 * np.ones(2),
        barrier_thresh=0.5,
        max_steps=200,
        **kwargs,
    ),
}

default_env_config = {
    "random_seed": args.seed,
    "quiet": args.quiet,
    "device": args.device,
    "bs": args.num_parallel,
}
blacklist_keys = lambda d, blacklist: {k: d[k] for k in d if not (k in blacklist)}
vecenv.register('RLGPU',
                lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'vecenv_type': 'RLGPU',
    'env_creator': lambda **env_config: envs[args.env](
        **blacklist_keys(default_env_config, env_config.keys()),
        **env_config,
    ),
})

runner = Runner(RLGPUAlgoObserver())
file_path = os.path.dirname(__file__)
with open(os.path.join(file_path, "runner_config.yaml")) as f:
    runner_config = yaml.safe_load(f)
full_experiment_name = args.env + "_" + args.exp_name
runner_config["params"]["seed"] = args.seed
runner_config["params"]["config"]["num_actors"] = args.num_parallel
runner_config["params"]["config"]["max_epochs"] = args.epochs
runner_config["params"]["config"]["minibatch_size"] = args.num_parallel
runner_config["params"]["config"]["games_to_track"] = args.num_parallel
runner_config["params"]["config"]["mini_epochs"] = args.mini_epochs
runner_config["params"]["config"]["gamma"] = args.gamma
runner_config["params"]["config"]["horizon_length"] = args.horizon
runner_config["params"]["config"]["score_to_win"] = args.score_to_win
runner_config["params"]["config"]["name"] = args.env
runner_config["params"]["config"]["full_experiment_name"] = full_experiment_name
runner_config["params"]["network"]["mlp"]["units"] = [args.mlp_size_last * i for i in (4, 2, 1)]
runner_config["params"]["config"]["save_frequency"] = args.save_freq
runner_config["params"]["config"]["device"] = args.device
runner_config["params"]["network"].pop("rnn")

if args.quiet:
    with suppress_stdout_stderr():
        runner.load(runner_config)
else:
    runner.load(runner_config)

if __name__ == "__main__":
    if args.train_or_test == "train":
        runner.run({
            'train': True,
        })
    elif args.train_or_test == "test":
        checkpoint_dir = f"runs/{full_experiment_name}/nn"
        if args.epoch_index == -1:
            checkpoint_name = f"{checkpoint_dir}/{args.env}.pth"
        else:
            list_of_files = glob.glob(f"{checkpoint_dir}/last_{args.env}_ep_{args.epoch_index}_rew_*.pth")
            checkpoint_name = max(list_of_files, key=os.path.getctime)
        runner.run({
            'train': False,
            'play': True,
            'checkpoint' : checkpoint_name,
        })
