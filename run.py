import argparse
import os
import sys
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "rl_games"))
import yaml
import torch
import glob
import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import numpy as np

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

from envs.env_creators import env_creators, sys_param
from utils.rlgame_utils import RLGPUEnv, RLGPUAlgoObserver
from networks.a2c_qp_unrolled import A2CQPUnrolledBuilder

model_builder.register_network('qp_unrolled', A2CQPUnrolledBuilder)

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

parser = argparse.ArgumentParser()
parser.add_argument("train_or_test", type=str, help="Train or test")
parser.add_argument("env", type=str)
parser.add_argument("--noise-level", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--exp-name", type=str, default="default")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--num-parallel", type=int, default=100000)
parser.add_argument("--mini-epochs", type=int, default=5)
parser.add_argument("--mlp-size-last", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--horizon", type=int, default=200)
parser.add_argument("--max-steps-per-episode", type=int, default=500)
parser.add_argument("--score-to-win", type=int, default=int(1e9))
parser.add_argument("--save-freq", type=int, default=10)
parser.add_argument("--epoch-index", type=int, default=-1, help="For test only, -1 for using latest")
parser.add_argument("--quiet", action='store_true')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--qp-unrolled", action='store_true')
parser.add_argument("--n-qp", type=int, default=5)
parser.add_argument("--m-qp", type=int, default=4)
parser.add_argument("--qp-iter", type=int, default=10)
parser.add_argument("--shared-PH", action="store_true")
parser.add_argument("--affine-qb", action="store_true")
parser.add_argument("--warm-start", action="store_true")
parser.add_argument("--ws-loss-coef", type=float, default=10.)
parser.add_argument("--ws-update-rate", type=float, default=0.1)
parser.add_argument("--mpc-baseline-N", type=int, default=0)
parser.add_argument("--batch-test", action="store_true")
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--use-osqp-for-mpc", action="store_true")
args = parser.parse_args()


def get_num_parallel():
    if args.train_or_test == "train":
        return args.num_parallel
    elif args.train_or_test == "test":
        if args.batch_test:
            return args.num_parallel
        else:
            return 1

default_env_config = {
    "random_seed": args.seed,
    "quiet": args.quiet,
    "device": args.device,
    "bs": get_num_parallel(),
    "noise_level": args.noise_level,
    "max_steps": args.max_steps_per_episode,
    "keep_stats": (args.train_or_test == "test"),
    "run_name": args.run_name or args.exp_name,
    "exp_name": args.exp_name,
}

blacklist_keys = lambda d, blacklist: {k: d[k] for k in d if not (k in blacklist)}
vecenv.register('RLGPU',
                lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'vecenv_type': 'RLGPU',
    'env_creator': lambda **env_config: env_creators[args.env](
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
runner_config["params"]["config"]["steps_to_track_per_game"] = args.max_steps_per_episode
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

if args.batch_test:
    runner_config["params"]["config"]["player"]["games_num"] = args.num_parallel

if args.qp_unrolled:
    runner_config["params"]["network"]["name"] = "qp_unrolled"
    runner_config["params"]["network"]["custom"] = {
        "device": args.device,
        "n_qp": args.n_qp,
        "m_qp": args.m_qp,
        "qp_iter": args.qp_iter,
        "shared_PH": args.shared_PH,
        "affine_qb": args.affine_qb,
        "use_warm_starter": args.warm_start,
        "train_warm_starter": args.warm_start and args.train_or_test == "train",
        "ws_loss_coef": args.ws_loss_coef,
        "ws_update_rate": args.ws_update_rate,
        "mpc_baseline": None if not args.mpc_baseline_N else {
            "n_mpc": sys_param[args.env]["n"],
            "m_mpc": sys_param[args.env]["m"],
            "N": args.mpc_baseline_N,
            **sys_param[args.env]
        },
        "use_osqp_for_mpc": args.use_osqp_for_mpc,
    }

if args.mpc_baseline_N:
    # Unset observation and action normalization
    runner_config["params"]["config"]["clip_actions"] = False
    runner_config["params"]["config"]["normalize_input"] = False

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
        if not args.mpc_baseline_N:
            checkpoint_dir = f"runs/{full_experiment_name}/nn"
            if args.epoch_index == -1:
                checkpoint_name = f"{checkpoint_dir}/{args.env}.pth"
            else:
                list_of_files = glob.glob(f"{checkpoint_dir}/last_{args.env}_ep_{args.epoch_index}_rew_*.pth")
                checkpoint_name = max(list_of_files, key=os.path.getctime)
        else:
            checkpoint_name = None
        runner.run({
            'train': False,
            'play': True,
            'checkpoint' : checkpoint_name,
        })
