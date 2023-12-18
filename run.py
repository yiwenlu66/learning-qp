import argparse
import os
import sys
file_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_path, "rl_games"))
import yaml
import torch
import glob
import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import numpy as np

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

from src.envs.env_creators import env_creators, sys_param
from src.envs.mpc_baseline_parameters import get_mpc_baseline_parameters
from src.utils.rlgame_utils import RLGPUEnv, RLGPUAlgoObserver
from src.networks.a2c_qp_unrolled import A2CQPUnrolledBuilder

model_builder.register_network('qp_unrolled', A2CQPUnrolledBuilder)

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def float_list(string):
    """Convert a string into a list of floats."""
    try:
        return [float(item) for item in string.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Argument must be a comma-separated list of floats")


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
parser.add_argument("--strict-affine-layer", action="store_true")
parser.add_argument("--obs-has-half-ref", action="store_true")
parser.add_argument("--symmetric", action="store_true")
parser.add_argument("--no-b", action="store_true")
parser.add_argument("--warm-start", action="store_true")
parser.add_argument("--ws-loss-coef", type=float, default=10.)
parser.add_argument("--ws-update-rate", type=float, default=0.1)
parser.add_argument("--batch-test", action="store_true")
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--randomize", action="store_true")
parser.add_argument("--use-residual-loss", action="store_true")
parser.add_argument("--no-obs-normalization", action="store_true")
parser.add_argument("--imitate-mpc-N", type=int, default=0)
parser.add_argument("--initialize-from-experiment", type=str, default="")
parser.add_argument("--force-feasible", action="store_true")
parser.add_argument("--skip-to-steady-state", action="store_true")
parser.add_argument("--initial-lr", type=float, default=3e-4)
parser.add_argument("--lr-schedule", type=str, default="adaptive")
parser.add_argument("--reward-shaping", type=float_list, default=[0., 1., 0.])

parser.add_argument("--mpc-baseline-N", type=int, default=0)
parser.add_argument("--use-osqp-for-mpc", action="store_true")
parser.add_argument("--mpc-terminal-cost-coef", type=float, default=0.)
parser.add_argument("--robust-mpc-method", type=str, default="none", choices=["none", "scenario", "tube"])
parser.add_argument("--tube-mpc-tube-size", type=float, default=0.)
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
    "randomize": args.randomize,
    "skip_to_steady_state": args.skip_to_steady_state,
    "reward_shaping": args.reward_shaping,
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
runner_config["params"]["config"]["train_or_test"] = args.train_or_test
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
runner_config["params"]["config"]["learning_rate"] = args.initial_lr
runner_config["params"]["config"]["lr_schedule"] = args.lr_schedule
if args.no_obs_normalization:
    runner_config["params"]["config"]["normalize_input"] = False

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
        "strict_affine_layer": args.strict_affine_layer,
        "obs_has_half_ref": args.obs_has_half_ref,
        "use_warm_starter": args.warm_start,
        "train_warm_starter": args.warm_start and args.train_or_test == "train",
        "ws_loss_coef": args.ws_loss_coef,
        "ws_update_rate": args.ws_update_rate,
        "mpc_baseline": None if (not args.mpc_baseline_N and not args.imitate_mpc_N) else {**get_mpc_baseline_parameters(args.env, args.mpc_baseline_N or args.imitate_mpc_N, noise_std=args.noise_level), "terminal_coef": args.mpc_terminal_cost_coef},
        "imitate_mpc": args.imitate_mpc_N > 0,
        "use_osqp_for_mpc": args.use_osqp_for_mpc,
        "use_residual_loss": args.use_residual_loss,
        "symmetric": args.symmetric,
        "no_b": args.no_b,
        "force_feasible": args.force_feasible,
        "feasible_lambda": 10.,
        "train_or_test": args.train_or_test,
        "run_name": args.run_name,
    }

if args.mpc_baseline_N:
    # Unset observation and action normalization
    runner_config["params"]["config"]["clip_actions"] = False
    runner_config["params"]["config"]["normalize_input"] = False

if args.imitate_mpc_N:
    # Unset observation normalization
    runner_config["params"]["config"]["normalize_input"] = False
    # Make MPC output normalized action
    runner_config["params"]["network"]["custom"]["mpc_baseline"]["normalize"] = True

if args.robust_mpc_method != "none":
    runner_config["params"]["network"]["custom"]["mpc_baseline"]["robust_method"] = args.robust_mpc_method
    runner_config["params"]["network"]["custom"]["mpc_baseline"]["max_disturbance_per_dim"] = args.tube_mpc_tube_size

if args.quiet:
    with suppress_stdout_stderr():
        runner.load(runner_config)
else:
    runner.load(runner_config)

if __name__ == "__main__":
    if args.train_or_test == "train":
        runner_arg = {
            'train': True,
            'play': False,
        }
        if args.initialize_from_experiment:
            full_checkpoint_name = args.env + "_" + args.initialize_from_experiment
            checkpoint_dir = f"runs/{full_checkpoint_name}/nn"
            checkpoint_name = f"{checkpoint_dir}/{args.env}.pth"
            runner_arg['checkpoint'] = checkpoint_name
        runner.run(runner_arg)
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
