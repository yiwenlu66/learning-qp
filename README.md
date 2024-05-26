# MPC-Inspired Reinforcement Learning for Verifiable Model-Free Control

Code for the paper: [MPC-Inspired Reinforcement Learning for Verifiable Model-Free Control](https://arxiv.org/pdf/2312.05332)

## Installation

```
git clone --recursive git@github.com:yiwenlu66/learning-qp.git
pip install -r requirements.txt
```

Note: the `--recursive` option is necessary to make the code work correctly.

## Usage

```
python train_or_test env_name [--options]
```

The following scripts are also provided to reproduce the results in the paper:

- `experiments/tank/reproduce.sh` for reproducing the first part of Table 1
- `experiments/cartpole/reproduce.sh` for reproducing the second part of Table 1
- `experiments/tank/reproduce_disturbed.sh` for reproducing Table 2

**These scripts are run on GPU by default.** After running each reproducing script, the following data will be saved:

- Training logs in tensorboard format will be saved in `runs`
- Test results, including the trial output for each experiment and a summary table, all in CSV format, will be saved in `test_results`

## Code structure

- `rl_games`: A customized version of the [rl_games](https://github.com/Denys88/rl_games) library for RL training
- `src/envs`: GPU parallelized simulation environments, with interface similar to [Isaac Gym](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- `src/modules`: PyTorch modules, including the proposed QP-based policy and the underlying differentiable QP solver
- `src/networks`: Wrapper around the QP-based policy for interfacing with `rl_games`
- `src/utils`: Utility functions (customized PyTorch operations, MPC baselines, etc.)
- `experiments`: Sample scripts for running experiments

## License

The project is released under the MIT license. See [LICENSE](LICENSE) for details.

Part of the project is modified from [rl_games](https://github.com/Denys88/rl_games).

## Citation

If you find this project useful in your research, please consider citing:

```
@InProceedings{lu2024mpc,
  title={MPC-Inspired Reinforcement Learning for Verifiable Model-Free Control},
  author={Lu, Yiwen and Li, Zishuo and Zhou, Yihan and Li, Na and Mo, Yilin},
  booktitle={Proceedings of the 6th Conference on Learning for Dynamics and Control},
  year={2024}
}
```
