#!/bin/bash

N=16
noise=0

python ../../run.py test cartpole --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N ${N} --noise-level ${noise} --batch-test --use-osqp-for-mpc --mpc-terminal-cost-coef 10 --exp-name qp_test_reward_shaping_10_0.1_0_8_48 --run-name mpc_10t --max-steps-per-episode 100
