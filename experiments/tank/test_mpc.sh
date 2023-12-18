#!/bin/bash

N=10
noise=0

# python ../../run.py test tank --num-parallel 100 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N ${N} --noise-level ${noise} --batch-test --use-osqp-for-mpc --mpc-terminal-cost-coef 10 --quiet --exp-name qp_unrolled_shared_affine --randomize --noise-level 0.1 --run-name mpc_10t_perturbed --max-steps-per-episode 10
# python ../../run.py test tank --num-parallel 100 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N ${N} --noise-level ${noise} --batch-test --mpc-terminal-cost-coef 10 --quiet --exp-name qp_unrolled_shared_affine --run-name mpc_scenario --robust-mpc-method scenario --randomize --noise-level 0.1 --max-steps-per-episode 10
python ../../run.py test tank --num-parallel 100 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N ${N} --noise-level ${noise} --batch-test --mpc-terminal-cost-coef 10 --quiet --exp-name qp_unrolled_shared_affine --run-name mpc_tube --robust-mpc-method tube --randomize --noise-level 0.1 --max-steps-per-episode 100
