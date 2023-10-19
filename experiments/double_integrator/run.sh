#!/bin/bash

# Run the double integrator experiment
TRAIN_OR_TEST="$1"

n_qp=3
m_qp=9
noise_level=0

python ../../run.py $TRAIN_OR_TEST double_integrator \
            --num-parallel 100000 \
            --horizon 20 \
            --epochs 500 \
            --max-steps-per-episode 100 \
            --mini-epochs 1 \
            --qp-unrolled \
            --shared-PH \
            --affine-qb \
            --noise-level ${noise_level} \
            --n-qp ${n_qp} \
            --m-qp ${m_qp} \
            --no-obs-normalization \
            --use-residual-loss \
            --force-feasible \
            --no-q-bias \
            --no-b \
            --exp-name default

# python ../../run.py $TRAIN_OR_TEST double_integrator \
#         --num-parallel 100000 \
#         --horizon 20 \
#         --epochs 2000 \
#         --mini-epochs 1 \
#         --noise-level ${noise_level} \
#         --exp-name mlp