#!/bin/bash

# Store the command line argument in a global variable
TRAIN_OR_TEST="$1"

n_qp=8
m_qp=32
noise_level=0
python ../../run.py $TRAIN_OR_TEST tank \
--num-parallel 100000 \
--horizon 20 \
--epochs 2000 \
--mini-epochs 1 \
--qp-unrolled \
--shared-PH \
--affine-qb \
--noise-level ${noise_level} \
--n-qp ${n_qp} \
--m-qp ${m_qp} \
--use-residual-loss \
--no-obs-normalization \
--skip-to-steady-state \
--lr-schedule linear \
--exp-name test_skip_steady