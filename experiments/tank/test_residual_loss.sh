#!/bin/bash

# Store the command line argument in a global variable
TRAIN_OR_TEST="$1"

group_one() {
    export CUDA_VISIBLE_DEVICES=0
    n_qp=2
    m_qp=64
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
    --exp-name residual_loss_off
}

group_two() {
    export CUDA_VISIBLE_DEVICES=1
    n_qp=2
    m_qp=64
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
    --exp-name residual_loss_on
}

# Start both groups in parallel
group_one & group_two &

# Wait for both background tasks to complete
wait

