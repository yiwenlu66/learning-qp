#!/bin/bash

# Store the command line argument in a global variable
TRAIN_OR_TEST="$1"

run_imitate() {
    export CUDA_VISIBLE_DEVICES=0
    n_qp=8
    m_qp=32
    noise_level=0
    python ../../run.py $TRAIN_OR_TEST tank \
    --num-parallel 100000 \
    --horizon 20 \
    --epochs 200 \
    --mini-epochs 1 \
    --qp-unrolled \
    --shared-PH \
    --affine-qb \
    --noise-level ${noise_level} \
    --n-qp ${n_qp} \
    --m-qp ${m_qp} \
    --imitate-mpc-N 10 \
    --exp-name imitate
}

run_fine_tune() {
    export CUDA_VISIBLE_DEVICES=0
    n_qp=8
    m_qp=32
    noise_level=0
    python ../../run.py $TRAIN_OR_TEST tank \
    --num-parallel 100000 \
    --horizon 20 \
    --epochs 400 \
    --mini-epochs 1 \
    --qp-unrolled \
    --shared-PH \
    --affine-qb \
    --noise-level ${noise_level} \
    --n-qp ${n_qp} \
    --m-qp ${m_qp} \
    --initialize-from-experiment imitate \
    --no-obs-normalization \
    --exp-name fine_tune
}

run_imitate
run_fine_tune
