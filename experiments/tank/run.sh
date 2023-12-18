#!/bin/bash

# Store the command line argument in a global variable
TRAIN_OR_TEST="$1"

# Function to run commands 1-3 sequentially with VAR=1
group_one() {
    export CUDA_VISIBLE_DEVICES=0
    # for n_qp in 2 4 8 16; do
    for n_qp in 8; do
        # for m_qp in 2 4 8 16 32 64; do
        for m_qp in 32; do
            # for noise_level in 0 0.1; do
            for noise_level in 0; do
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
            --no-obs-normalization \
            --no-b \
            --exp-name shared_affine_noise${noise_level}_n${n_qp}_m${m_qp}-norm-b
            done
        done
    done
    # python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --gamma 0.999 --mini-epochs 1 --exp-name vanilla_rl
    # python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N 0 --noise-level 0 --batch-test --quiet --exp-name qp_unrolled_shared_affine
    # python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --qp-iter 100 --exp-name qp_unrolled_shared_more_iter
}

# Function to run commands 4-6 sequentially with VAR=2
group_two() {
    export CUDA_VISIBLE_DEVICES=1
    # python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --exp-name qp_unrolled_shared
    # python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --exp-name qp_unrolled
    # python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --warm-start --exp-name qp_unrolled_ws
    # python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --exp-name qp_unrolled_shared
    # python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --qp-iter 10 --warm-start --exp-name qp_unrolled_shared_ws
    # python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 1 --mini-epochs 1 --qp-unrolled --shared-PH --exp-name computation-test
    for n_qp in 2 4 8 16; do
        for m_qp in 2 4 8 16 32 64; do
            for noise_level in 0.2 0.5; do
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
            --randomize \
            --exp-name shared_affine_noise${noise_level}_n${n_qp}_m${m_qp}+rand
            done
        done
    done
    for noise_level in 0 0.1 0.2 0.5; do
        python ../../run.py $TRAIN_OR_TEST tank \
        --num-parallel 100000 \
        --horizon 20 \
        --epochs 2000 \
        --mini-epochs 1 \
        --noise-level ${noise_level} \
        --randomize \
        --exp-name mlp_noise${noise_level}+rand
    done
}

# Start both groups in parallel
# group_one & group_two &
group_one

# Wait for both background tasks to complete
# wait
