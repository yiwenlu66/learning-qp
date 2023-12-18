#!/bin/bash

# Store the command line argument in a global variable
TRAIN_OR_TEST="$1"

# Function to run commands 1-3 sequentially with VAR=1
group_one() {
    export CUDA_VISIBLE_DEVICES=0
    for n_qp in 8; do
        for m_qp in 32; do
            for noise_level in 0; do
            python ../../run.py $TRAIN_OR_TEST cartpole \
            --num-parallel 100000 \
            --horizon 20 \
            --epochs 500 \
            --mini-epochs 1 \
            --qp-unrolled \
            --shared-PH \
            --affine-qb \
            --noise-level ${noise_level} \
            --n-qp ${n_qp} \
            --m-qp ${m_qp} \
            --exp-name shared_affine_noise${noise_level}_n${n_qp}_m${m_qp}
            done
        done
    done
    # for n_qp in 2 16; do
    #     for m_qp in 4 64; do
    #         for noise_level in 0; do
    #         python ../../run.py $TRAIN_OR_TEST cartpole \
    #         --num-parallel 100000 \
    #         --horizon 20 \
    #         --epochs 500 \
    #         --mini-epochs 1 \
    #         --qp-unrolled \
    #         --shared-PH \
    #         --affine-qb \
    #         --noise-level ${noise_level} \
    #         --n-qp ${n_qp} \
    #         --m-qp ${m_qp} \
    #         --randomize \
    #         --exp-name shared_affine_noise${noise_level}_n${n_qp}_m${m_qp}+rand
    #         done
    #     done
    # done
    # for noise_level in 0 0.5; do
    #     python ../../run.py $TRAIN_OR_TEST cartpole \
    #     --num-parallel 100000 \
    #     --horizon 20 \
    #     --epochs 500 \
    #     --mini-epochs 1 \
    #     --noise-level ${noise_level} \
    #     --exp-name mlp_noise${noise_level}
    # done
}

# Function to run commands 4-6 sequentially with VAR=2
group_two() {
    export CUDA_VISIBLE_DEVICES=1
    for n_qp in 2 16; do
        for m_qp in 4 64; do
            for noise_level in 0.5; do
            python ../../run.py $TRAIN_OR_TEST cartpole \
            --num-parallel 100000 \
            --horizon 20 \
            --epochs 500 \
            --mini-epochs 1 \
            --qp-unrolled \
            --shared-PH \
            --affine-qb \
            --noise-level ${noise_level} \
            --n-qp ${n_qp} \
            --m-qp ${m_qp} \
            --exp-name shared_affine_noise${noise_level}_n${n_qp}_m${m_qp}
            done
        done
    done
    for n_qp in 2 16; do
        for m_qp in 4 64; do
            for noise_level in 0.5; do
            python ../../run.py $TRAIN_OR_TEST cartpole \
            --num-parallel 100000 \
            --horizon 20 \
            --epochs 500 \
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

    for noise_level in 0 0.5; do
        python ../../run.py $TRAIN_OR_TEST cartpole \
        --num-parallel 100000 \
        --horizon 20 \
        --epochs 500 \
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
