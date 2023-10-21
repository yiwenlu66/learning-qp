#!/bin/bash

# Run the double integrator experiment
TRAIN_OR_TEST="$1"

n_qp=3
m_qp=9
noise_level=0

g1() {
    export CUDA_VISIBLE_DEVICES=0
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
            --exp-name default
}

g2() {
    export CUDA_VISIBLE_DEVICES=1
    # python ../../run.py $TRAIN_OR_TEST double_integrator \
    #         --num-parallel 100000 \
    #         --horizon 20 \
    #         --epochs 500 \
    #         --max-steps-per-episode 100 \
    #         --mini-epochs 1 \
    #         --qp-unrolled \
    #         --shared-PH \
    #         --affine-qb \
    #         --noise-level ${noise_level} \
    #         --n-qp ${n_qp} \
    #         --m-qp ${m_qp} \
    #         --no-obs-normalization \
    #         --use-residual-loss \
    #         --force-feasible \
    #         --symmetric \
    #         --exp-name symmetric
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
            --symmetric \
            --no-b \
            --exp-name symmetric_no_b
}

g3() {
    export CUDA_VISIBLE_DEVICES=1
    python ../../run.py $TRAIN_OR_TEST double_integrator \
         --num-parallel 100000 \
         --horizon 20 \
         --epochs 500 \
         --max-steps-per-episode 100 \
         --mini-epochs 1 \
         --noise-level ${noise_level} \
         --exp-name mlp
}

# g1 & g2 & g3 &
g2 &

wait
