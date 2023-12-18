#!/bin/bash

group_one() {
    export CUDA_VISIBLE_DEVICES=0
    # for N in 1 2 4 8 16; do
    for N in 4; do
        # for noise in 0 0.1; do
        for noise in 0.1; do
            python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N ${N} --noise-level ${noise} --batch-test --quiet --randomize --exp-name qp_unrolled_shared_affine --run-name N${N}_noise${noise}_rand
        done
    done
    # # for noise in 0 0.1; do
    # for noise in 0.1; do
    #     # for n in 2 4 8 16; do
    #     for n in 8; do
    #         # for m in 2 4 8 16 32 64; do
    #         for m in 32; do
    #             # python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --noise-level ${noise} --batch-test --n-qp ${n} --m-qp ${m} --quiet --exp-name shared_affine_noise${noise}_n${n}_m${m}+rand --randomize --run-name N0_n${n}_m${m}_noise${noise}_rand
    #             python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --noise-level ${noise} --batch-test --n-qp ${n} --m-qp ${m} --quiet --randomize --exp-name shared_affine_noise${noise}_n${n}_m${m}+rand --run-name N0_n${n}_m${m}_noise${noise}_rand
    #         done
    #     done
    # done
}

group_two() {
    export CUDA_VISIBLE_DEVICES=1
    # for N in 1 2 4 8 16; do
    #     for noise in 0.2 0.5; do
    #         python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N ${N} --noise-level ${noise} --batch-test --quiet --exp-name qp_unrolled_shared_affine --randomize --run-name N${N}_noise${noise}_rand --use-osqp-for-mpc
    #     done
    # done
    # for noise in 0.2 0.5; do
    #     for n in 2 4 8 16; do
    #         for m in 2 4 8 16 32 64; do
    #             python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --noise-level ${noise} --batch-test --n-qp ${n} --m-qp ${m} --quiet --exp-name shared_affine_noise${noise}_n${n}_m${m}+rand --randomize --run-name N0_n${n}_m${m}_noise${noise}_rand
    #         done
    #     done
    # done
    # for noise in 0 0.1 0.2 0.5; do
    for noise in 0.1; do
        python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --mpc-baseline-N 0 --noise-level ${noise} --batch-test --exp-name mlp_noise${noise}+rand --randomize --run-name mlp_noise${noise}_rand
    done
}

# Start both groups in parallel
# group_one & group_two &
group_two

# Wait for both background tasks to complete
# wait
