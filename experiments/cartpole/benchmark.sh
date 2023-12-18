#!/bin/bash

group_one() {
    export CUDA_VISIBLE_DEVICES=0
    # for N in 1 4 16; do
    #     for noise in 0 0.5; do
    #         python ../../run.py test cartpole --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N ${N} --noise-level ${noise} --batch-test --quiet --exp-name qp_unrolled_shared_affine --run-name N${N}_noise${noise} --use-osqp-for-mpc
    #     done
    # done
    for N in 1 4 16; do
        for noise in 0.5; do
            python ../../run.py test cartpole --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N ${N} --noise-level ${noise} --batch-test --quiet --exp-name qp_unrolled_shared_affine --randomize --run-name N${N}_noise${noise}_rand --use-osqp-for-mpc
        done
    done
    # for noise in 0 0.5; do
    #     for n in 2 16; do
    #         for m in 4 64; do
    #             python ../../run.py test cartpole --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --noise-level ${noise} --batch-test --n-qp ${n} --m-qp ${m} --quiet --exp-name shared_affine_noise${noise}_n${n}_m${m} --run-name N0_n${n}_m${m}_noise${noise}
    #         done
    #     done
    # done
    # for noise in 0 0.5; do
    #     for n in 2 16; do
    #         for m in 4 64; do
    #             python ../../run.py test cartpole --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --noise-level ${noise} --batch-test --n-qp ${n} --m-qp ${m} --quiet --exp-name shared_affine_noise${noise}_n${n}_m${m}+rand --randomize --run-name N0_n${n}_m${m}_noise${noise}_rand
    #         done
    #     done
    # done
}

group_two() {
    export CUDA_VISIBLE_DEVICES=1
    # for N in 1 2 4 8 16; do
    #     for noise in 0.2 0.5; do
    #         python ../../run.py test cartpole --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N ${N} --noise-level ${noise} --batch-test --quiet --exp-name qp_unrolled_shared_affine --randomize --run-name N${N}_noise${noise}_rand --use-osqp-for-mpc
    #     done
    # done
    # for noise in 0.2 0.5; do
    #     for n in 2 4 8 16; do
    #         for m in 2 4 8 16 32 64; do
    #             python ../../run.py test cartpole --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --noise-level ${noise} --batch-test --n-qp ${n} --m-qp ${m} --quiet --exp-name shared_affine_noise${noise}_n${n}_m${m}+rand --randomize --run-name N0_n${n}_m${m}_noise${noise}_rand
    #         done
    #     done
    # done
    # for noise in 0 0.5; do
    #     python ../../run.py test cartpole --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --mpc-baseline-N 0 --noise-level ${noise} --batch-test --exp-name mlp_noise${noise} # --run-name mlp_noise${noise}
    # done
    # for noise in 0 0.5; do
    #     python ../../run.py test cartpole --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --mpc-baseline-N 0 --noise-level ${noise} --batch-test --exp-name mlp_noise${noise}+rand --randomize --run-name mlp_noise${noise}_rand
    # done
}

# Start both groups in parallel
group_one & group_two &

# Wait for both background tasks to complete
wait
