#!/bin/bash

for N in 16 8 4 2 1 0; do
    for noise in 0 0.1 0.2 0.5; do
        python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N ${N} --noise-level ${noise} --batch-test --quiet --exp-name qp_unrolled_shared_affine --run-name N${N}_noise${noise}
    done
done

for noise in 0 0.1 0.2 0.5; do
    for n in 2 4 8 16; do
        for m in 2 4 8 16; do
            python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --noise-level ${noise} --batch-test --n-qp ${n} --m-qp ${m} --quiet --exp-name shared_affine_noise${noise}_n${n}_m${m} --run-name N0_n${n}_m${m}_noise${noise}
        done
    done
done

for noise in 0 0.1 0.2 0.5; do
    python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 2000 --mini-epochs 1 --mpc-baseline-N 0 --noise-level ${noise} --batch-test --quiet --exp-name vanilla_rl --run-name mlp_noise${noise}
done
