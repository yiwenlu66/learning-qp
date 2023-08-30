#!/bin/bash

for N in 0 1 2 4 8 16; do
    for noise in 0 0.1 0.2 0.5; do
        python ../../run.py test tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --affine-qb --mpc-baseline-N ${N} --noise-level ${noise} --batch-test --quiet --exp-name qp_unrolled_shared_affine --run-name N${N}_noise${noise}
    done
done
