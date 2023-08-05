#!/bin/sh

# python ../../run.py $1 tank --num-parallel 100000 --horizon 20 --epochs 300 --gamma 0.999 --mini-epochs 1 --exp-name vanilla_rl
# python ../../run.py $1 tank --num-parallel 100000 --horizon 20 --epochs 300 --mini-epochs 1 --qp-unrolled --exp-name qp_unrolled
python ../../run.py $1 tank --num-parallel 100000 --horizon 20 --epochs 300 --mini-epochs 1 --qp-unrolled --shared-PH --exp-name qp_unrolled_shared
