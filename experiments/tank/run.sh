#!/bin/bash

# Store the command line argument in a global variable
TRAIN_OR_TEST="$1"

# Function to run commands 1-3 sequentially with VAR=1
group_one() {
    export CUDA_VISIBLE_DEVICES=0
    python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --gamma 0.999 --mini-epochs 1 --exp-name vanilla_rl
    python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --qp-iter 100 --exp-name qp_unrolled_shared_more_iter
}

# Function to run commands 4-6 sequentially with VAR=2
group_two() {
    export CUDA_VISIBLE_DEVICES=1
    python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --exp-name qp_unrolled
    python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --warm-start --exp-name qp_unrolled_ws
    python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --exp-name qp_unrolled_shared
    python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 2000 --mini-epochs 1 --qp-unrolled --shared-PH --qp-iter 10 --warm-start --exp-name qp_unrolled_shared_ws
    # python ../../run.py $TRAIN_OR_TEST tank --num-parallel 100000 --horizon 20 --epochs 1 --mini-epochs 1 --qp-unrolled --shared-PH --exp-name computation-test
}

# Start both groups in parallel
group_one & group_two &

# Wait for both background tasks to complete
wait
