#!/bin/bash

run_task() {
    local gpu_id=$1
    local c1=$2
    local c2=$3
    local c3=$4
    local n_qp=$5
    local m_qp=$6
    CUDA_VISIBLE_DEVICES=$gpu_id python ../../run.py test cartpole --num-parallel 10000 --horizon 20 --epochs 5000 --mini-epochs 1 --noise-level 0. --reward-shaping ${c1},${c2},${c3} --n-qp $n_qp --m-qp $m_qp --shared-PH --affine-qb --use-residual-loss --no-obs-normalization --force-feasible --batch-test --exp-name mlp_test_reward_shaping_${c1}_${c2}_${c3}_${n_qp}_${m_qp} --lr-schedule linear --initial-lr "5e-4" --max-steps-per-episode 100
}

run_task 1 10 0.1 0 8 48 &

wait
