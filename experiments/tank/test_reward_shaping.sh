#!/bin/bash

# Define the number of GPUs available
NUM_GPUS=6  # Change this to the number of GPUs you have

# Function to find the first idle GPU
find_idle_gpu() {
    for (( i=0; i<$NUM_GPUS; i++ )); do
        # Check if there are no processes on GPU i
        if [ -z "$(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader)" ]; then
            echo $i
            return
        fi
    done
    echo "-1"  # Return -1 if no idle GPU is found
}

# Function to run the Python script on a specific GPU
run_task() {
    local gpu_id=$1
    local c1=$2
    local c2=$3
    local c3=$4
    local n_qp=$5
    local m_qp=$6
    CUDA_VISIBLE_DEVICES=$gpu_id python ../../run.py train tank --num-parallel 100000 --horizon 20 --epochs 5000 --mini-epochs 1 --noise-level 0. --reward-shaping ${c1},${c2},${c3} --qp-unrolled --n-qp $n_qp --m-qp $m_qp --shared-PH --affine-qb --use-residual-loss --no-obs-normalization --force-feasible --exp-name qp_test_reward_shaping_${c1}_${c2}_${c3}_${n_qp}_${m_qp} --lr-schedule linear --initial-lr "5e-4" &
}

# # Main loop for grid search
# for c1 in 50; do
#     for c2 in 0.05; do
#         for c3 in 2; do
#             gpu_id=-1
#             # Wait for an idle GPU to become available
#             while [ $gpu_id -eq -1 ]; do
#                 gpu_id=$(find_idle_gpu)
#                 sleep 1  # Wait a bit before checking again
#             done

#             run_task $gpu_id $c1 $c2 $c3 4 24
#             # Optional: wait briefly to allow the task to start
#             sleep 10

#             gpu_id=-1
#             # Wait for an idle GPU to become available
#             while [ $gpu_id -eq -1 ]; do
#                 gpu_id=$(find_idle_gpu)
#                 sleep 1  # Wait a bit before checking again
#             done
#             run_task $gpu_id $c1 $c2 $c3 8 48
#             # Optional: wait briefly to allow the task to start
#             sleep 10
#         done
#     done
# done

run_task 1 50 0.05 2 4 24

# Wait for all background jobs to finish
wait
