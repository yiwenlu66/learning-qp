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
    CUDA_VISIBLE_DEVICES=$gpu_id python ../../run.py train tank --num-parallel 500000 --horizon 20 --epochs 4000 --mini-epochs 1 --noise-level 0. --reward-shaping ${c1},${c2},${c3} --exp-name mlp_test_reward_shaping_batch3_${c1}_${c2}_${c3} --lr-schedule linear &
}

# Main loop for grid search
for c1 in 20 30 50; do
    for c2 in 0.2 0.3 0.5; do
        for c3 in 1 2 3; do
            gpu_id=-1
            # Wait for an idle GPU to become available
            while [ $gpu_id -eq -1 ]; do
                gpu_id=$(find_idle_gpu)
                sleep 1  # Wait a bit before checking again
            done

            run_task $gpu_id $c1 $c2 $c3

            # Optional: wait briefly to allow the task to start
            sleep 10
        done
    done
done

# Wait for all background jobs to finish
wait
