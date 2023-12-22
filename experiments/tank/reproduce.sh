#!/bin/bash

# 0. Background utils and GPU scheduler

# Define the number of GPUs available
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Function to find the first idle GPU
find_idle_gpu() {
    for (( i=0; i<$NUM_GPUS; i++ )); do
        # Check if GPU volatile utilization is 0%
        if [ "$(nvidia-smi -i $i --query-gpu=utilization.gpu --format=csv,noheader,nounits)" -eq 0 ]; then
            echo $i
            return
        fi
    done
    echo "-1"  # Return -1 if no idle GPU is found
}

find_gpu_and_run_task() {
    local run_task_function="$1"
    shift  # Remove the first argument (run_task_function name)

    # Initialize GPU ID as -1 indicating no GPU is available initially
    local gpu_id=-1

    # Wait for an idle GPU to become available
    while [ "$gpu_id" -eq -1 ]; do
        gpu_id=$(find_idle_gpu)
        sleep 1  # Wait a bit before checking again
    done

    # Call the run_task function with the GPU ID and additional arguments, and send it to the background
    $run_task_function $gpu_id $@ > /dev/null &

    # Capture the PID of the last background process
    local task_pid=$!

    # Optional: wait briefly to allow the task to start
    sleep 10

    # Output the PID
    echo $task_pid
}


# 1. Training
# 1.1 MLP of different sizes

train_mlp() {
    local gpu_id=$1
    local c1=$2
    local c2=$3
    local c3=$4
    local mlp_last_size=$5
    CUDA_VISIBLE_DEVICES=$gpu_id python ../../run.py train tank --num-parallel 100000 --horizon 20 --epochs 5000 --mini-epochs 1 --noise-level 0. --reward-shaping ${c1},${c2},${c3} --no-obs-normalization --mlp-size-last $mlp_last_size --batch-test --exp-name reproduce_mlp_${mlp_last_size} --lr-schedule linear --initial-lr "5e-4" --quiet
}

# 1.2 QP of different sizes

train_qp() {
    local gpu_id=$1
    local c1=$2
    local c2=$3
    local c3=$4
    local n_qp=$5
    local m_qp=$6
    CUDA_VISIBLE_DEVICES=$gpu_id python ../../run.py train tank --num-parallel 100000 --horizon 20 --epochs 5000 --mini-epochs 1 --noise-level 0. --reward-shaping ${c1},${c2},${c3} --n-qp $n_qp --m-qp $m_qp --qp-unrolled --shared-PH --affine-qb --strict-affine-layer --obs-has-half-ref --use-residual-loss --no-obs-normalization --force-feasible --batch-test --exp-name reproduce_qp_${n_qp}_${m_qp} --lr-schedule linear --initial-lr "5e-4" --quiet
}


# 2. Testing
# 2.1 MPC under different configurations

test_mpc() {
    local gpu_id=$1
    local N=$2
    local terminal_coef=$3
    local n_qp=4
    local m_qp=24
    CUDA_VISIBLE_DEVICES=$gpu_id python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 5000 --mini-epochs 1 --noise-level 0. --reward-shaping 50,0.05,2 --n-qp $n_qp --m-qp $m_qp --qp-unrolled --shared-PH --affine-qb --strict-affine-layer --obs-has-half-ref --use-residual-loss --no-obs-normalization --force-feasible --batch-test --mpc-baseline-N $N --mpc-terminal-cost-coef $terminal_coef --use-osqp-for-mpc --exp-name reproduce_qp_${n_qp}_${m_qp} --run-name reproduce_mpc_${N}_${terminal_coef} --lr-schedule linear --initial-lr "5e-4" --quiet
}

test_mpc_bg() {
    test_mpc $@ > /dev/null &
}

test_mpc_all() {
    test_mpc_bg 0 2 0
    test_mpc_bg 0 2 1
    test_mpc_bg 0 2 10
    test_mpc_bg 0 2 100
    test_mpc_bg 0 4 0
    test_mpc_bg 0 4 1
    test_mpc_bg 0 4 10
    test_mpc_bg 0 4 100
    test_mpc_bg 1 8 0
    test_mpc_bg 1 8 1
    test_mpc_bg 1 8 10
    test_mpc_bg 1 8 100
    test_mpc_bg 1 16 0
    test_mpc_bg 1 16 1
    test_mpc_bg 1 16 10
    test_mpc_bg 1 16 100
    wait
}

# 2.2 MLP

test_mlp() {
    local gpu_id=$1
    local c1=$2
    local c2=$3
    local c3=$4
    local mlp_last_size=$5
    CUDA_VISIBLE_DEVICES=$gpu_id python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 5000 --mini-epochs 1 --noise-level 0. --reward-shaping ${c1},${c2},${c3} --no-obs-normalization --mlp-size-last $mlp_last_size --batch-test --exp-name reproduce_mlp_${mlp_last_size} --lr-schedule linear --initial-lr "5e-4" --quiet
}

# 2.3 QP

test_qp() {
    local gpu_id=$1
    local c1=$2
    local c2=$3
    local c3=$4
    local n_qp=$5
    local m_qp=$6
    CUDA_VISIBLE_DEVICES=$gpu_id python ../../run.py test tank --num-parallel 10000 --horizon 20 --epochs 5000 --mini-epochs 1 --noise-level 0. --reward-shaping ${c1},${c2},${c3} --n-qp $n_qp --m-qp $m_qp --qp-unrolled --shared-PH --affine-qb --strict-affine-layer --obs-has-half-ref --use-residual-loss --no-obs-normalization --force-feasible --batch-test --exp-name reproduce_qp_${n_qp}_${m_qp} --lr-schedule linear --initial-lr "5e-4" --quiet
}

# Utility function for train and test
train_and_test() {
    local train_function="$1"
    shift
    local test_function="$1"
    shift

    train_pid=$(find_gpu_and_run_task $train_function $@)
    while [ -e /proc/$train_pid ]; do
        sleep 1
    done
    test_pid=$(find_gpu_and_run_task $test_function $@)
    while [ -e /proc/$test_pid ]; do
        sleep 1
    done
}

run_and_delay() {
    local run_function="$1"
    shift

    $run_function $@ &
    local run_pid=$!
    sleep 10
    echo $run_pid
}

# Finally run all the tasks

run_and_delay test_mpc_all
run_and_delay train_and_test train_mlp test_mlp 50 0.05 2 8
run_and_delay train_and_test train_mlp test_mlp 50 0.05 2 16
run_and_delay train_and_test train_mlp test_mlp 50 0.05 2 32
run_and_delay train_and_test train_mlp test_mlp 50 0.05 2 64
run_and_delay train_and_test train_qp test_qp 50 0.05 2 4 24
run_and_delay train_and_test train_qp test_qp 50 0.05 2 8 48
run_and_delay train_and_test train_qp test_qp 50 0.05 2 16 96

wait

python reproduce_table.py
