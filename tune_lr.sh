#!/bin/bash

# Ensure compatibility with zsh
setopt shwordsplit 2>/dev/null || true

# Default wandb group
WANDB_GROUP="default-group"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --wandb-group) WANDB_GROUP="$2"; shift ;;
        *) echo "Unknown parameter: $1" ;;
    esac
    shift
done

# Define directories
RESULTS_DIR="results"
PIDS_DIR="./pids"

# Create the necessary directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$PIDS_DIR"

# Get a timestamp for this batch of experiments
start_ts=$(date +"%Y-%m-%d %H:%M:%S.%N")
run_num=0

# Infer GPU count using nvidia-smi
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $gpu_count GPUs available for use."

# Loop through learning rates from 10^-6 to 10^-1 in steps at each exponent
for lr_exp in {-7..-1}; do
    initial_lr=$(echo "10^$lr_exp" | bc -l)
    
    # Determine GPU for this run
    gpu_id=$((run_num % gpu_count))

    # Create a unique directory for each run based on timestamp and learning rate
    run_dir="${RESULTS_DIR}/${start_ts}/lr_${initial_lr}"
    mkdir -p "$run_dir"

    # Start each run with the specified GPU and configuration, log output to the directory
    boltz run densenet --gpu "$gpu_id" --wandb-group "$WANDB_GROUP" --only-train baselines --no-same-model-init --wandb-legend-params initial_lr --initial-lr "$initial_lr" > "${run_dir}/metrics.log" 2>&1 &
    
    # Save the PID for later tracking
    echo $! > "${PIDS_DIR}/run${run_num}.pid"
    echo "Started run ${run_num} with GPU ${gpu_id} (initial_lr: ${initial_lr}) with PID $(cat ${PIDS_DIR}/run${run_num}.pid)"
    
    # Increment run number for the next combination
    run_num=$((run_num + 1))
done

echo "All processes started."
