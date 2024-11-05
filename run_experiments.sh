#!/bin/bash

# Ensure compatibility with zsh
setopt shwordsplit 2>/dev/null || true

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

# Loop through all combinations
for norm in "" "batch" "group"; do
    for compression in 1 10 100 1000; do
        # Determine GPU for this run
        gpu_id=$((run_num % gpu_count))

        # Create a unique directory for each run based on timestamp, `norm`, and `compression`
        run_dir="${RESULTS_DIR}/${start_ts}/logs/norm=${norm}:${compression}"
        
        # Ensure the directory exists
        mkdir -p "$run_dir"

        # Start each run with the specified GPU and configuration, log output to the directory
        boltz run densenet --gpu "$gpu_id" --only-train miners --wandb-legend-params norm --model-kwargs "{\"norm\": \"$norm\"}" --compression-factors $compression > "${run_dir}/metrics.log" 2>&1 &
        
        # Save the PID for later tracking
        echo $! > "${PIDS_DIR}/run${run_num}.pid"
        echo "Started run ${run_num} with GPU ${gpu_id} (${norm}, compression: ${compression}) with PID $(cat ${PIDS_DIR}/run${run_num}.pid)"
        
        # Increment run number for the next combination
        run_num=$((run_num + 1))
    done
done

echo "All processes started."
