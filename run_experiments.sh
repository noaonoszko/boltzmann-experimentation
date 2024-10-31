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

# Loop through all combinations
for same_model in "--same-model-init" "--no-same-model-init"; do
    for compression in 1 10 100 1000; do
        # Create a unique directory for each run based on timestamp, `same_model`, and `compression`
        model_init_status=$( [[ $same_model == "--same-model-init" ]] && echo "True" || echo "False" )
        run_dir="${RESULTS_DIR}/${start_ts}/logs/${model_init_status}:${compression}"
        
        # Ensure the directory exists
        mkdir -p "$run_dir"

        # Start each run with the specified GPU and configuration, log output to the directory
        be run custom-densenet --num-communication-rounds 3000 --gpu "$run_num" --only-train miners $same_model --compression-factors $compression --model-kwargs '{"batch_norm": False}' > "${run_dir}/metrics.log" 2>&1 &
        
        # Save the PID for later tracking
        echo $! > "${PIDS_DIR}/run${run_num}.pid"
        echo "Started run ${run_num} with GPU ${run_num} (${same_model}, compression: ${compression}) with PID $(cat ${PIDS_DIR}/run${run_num}.pid)"
        
        # Increment run number for the next combination
        run_num=$((run_num + 1))
    done
done

echo "All processes started."
