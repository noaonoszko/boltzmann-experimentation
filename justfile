# Run all experiments by calling the shell script
run-all:
    ./run_experiments.sh
#
# Run experiments with varying learning rates from 10^-6 to 10^-1
tune-lr:
    ./tune_lr.sh
