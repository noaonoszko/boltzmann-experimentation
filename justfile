# Run all experiments by calling the shell script with a specified wandb group
run-all wandb_group:
    ./run_experiments.sh --wandb-group {{wandb_group}}

# Run experiments with varying learning rates with a specified wandb group
tune-lr wandb_group:
    ./tune_lr.sh --wandb-group {{wandb_group}}
