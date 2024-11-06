from typing import Annotated
from cyclopts import App, Group, Parameter, validators
import torch
from tqdm import tqdm
from tqdm.auto import trange

import wandb
from boltz.config.literals import GPU, ONLY_TRAIN, OPTIMIZER
from boltz.utils.logger import (
    add_file_logger,
    general_logger,
    init_wandb_run,
    metrics_logger,
)
from boltz.training.loss import (
    ExactLoss,
)
from boltz.training.training_loop import TrainingLoop
from boltz.factories import TrainingComponentsFactory
from boltz.data.loaders import infinite_data_loader_generator
from boltz.training.miner import Miner
from boltz.factories import ModelFactory, DatasetFactory
from boltz.config.literals import MODEL_TYPE
from boltz.config.settings import general_settings as g, start_ts
from boltz.training.validator import Validator
import ast

app = App()

training_duration = Group(
    "Measure of training duration (choose one)",
    default_parameter=Parameter(negative=""),  # Disable "--no-" flags
    validator=validators.LimitedChoice(min=0, max=1),  # Mutually Exclusive Options
)


@app.command
def run(
    wandb_group: str,
    model_type: MODEL_TYPE,
    num_miners: int = 5,
    num_comrounds: Annotated[int | None, Parameter(group=training_duration)] = None,
    num_epochs: Annotated[int, Parameter(group=training_duration)] = 300,
    batch_size_train: int = 128,
    batch_size_val: int = 512,
    gpu: GPU | None = None,
    only_train: ONLY_TRAIN | None = None,
    log_to_wandb: bool = True,
    same_model_init: bool | None = None,
    compression_factors: list[int] = [1, 10, 100, 1000],
    model_kwargs: str | None = None,
    agg_bn_params: bool = True,
    wandb_legend_params: list[str] | None = None,
    optimizer: OPTIMIZER | None = None,
    initial_lr: float | None = None,
):
    # Change pydantic settings
    parsed_model_kwargs = ast.literal_eval(model_kwargs) if model_kwargs else {}
    g.model_kwargs |= parsed_model_kwargs
    g.num_miners = num_miners if num_miners else g.num_miners
    g.num_epochs = num_epochs if num_epochs else g.num_epochs
    g.batch_size_train = batch_size_train
    g.batch_size_val = batch_size_val
    g.set_device(gpu)
    g.log_to_wandb = log_to_wandb
    g.agg_bn_params = agg_bn_params
    g.wandb_legend_params = (
        wandb_legend_params if wandb_legend_params else g.wandb_legend_params
    )
    g.optimizer = optimizer if optimizer else g.optimizer
    g.initial_lr = initial_lr if initial_lr else g.initial_lr
    same_model_init_values = (
        [True, False] if same_model_init is None else [same_model_init]
    )

    # Use TrainingComponentsFactory to create components based on the initialized torch_model
    t = TrainingComponentsFactory.create_components(model_type)

    SEED = 42

    # Generate the appropriate dataset for the model type
    train_dataset, val_dataset = DatasetFactory.create_dataset(model_type)
    general_logger.success(
        f"Created train dataset of length {len(train_dataset)} and val dataset of length {len(val_dataset)} for model {model_type}"
    )
    if num_comrounds is None:
        g.num_comrounds = int(
            g.num_epochs * len(train_dataset) / g.batch_size_train / g.num_miners
        )
    else:
        g.num_comrounds = num_comrounds
    general_logger.info(
        f"Starting experiment on device {g.device} with {same_model_init_values=} "
        f"and {compression_factors=} and {g.num_comrounds=}"
    )

    # Infinite iterator for training
    infinite_train_loader = infinite_data_loader_generator(train_dataset, train=True)
    infinite_val_loader = infinite_data_loader_generator(val_dataset, train=False)
    validator_val_losses_compression_factor = {}

    def train_baselines() -> None:
        for same_model_init in tqdm(same_model_init_values):
            if same_model_init:
                torch.manual_seed(SEED)
            model = ModelFactory.create_model(t)
            general_logger.info(f"Model has {model.num_params()/1e6:.0f}M params.")
            if log_to_wandb:
                wandb.finish()
                run_name = "Central"
                init_wandb_run(
                    run_name=run_name,
                    group=wandb_group,
                    model_type=model_type,
                    training_components=t,
                )
            val_batch = next(infinite_val_loader)
            model.val_step(val_batch)
            for _ in trange(g.num_comrounds):
                batch = next(infinite_train_loader)
                model.torch_model.train()
                model.train_step(batch)
                val_batch = next(infinite_val_loader)
                model.torch_model.eval()
                model.val_step(val_batch)
                if model.lr_scheduler is not None:
                    model.lr_scheduler.step()

    # Train baselines
    if only_train in (None, "baselines"):
        g.batch_size_train *= g.num_miners
        t.initial_lr *= g.num_miners
        train_baselines()
        g.batch_size_train //= g.num_miners
        t.initial_lr //= g.num_miners
        general_logger.success("Trained baselines")
    if only_train == "baselines":
        return
    else:
        general_logger.info("Training only miners")

    metrics_logger_id = None
    for same_model_init in tqdm(same_model_init_values):
        for compression_idx, compression_factor in enumerate(tqdm(compression_factors)):
            log_dir = (
                g.results_dir
                / f"{start_ts}/logs/{same_model_init}:{compression_factor}"
            )
            if metrics_logger_id is not None:
                metrics_logger.remove(metrics_logger_id)
            add_file_logger(log_dir)

            general_logger.info(
                f"Running experiment with compression factor {compression_factor}"
            )
            g.compression_factor = compression_factor
            # Create torch models
            miner_models = []
            for _ in range(g.num_miners):
                if same_model_init:
                    torch.manual_seed(SEED)
                miner_models.append(ModelFactory.create_model(t))
            if same_model_init:
                torch.manual_seed(SEED)
            validator_model = ModelFactory.create_model(t)
            validator_model.torch_model.eval()
            general_logger.success(
                f"Created {len(miner_models)} miner models and a validator model"
            )

            # Create miners
            miners = [
                Miner(
                    model,
                    i,
                )
                for i, model in enumerate(miner_models)
            ]
            general_logger.success(f"Created {len(miners)} miners")

            # Choose the loss method dynamically using dependency injection
            loss_method = ExactLoss()
            general_logger.success(f"Created loss of type {type(loss_method)}")

            # Create validator
            validator = Validator(
                validator_model,
                g.compression_factor,
                loss_method,
            )
            general_logger.success("Created validator")

            if log_to_wandb:
                wandb.finish()
                run_name = f"{compression_factor}x"
                init_wandb_run(
                    run_name=run_name,
                    group=wandb_group,
                    model_type=model_type,
                    training_components=t,
                )

            # Validate the initial model
            val_batch = next(infinite_val_loader)
            validator.model.val_step(val_batch)

            # Training loop
            TrainingLoop(
                validator=validator,
                miners=miners,
                infinite_train_loader=infinite_train_loader,
                infinite_val_loader=infinite_val_loader,
            ).run()

            validator_val_losses_compression_factor.update(
                {f"{same_model_init}/{compression_factor}": validator.model.val_losses}
            )


if __name__ == "__main__":
    app()
