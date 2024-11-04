import cyclopts
import torch
from tqdm import tqdm
from tqdm.auto import trange

import wandb
from boltzmann_experimentation.config.literals import GPU, ONLY_TRAIN
from boltzmann_experimentation.utils.logger import (
    add_file_logger,
    general_logger,
    init_wandb_run,
    metrics_logger,
)
from boltzmann_experimentation.training.loss import (
    ExactLoss,
)
from boltzmann_experimentation.training.training_loop import TrainingLoop
from boltzmann_experimentation.factories import TrainingComponentsFactory
from boltzmann_experimentation.data.loaders import infinite_data_loader_generator
from boltzmann_experimentation.training.miner import Miner
from boltzmann_experimentation.factories import ModelFactory, DatasetFactory
from boltzmann_experimentation.config.literals import MODEL_TYPE
from boltzmann_experimentation.config.settings import general_settings as g, start_ts
from boltzmann_experimentation.training.validator import Validator
import ast

app = cyclopts.App()


@app.command
def run(
    model_type: MODEL_TYPE,
    num_miners: int = 5,
    num_communication_rounds: int = 3000,
    batch_size_train: int = 128,
    batch_size_val: int = 512,
    gpu: GPU | None = None,
    only_train: ONLY_TRAIN | None = None,
    log_to_wandb: bool = True,
    same_model_init: bool | None = None,
    compression_factors: list[int] = [1, 10, 100, 1000],
    model_kwargs: str | None = None,
    agg_bn_params: bool = True,
):
    # Change pydantic settings
    parsed_model_kwargs = ast.literal_eval(model_kwargs) if model_kwargs else {}
    g.model_kwargs |= parsed_model_kwargs
    g.num_miners = num_miners if num_miners else g.num_miners
    g.num_communication_rounds = (
        num_communication_rounds
        if num_communication_rounds
        else g.num_communication_rounds
    )
    g.batch_size_train = batch_size_train
    g.batch_size_val = batch_size_val
    g.set_device(gpu)
    g.log_to_wandb = log_to_wandb
    g.agg_bn_params = agg_bn_params
    same_model_init_values = (
        [True, False] if same_model_init is None else [same_model_init]
    )

    general_logger.info(
        f"Starting experiment on device {g.device} with {same_model_init_values=} and {compression_factors=}"
    )
    # Use TrainingComponentsFactory to create components based on the initialized torch_model
    t = TrainingComponentsFactory.create_components(model_type)

    SEED = 42

    # Generate the appropriate dataset for the model type
    train_dataset, val_dataset = DatasetFactory.create_dataset(model_type)
    general_logger.success(
        f"Created train dataset of length {len(train_dataset)} and val dataset of length {len(val_dataset)} for model {model_type}"
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
                run_name = f"Central training: {'Same Init' if same_model_init else 'Diff Init'}"
                init_wandb_run(
                    run_name=run_name, model_type=model_type, training_components=t
                )
            val_batch = next(infinite_val_loader)
            model.val_step(val_batch)
            for _ in trange(g.num_communication_rounds):
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
                run_name = f"{'Same Init' if same_model_init else 'Diff Init'}: Compression {compression_factor}"
                init_wandb_run(
                    run_name=run_name, model_type=model_type, training_components=t
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
