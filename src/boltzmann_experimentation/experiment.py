import cyclopts
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from tqdm.auto import trange

import wandb
from boltzmann_experimentation.dataset import (
    DatasetFactory,
    infinite_data_loader_generator,
)
from boltzmann_experimentation.config.literals import GPU, ONLY_TRAIN
from boltzmann_experimentation.logger import (
    add_file_logger,
    general_logger,
    init_wandb_run,
    metrics_logger,
)
from boltzmann_experimentation.loss import (
    ExactLoss,
)
from boltzmann_experimentation.miner import Miner
from boltzmann_experimentation.factories import ModelFactory
from boltzmann_experimentation.config.literals import MODEL_TYPE
from boltzmann_experimentation.config.settings import general_settings as g, start_ts
from boltzmann_experimentation.validator import Validator
from boltzmann_experimentation.viz import (
    InteractivePlotter,
)

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
):
    # Change pydantic settings
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
    same_model_init_values = (
        [True, False] if same_model_init is None else [same_model_init]
    )

    general_logger.info(
        f"Starting experiment on device {g.device} with {same_model_init_values=} and {compression_factors=}"
    )

    PLOT_INTERACTIVELY = False
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
            model = ModelFactory.create_model(model_type)
            general_logger.info(f"Model has {model.num_params()/1e6:.0f}M params.")
            if log_to_wandb:
                wandb.finish()
                run_name = f"Central training: {'Same Init' if same_model_init else 'Diff Init'}"
                init_wandb_run(run_name=run_name, model_type=model_type)
            val_batch = next(infinite_val_loader)
            model.val_step(val_batch)
            for _ in trange(g.num_communication_rounds):
                batch = next(infinite_train_loader)
                model.train_step(batch)
                val_batch = next(infinite_val_loader)
                model.val_step(val_batch)

    # Train baselines
    if only_train in (None, "baselines"):
        train_baselines()
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
                miner_models.append(ModelFactory.create_model(model_type))
            if same_model_init:
                torch.manual_seed(SEED)
            validator_model = ModelFactory.create_model(model_type)
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

            # Set up interactive logging
            interactive_plotter = None
            if PLOT_INTERACTIVELY:
                xlim = (
                    val_dataset.features.min().item(),
                    val_dataset.features.max().item(),
                )
                ylim = (
                    val_dataset.targets.min().item() - 2,
                    val_dataset.targets.max().item() + 2,
                )
                interactive_plotter = InteractivePlotter(xlim, ylim)

            if log_to_wandb:
                wandb.finish()
                run_name = f"{'Same Init' if same_model_init else 'Diff Init'}: Compression {compression_factor}"
                init_wandb_run(run_name=run_name, model_type=model_type)

            # Validate the initial model
            val_batch = next(infinite_val_loader)
            validator.model.val_step(val_batch)

            # Training loop
            for round_num in trange(g.num_communication_rounds):
                validator.reset_slices_and_indices()
                for miner in miners:
                    miner.data = next(infinite_train_loader)
                    miner.model.train_step(miner.data)
                    slice = miner.get_slice_from_indices(validator.slice_indices)
                    validator.add_miner_slice(slice)

                for miner in miners:
                    validator.calculate_miner_score(
                        validator.slices[miner.id],
                        miner.id,
                        miner.data,
                    )

                validator.model.torch_model = validator.model.aggregate_slices(
                    slices=list(validator.slices.values()),
                    slice_indices=validator.slice_indices,
                )
                val_batch = next(infinite_val_loader)
                validator.model.val_step(val_batch)

                for miner in miners:
                    validator_model_params = torch.nn.utils.parameters_to_vector(
                        validator.model.torch_model.parameters()
                    )
                    validator_slice = validator_model_params[validator.slice_indices]
                    miner.model.update_with_slice(
                        validator_slice, validator.slice_indices
                    )
                    if miner.model.lr_scheduler is not None:
                        miner.model.lr_scheduler.step()

                if PLOT_INTERACTIVELY and interactive_plotter is not None:
                    interactive_plotter.plot_data_and_model(
                        torch_model=validator.model.torch_model,
                        features=val_dataset.features,
                        targets=val_dataset.targets,
                    )

            if PLOT_INTERACTIVELY:
                # Disable interactive mode when done and keep the final plot displayed
                plt.ioff()
                plt.show()

            validator_val_losses_compression_factor.update(
                {f"{same_model_init}/{compression_factor}": validator.model.val_losses}
            )

            # plot_scores(validator.scores)


if __name__ == "__main__":
    app()
