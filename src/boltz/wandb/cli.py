"""CLI module for managing and modifying Weights & Biases (W&B) runs."""

from datetime import datetime, timedelta
from texp.utils.logger import general_logger as l
from cyclopts import App

import wandb
import ast
from wandb.apis.public.runs import Run, Runs

app = App()

app.command(group := App(name="group"))
api = wandb.Api()
WANDB_PATH = "noa-onoszko-reflog-ab/chakana"


@group.command
def rename(
    new: str,
    old: str | None = None,
    between_hours_ago: list[int] | None = None,
    config_filters: str | None = None,
    dry_run: bool = True,
    verbose: bool = False,
) -> None:
    """Rename the group of W&B runs that match specified filters.

    This command changes the group name of W&B runs according to provided
    filters, such as a specific group name, time range, and configuration
    settings. Optionally, the operation can be previewed without making
    actual changes using `dry_run`.

    Example:
        `wb group rename lr-tuning-adam-higher-bs --between-hours-ago 7 100
        --verbose --config-filters '{"wandb_legend_params": ["initial_lr"]}'
        --no-dry-run`

        This example renames the group of W&B runs to `lr-tuning-adam-higher-bs`
        for runs created between 7 and 100 hours ago. It filters runs where
        the `wandb_legend_params` configuration parameter contains `initial_lr`.
        The `--verbose` flag provides detailed information about each run found,
        and the `--no-dry-run` flag applies the changes immediately rather than
        just previewing them.

    Parameters:
        new: New group name to assign to the runs.
        old: Current group name to filter runs by.
        between_hours_ago: Time range in hours
            to filter runs created within the last specified hours.
            Format: [start_hours, end_hours].
        config_filters: String representation of a dictionary
            for filtering runs by specific configuration parameters.
        dry_run: If True, only preview changes without
            saving them. Default is True.
        verbose: If True, outputs detailed information
            about each run found.
    """
    filters = ast.literal_eval(config_filters) if config_filters else {}
    filters = {f"config.{key}": value for key, value in filters.items()}
    if old is not None:
        filters |= {"group": old}
    if between_hours_ago:
        start = (datetime.now() - timedelta(hours=between_hours_ago[1] + 1)).isoformat()
        end = (datetime.now() - timedelta(hours=between_hours_ago[0] + 1)).isoformat()
        filters |= {"created_at": {"$gte": start, "$lt": end}}
        l.info(f"Filtering on runs created the last {between_hours_ago} hours")
    l.info(filters)
    runs_: Runs = api.runs(
        path=WANDB_PATH,
        filters=filters,
        order="+created_at",
    )
    l.info(f"Found {len(runs_)} runs{':' if verbose and runs_ else ''}")
    if verbose:
        for run in runs_:
            run: Run
            l.info(f"{run.name}")

    if not dry_run:
        for run in runs_:
            run: Run
            run.group = new
            run.save()
        if runs_:
            l.success("Renamed all of them")
