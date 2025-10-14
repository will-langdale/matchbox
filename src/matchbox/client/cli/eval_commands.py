"""CLI commands for entity evaluation."""

import logging
from typing import Annotated

import typer

from matchbox.client import _handler
from matchbox.client.cli.eval.app import EntityResolutionApp
from matchbox.common.dtos import CollectionName, ModelResolutionPath

eval_app = typer.Typer(help="Entity evaluation commands")


@eval_app.command()
def start(
    collection: Annotated[
        str, typer.Option("--collection", "-c", help="Collection name (required)")
    ],
    resolution: Annotated[
        str | None,
        typer.Option(
            "--resolution",
            "-r",
            help="Resolution name (defaults to collection's final_step)",
        ),
    ] = None,
    samples: Annotated[
        int,
        typer.Option("--samples", "-n", help="Number of entity clusters to evaluate"),
    ] = 100,
    user: Annotated[
        str | None,
        typer.Option(
            "--user", "-u", help="Username for authentication (overrides settings)"
        ),
    ] = None,
    warehouse: Annotated[
        str | None,
        typer.Option(
            "--warehouse", "-w", help="Warehouse database URL (overrides settings)"
        ),
    ] = None,
    log_file: Annotated[
        str | None,
        typer.Option(
            "--log",
            help="Log file path to redirect all logging output (keeps UI clean)",
        ),
    ] = None,
) -> None:
    """Start the interactive entity resolution evaluation tool."""
    try:
        # Set up logging redirect if --log specified
        if log_file:
            _setup_logging_redirect(log_file)

        # Fetch collection and construct ModelResolutionPath
        collection_name = CollectionName(collection)
        collection_obj = _handler.get_collection(collection_name)
        run_id = collection_obj.default_run

        # Get resolution name from --resolution or use DAG's final_step
        if resolution is None:
            # Load DAG to get final_step
            from matchbox.client.dags import DAG
            from matchbox.client.sources import Location

            dag = DAG(name=collection)
            dag.load_default(location=Location())
            resolution_name = dag.final_step.name
        else:
            resolution_name = resolution

        # Construct the ModelResolutionPath
        resolution_path = ModelResolutionPath(
            collection=collection_name,
            run=run_id,
            name=resolution_name,
        )

        app = EntityResolutionApp(
            resolution=resolution_path,
            num_samples=samples,
            user=user,
            warehouse=warehouse,
        )
        app.run()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully - exit with success code
        pass


def _setup_logging_redirect(log_file_path: str) -> None:
    """Redirect all logging output to file to keep Textual UI clean."""
    # Get the root logger to capture ALL logging (including third-party)
    root_logger = logging.getLogger()

    # Remove any existing handlers (console handlers that print to screen)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up file handler for all logs
    file_handler = logging.FileHandler(log_file_path, mode="w")

    # Use detailed format since these are going to file, not screen
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)

    # Add file handler to root logger
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)  # Capture INFO and above

    # Also suppress stdout/stderr from other libraries that might bypass logging
    # This keeps the Textual UI completely clean
