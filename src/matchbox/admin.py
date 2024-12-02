import logging
from pathlib import Path
from datetime import datetime

import click
import tomli
from dotenv import find_dotenv, load_dotenv

from matchbox.server import MatchboxDBAdapter, inject_backend
from matchbox.server.base import (
    Source,
)
from matchbox.server.models import SourceWarehouse

logger = logging.getLogger("admin_pipeline")

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


@inject_backend
def index_dataset(backend: MatchboxDBAdapter, dataset: Source) -> None:
    backend.index(dataset=dataset, engine=dataset.database.engine)


def load_datasets_from_config(datasets: Path) -> dict[str, Source]:
    """Loads datasets for indexing from the datasets settings TOML file."""
    config = tomli.loads(datasets.read_text())

    warehouses: dict[str, SourceWarehouse] = {}
    for alias, warehouse_config in config["warehouses"].items():
        warehouses[alias] = SourceWarehouse(alias=alias, **warehouse_config)

    datasets: dict[str, Source] = {}
    for dataset_name, dataset_config in config["datasets"].items():
        warehouse_alias = dataset_config.get("database")
        dataset_config["database"] = warehouses[warehouse_alias]
        datasets[dataset_name] = Source(**dataset_config)

    return datasets


@click.command()
@click.argument(
    "datasets", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@inject_backend
def make_matchbox(backend: MatchboxDBAdapter, datasets: Path) -> None:
    
    for dataset in datasets:
        logger.info(f"Indexing {dataset}")
        index_dataset(Source(dataset))
        logger.info(f"Finished indexing {dataset}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    make_matchbox()
