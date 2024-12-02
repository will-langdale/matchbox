import logging
from pathlib import Path

import click
import tomli
from dotenv import find_dotenv, load_dotenv

from matchbox.common.db import SourceWarehouse
from matchbox.server.base import (
    MatchboxSettings,
    Source,
)

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


def load_datasets_from_config(datasets: Path) -> dict[str, Source]:
    """Loads datasets for indexing from the datasets settings TOML file."""
    config = tomli.load(datasets)

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
def make_cmf(datasets: Path) -> None:
    backend = MatchboxSettings().backend
    dataset_dict = load_datasets_from_config(datasets=datasets)

    for dataset in dataset_dict.values():
        backend.index(dataset=dataset, engine=dataset.database.engine)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    make_cmf()
