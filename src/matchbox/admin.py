import logging
from pathlib import Path
from typing import Dict

import click
import tomli
from sqlalchemy import Engine, String, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from matchbox import locations as loc
from matchbox.data import ENGINE, CMFBase, SourceData, SourceDataset
from matchbox.data import utils as du


def init_db(base, engine: Engine = ENGINE):
    base.metadata.create_all(engine)


def add_dataset(dataset: Dict[str, str], engine: Engine = ENGINE) -> None:
    logic_logger = logging.getLogger("cmf_logic")
    db_logger = logging.getLogger("sqlalchemy.engine")
    db_logger.setLevel(logging.WARNING)

    with Session(engine) as session:
        ##########################
        # Insert dataset section #
        ##########################

        logic_logger.info(f"Adding {dataset['schema']}.{dataset['table']}")

        to_insert = [
            {
                "db_schema": dataset["schema"],
                "db_table": dataset["table"],
                "db_id": dataset["id"],
            }
        ]

        ins_stmt = insert(SourceDataset)
        ins_stmt = ins_stmt.on_conflict_do_nothing(
            index_elements=[
                SourceDataset.db_schema,
                SourceDataset.db_table,
            ]
        )

        session.execute(ins_stmt, to_insert)

        new_dataset = (
            session.query(SourceDataset)
            .filter_by(db_schema=dataset["schema"], db_table=dataset["table"])
            .first()
        )

        session.commit()

        logic_logger.info(
            f"{dataset['schema']}.{dataset['table']} added to SourceDataset"
        )

        #######################
        # Insert data section #
        #######################

        source_table = du.dataset_to_table(new_dataset, engine)

        # Retrieve the SHA1 of data and an array of row IDs (as strings)
        # Array because we can't guarantee non-duplicated data
        cols = tuple(
            [col for col in list(source_table.c.keys()) if col != dataset["id"]]
        )
        slct_stmt = select(
            func.digest(func.concat(*source_table.c[cols]), "sha1").label("sha1"),
            func.array_agg(source_table.c[dataset["id"]].cast(String)).label("id"),
        ).group_by(*source_table.c[cols])

        raw_result = session.execute(slct_stmt)

        logic_logger.info(
            f"Retrieved raw data from {dataset['schema']}.{dataset['table']}"
        )

        # Create list of (sha1, id, dataset)-keyed dicts using RowMapping:
        # https://docs.sqlalchemy.org/en/20/core/
        # connections.html#sqlalchemy.engine.Row._mapping
        to_insert = [
            dict(data._mapping, **{"dataset": new_dataset.uuid})
            for data in raw_result.all()
        ]

        # Insert it using PostgreSQL upsert
        # https://docs.sqlalchemy.org/en/20/dialects/
        # postgresql.html#insert-on-conflict-upsert
        ins_stmt = insert(SourceData)
        ins_stmt = ins_stmt.on_conflict_do_update(
            index_elements=[SourceData.sha1, SourceData.dataset], set_=ins_stmt.excluded
        )
        session.execute(ins_stmt, to_insert)

        session.commit()

        logic_logger.info(
            f"Inserted raw data from {dataset['schema']}.{dataset['table']}"
        )

        logic_logger.info(f"Finished {dataset['schema']}.{dataset['table']}")


def update_db_with_datasets(engine: Engine = ENGINE) -> None:
    with open(Path(loc.CMF, "datasets.toml"), "rb") as f:
        datasets = tomli.load(f)["datasets"]

    for dataset in datasets.values():
        add_dataset(dataset, engine)


@click.command()
def make_cmf() -> None:
    init_db(CMFBase, ENGINE)
    update_db_with_datasets(ENGINE)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    make_cmf()
