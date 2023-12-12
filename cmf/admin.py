import logging
from pathlib import Path
from typing import Dict

import click
import yaml
from sqlalchemy import Engine, String, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from cmf import locations as loc
from cmf.data import ENGINE, CMFBase, SourceData, SourceDataset
from cmf.data import utils as du


def init_db(base, engine: Engine = ENGINE):
    base.metadata.create_all(engine)


def add_dataset(dataset: Dict[str, str], engine: Engine = ENGINE) -> None:
    logger = logging.getLogger(__name__)
    with Session(engine) as session:
        ##########################
        # Insert dataset section #
        ##########################

        logger.info(f"Adding {dataset['schema']}.{dataset['table']}")

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

        logger.info(f"{dataset['schema']}.{dataset['table']} added to SourceDataset")

        #######################
        # Insert data section #
        #######################

        source_table = du.dataset_to_table(new_dataset, engine)

        # Retrieve the SHA1 and ID of each row
        cols = tuple(
            [col for col in list(source_table.c.keys()) if col != dataset["id"]]
        )
        slct_stmt = select(
            func.digest(func.concat(*source_table.c[cols]), "sha1").label("sha1"),
            func.array_agg(source_table.c[dataset["id"]].cast(String)).label("id"),
        ).group_by(*source_table.c[cols])

        raw_result = session.execute(slct_stmt)

        logger.info(f"Retrieved raw data from {dataset['schema']}.{dataset['table']}")

        to_insert = [
            dict(data._mapping, **{"dataset": new_dataset.uuid})
            for data in raw_result.all()
        ]

        # Insert it
        ins_stmt = insert(SourceData)
        ins_stmt = ins_stmt.on_conflict_do_nothing(
            index_elements=[SourceData.id, SourceData.dataset]
        )
        session.execute(ins_stmt, to_insert)

        session.commit()

        logger.info(f"Inserted raw data from {dataset['schema']}.{dataset['table']}")

        logger.info(f"Finished {dataset['schema']}.{dataset['table']}")


def update_db_with_datasets(engine: Engine = ENGINE) -> None:
    with open(Path(loc.CMF, "datasets.yaml"), "rb") as f:
        datasets = yaml.load(f, yaml.Loader)

    for dataset in datasets.values():
        add_dataset(dataset, engine)


@click.command()
def make_cmf() -> None:
    init_db(CMFBase, ENGINE)
    update_db_with_datasets(ENGINE)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=du.LOG_FMT,
    )

    make_cmf()
