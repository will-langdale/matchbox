import logging

from sqlalchemy import Engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from matchbox.common.hash import dataset_to_hashlist
from matchbox.server.base import IndexableDataset
from matchbox.server.postgresql.data import SourceData, SourceDataset


def index_dataset(dataset: IndexableDataset, engine: Engine) -> None:
    """Indexes a dataset from your data warehouse within Matchbox."""

    logic_logger = logging.getLogger("mb_logic")
    db_logger = logging.getLogger("sqlalchemy.engine")
    db_logger.setLevel(logging.WARNING)

    ##################
    # Insert dataset #
    ##################

    with Session(engine) as session:
        logic_logger.info(f"Adding {dataset}")

        to_insert = [
            {
                "db_schema": dataset.db_schema,
                "db_table": dataset.db_table,
                "db_id": dataset.db_pk,
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
            .filter_by(db_schema=dataset.db_schema, db_table=dataset.db_table)
            .first()
        )

        new_dataset_uuid = new_dataset.uuid

        session.commit()

        logic_logger.info(f"{dataset} added to SourceDataset")

    ###############
    # Insert data #
    ###############

    to_insert = dataset_to_hashlist(dataset=dataset, uuid=new_dataset_uuid)

    with Session(engine) as session:
        # Insert it using PostgreSQL upsert
        # https://docs.sqlalchemy.org/en/20/dialects/
        # postgresql.html#insert-on-conflict-upsert
        ins_stmt = insert(SourceData)
        ins_stmt = ins_stmt.on_conflict_do_update(
            index_elements=[SourceData.sha1, SourceData.dataset], set_=ins_stmt.excluded
        )
        session.execute(ins_stmt, to_insert)

        session.commit()

        logic_logger.info(f"Inserted raw data from {dataset}")

        logic_logger.info(f"Finished {dataset}")
