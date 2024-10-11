import logging

from sqlalchemy import Engine, String, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from matchbox.server.base import IndexableDataset
from matchbox.server.postgresql import utils as du
from matchbox.server.postgresql.data import SourceData, SourceDataset


def index_dataset(
    dataset: IndexableDataset, engine: Engine, warehouse_engine: Engine
) -> None:
    """Indexes a dataset from your data warehouse within Matchbox."""

    logic_logger = logging.getLogger("mb_logic")
    db_logger = logging.getLogger("sqlalchemy.engine")
    db_logger.setLevel(logging.WARNING)

    with Session(engine) as session:
        ##########################
        # Insert dataset section #
        ##########################

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

        session.commit()

        logic_logger.info(f"{dataset} added to SourceDataset")

        #######################
        # Insert data section #
        #######################

    with Session(warehouse_engine) as warehouse_session:
        source_table = du.dataset_to_table(new_dataset, warehouse_engine)

        # Retrieve the SHA1 of data and an array of row IDs (as strings)
        # Array because we can't guarantee non-duplicated data
        cols = tuple(
            [col for col in list(source_table.c.keys()) if col != dataset.db_pk]
        )
        slct_stmt = select(
            func.digest(func.concat(*source_table.c[cols]), "sha1").label("sha1"),
            func.array_agg(source_table.c[dataset.db_pk].cast(String)).label("id"),
        ).group_by(*source_table.c[cols])

        raw_result = warehouse_session.execute(slct_stmt)

        logic_logger.info(f"Retrieved raw data from {dataset}")

        # Create list of (sha1, id, dataset)-keyed dicts using RowMapping:
        # https://docs.sqlalchemy.org/en/20/core/
        # connections.html#sqlalchemy.engine.Row._mapping
        to_insert = [
            dict(data._mapping, **{"dataset": new_dataset.uuid})
            for data in raw_result.all()
        ]

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
