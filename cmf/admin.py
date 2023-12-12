import yaml
from pathlib import Path

from cmf import locations as loc
from cmf.data import CMFBase, ENGINE, SourceDataset, SourceData
from cmf.data import utils as du

from sqlalchemy import Engine, Session, func, select, String
from sqlalchemy.dialects.postgresql import insert

from typing import Dict


def init_db(base, engine: Engine = ENGINE):
    base.metadata.create_all(engine)


def add_dataset(dataset: Dict[str, str], engine: Engine = ENGINE) -> None:
    with Session(engine) as session:
        ##########################
        # Insert dataset section #
        ##########################

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

        get_uuid_stmt = select(SourceDataset).where(
            SourceDataset.db_schema == dataset["schema"],
            SourceDataset.db_table == dataset["table"],
        )

        new_dataset = session.execute(get_uuid_stmt).scalar()

        session.commit()

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

        to_insert = [
            dict(data._mapping, **{"dataset": new_dataset.uuid})
            for data in raw_result.all()
        ]

        # Insert it
        ins_stmt = insert(SourceData)
        ins_stmt = ins_stmt.on_conflict_do_update(
            index_elements=[SourceData.sha1], set_=ins_stmt.excluded
        )
        session.execute(ins_stmt, to_insert)

        session.commit()


def update_db_with_datasets(engine: Engine = ENGINE) -> None:
    with open(Path(loc.CMF, "datasets.yaml"), "rb") as f:
        datasets = yaml.load(f, yaml.Loader)

    for dataset in datasets:
        add_dataset(dataset, engine)


if __name__ == "__main__":
    init_db(CMFBase, ENGINE)
    update_db_with_datasets(ENGINE)
