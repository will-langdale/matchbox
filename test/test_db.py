import os

from dotenv import find_dotenv, load_dotenv
from sqlalchemy.orm import Session

from cmf.admin import add_dataset
from cmf.data import SourceDataset

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def test_database(db_engine):
    with Session(db_engine[1]) as session:
        session.add(SourceDataset(db_schema="z", db_table="y", db_id="x"))
        session.commit()

        data = session.query(SourceDataset).filter_by(db_schema="z").first()

        session.close()

        assert data.db_table == "y"


def test_add_data(db_engine):
    datasets = {
        "crn_table": {
            "schema": os.getenv("SCHEMA"),
            "table": "crn",
            "id": "id",
        }
    }
    for dataset in datasets.values():
        add_dataset(dataset, db_engine[1])

        with Session(db_engine[1]) as session:
            data = (
                session.query(SourceDataset)
                .filter_by(db_schema=dataset["schema"], db_table=dataset["table"])
                .first()
            )

            session.close()

            assert data.db_table == dataset["table"]
