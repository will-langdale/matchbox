from sqlalchemy.orm import Session

from cmf.data import SourceDataset


def test_database(db_engine):
    with Session(db_engine[1]) as session:
        session.add(SourceDataset(db_schema="z", db_table="y", db_id="x"))
        session.commit()

        data = session.query(SourceDataset).filter_by(db_schema="z").first()

        session.close()

        assert data.db_table == "y"
