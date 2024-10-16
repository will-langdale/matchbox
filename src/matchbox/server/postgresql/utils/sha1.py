from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from matchbox.common.exceptions import MatchboxDBDataError
from matchbox.helpers.selector import get_schema_table_names
from matchbox.server.postgresql.data import SourceDataset
from matchbox.server.postgresql.models import Models


def table_name_to_uuid(schema_table: str, engine: Engine) -> bytes:
    """Takes a table's full schema.table name and returns its UUID.

    Args:
        schema_table (str): The string name of the table in the form schema.table
        engine (sqlalchemy.Engine): The CMF connection engine

    Raises:
        CMFSourceError if table not found in database

    Returns:
        The UUID of the dataset
    """
    db_schema, db_table = get_schema_table_names(schema_table)

    with Session(engine) as session:
        stmt = select(SourceDataset.uuid).where(
            SourceDataset.db_schema == db_schema, SourceDataset.db_table == db_table
        )
        dataset_uuid = session.execute(stmt).scalar()

    if dataset_uuid is None:
        raise MatchboxDBDataError(table=SourceDataset.__tablename__, data=schema_table)

    return dataset_uuid


def model_name_to_sha1(run_name: str, engine: Engine) -> bytes:
    """Takes a model's name and returns its SHA-1 hash.

    Args:
        run_name (str): The string name of the model in the database
        engine (sqlalchemy.Engine): The CMF connection engine

    Raises:
        CMFSourceError if model not found in database

    Returns:
        The SHA-1 hash of the model
    """
    with Session(engine) as session:
        stmt = select(Models.sha1).where(Models.name == run_name)
        model_sha1 = session.execute(stmt).scalar()

    if model_sha1 is None:
        raise MatchboxDBDataError(table=Models.__tablename__, data=run_name)

    return model_sha1
