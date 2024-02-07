import contextlib
import cProfile
import hashlib
import io
import pstats
import uuid
from typing import List, TypeVar, Union

from pandas import DataFrame, Series
from sqlalchemy import Engine, MetaData, Table, select
from sqlalchemy.orm import Session

from cmf.data import ENGINE, SourceDataset
from cmf.data.models import Models

T = TypeVar("T")

# Data conversion


def get_schema_table_names(full_name: str, validate: bool = False) -> (str, str):
    """
    Takes a string table name and returns the unquoted schema and
    table as a tuple. If you insert these into a query, you need to
    add double quotes in from statements, or single quotes in where.

    Parameters:
        full_name: A string indicating a Postgres table
        validate: Whether to error if both schema and table aren't
        detected

    Raises:
        ValueError: When the function can't detect either a
        schema.table or table format in the input
        ValidationError: If both schema and table can't be detected
        when the validate argument is True

    Returns:
        (schema, table): A tuple of schema and table name. If schema
        cannot be inferred, returns None.
    """

    schema_name_list = full_name.replace('"', "").split(".")

    if len(schema_name_list) == 1:
        schema = None
        table = schema_name_list[0]
    elif len(schema_name_list) == 2:
        schema = schema_name_list[0]
        table = schema_name_list[1]
    else:
        raise ValueError(
            f"""
            Could not identify schema and table in {full_name}.
        """
        )

    if validate and schema is None:
        raise ("Schema could not be detected and validation required.")

    return (schema, table)


def dataset_to_table(dataset: SourceDataset, engine: Engine = ENGINE) -> Table:
    """Takes a CMF SourceDataset object and returns a SQLAlchemy Table."""
    with Session(engine) as session:
        source_schema = MetaData(schema=dataset.db_schema)
        source_table = Table(
            dataset.db_table,
            source_schema,
            schema=dataset.db_schema,
            autoload_with=session.get_bind(),
        )
    return source_table


def string_to_table(db_schema: str, db_table: str, engine: Engine = ENGINE) -> Table:
    """Takes strings and returns a SQLAlchemy Table."""
    with Session(engine) as session:
        source_schema = MetaData(schema=db_schema)
        source_table = Table(
            db_table,
            source_schema,
            schema=db_schema,
            autoload_with=session.get_bind(),
        )
    return source_table


def string_to_dataset(
    db_schema: str, db_table: str, engine: Engine = ENGINE
) -> SourceDataset:
    """Takes a CMF SourceDataset object and returns a CMF SourceDataset"""
    with Session(engine) as session:
        dataset = (
            session.query(SourceDataset)
            .filter_by(db_schema=db_schema, db_table=db_table)
            .first()
        )
    return dataset


# SHA-1 hashing helper functions


def table_name_to_sha1(schema_table: str, engine: Engine = ENGINE) -> bytes:
    """Takes a table's full schema.table name and returns its SHA-1 hash."""
    db_schema, db_table = get_schema_table_names(schema_table)

    with Session(engine) as session:
        stmt = select(SourceDataset.uuid).where(
            SourceDataset.db_schema == db_schema, SourceDataset.db_table == db_table
        )
        dataset_sha1 = session.execute(stmt).scalar()

    return dataset_sha1


def model_name_to_sha1(run_name: str, engine: Engine = ENGINE) -> bytes:
    """Takes a model's name and returns its SHA-1 hash."""
    with Session(engine) as session:
        stmt = select(Models.sha1).where(Models.name == run_name)
        model_sha1 = session.execute(stmt).scalar()

    return model_sha1


def prep_for_hash(item: Union[bytes, bool, str, int, float, bytearray]) -> bytes:
    """Encodes strings so they can be hashed, otherwises, passes through."""
    if isinstance(item, bytes):
        return item
    elif isinstance(item, str):
        return bytes(item.encode())
    elif isinstance(item, uuid.UUID):
        return item.bytes
    else:
        return bytes(item)


def list_to_value_ordered_sha1(list_: List[T]) -> bytes:
    """Returns a single SHA1 hash of a list ordered by its values.

    List must be sorted as the different orders of value must produce the same hash.
    """
    sorted_vals = sorted(list_)
    hashed_vals_list = [hashlib.sha1(prep_for_hash(i)) for i in sorted_vals]

    hashed_vals = hashed_vals_list[0]
    for val in hashed_vals_list[1:]:
        hashed_vals.update(val.digest())

    return hashed_vals.digest()


def columns_to_value_ordered_sha1(data: DataFrame, columns: List[str]) -> Series:
    """Returns the SHA1 hash of columns ordered by their values."""
    try:
        # Deals with byte arrays from duckdb's .df()
        bytes_records = data.filter(columns).map(bytes).to_dict("records")
    except TypeError:
        # Deals with objects found in normal dataframes for offline joins
        bytes_records = data.filter(columns).astype(bytes).to_dict("records")

    hashed_records = []

    for record in bytes_records:
        hashed_vals = list_to_value_ordered_sha1(record.values())
        hashed_records.append(hashed_vals)

    return Series(hashed_records)


# SQLAlchemy profiling


@contextlib.contextmanager
def sqa_profiled():
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    # uncomment this to see who's calling what
    # ps.print_callers()
    print(s.getvalue())
