import hashlib
import uuid
from typing import List, TypeVar, Union

from pandas import DataFrame, Series
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from cmf.data import ENGINE, SourceDataset
from cmf.data.models import Models
from cmf.data.utils.db import get_schema_table_names

T = TypeVar("T")


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
    try:
        sorted_vals = sorted(list_)
    except Exception as e:
        raise TypeError("Can only order lists or columns of the same datatype.") from e

    hashed_vals_list = [hashlib.sha1(prep_for_hash(i)) for i in sorted_vals]

    hashed_vals = hashed_vals_list[0]
    for val in hashed_vals_list[1:]:
        hashed_vals.update(val.digest())

    return hashed_vals.digest()


def columns_to_value_ordered_sha1(data: DataFrame, columns: List[str]) -> Series:
    """Returns the rowwise SHA1 hash of columns ordered by the row's values.

    Used to add a column to a dataframe that represents the SHA1 hash of each its
    rows, but where the order of the row values doesn't change the hash value.
    """
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
