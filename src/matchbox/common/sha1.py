import hashlib
import uuid
from typing import TypeVar

from pandas import DataFrame, Series

T = TypeVar("T")
HashableItem = TypeVar("HashableItem", bytes, bool, str, int, float, bytearray)


def prep_for_hash(item: HashableItem) -> bytes:
    """Encodes strings so they can be hashed, otherwises, passes through."""
    if isinstance(item, bytes):
        return item
    elif isinstance(item, str):
        return bytes(item.encode())
    elif isinstance(item, uuid.UUID):
        return item.bytes
    else:
        return bytes(item)


def list_to_value_ordered_sha1(list_: list[T]) -> bytes:
    """Returns a single SHA1 hash of a list ordered by its values.

    List must be sorted as the different orders of value must produce the same hash.
    """
    try:
        sorted_vals = sorted(list_)
    except TypeError as e:
        raise TypeError("Can only order lists or columns of the same datatype.") from e

    hashed_vals_list = [hashlib.sha1(prep_for_hash(i)) for i in sorted_vals]

    hashed_vals = hashed_vals_list[0]
    for val in hashed_vals_list[1:]:
        hashed_vals.update(val.digest())

    return hashed_vals.digest()


def columns_to_value_ordered_sha1(data: DataFrame, columns: list[str]) -> Series:
    """Returns the rowwise SHA1 hash ordered by the row's values, ignoring column order.

    This function is used to add a column to a dataframe that represents the SHA1
    hash of each its rows, but where the order of the row values doesn't change the
    hash value. Column order is ignored in favour of value order.

    This is primarily used to give a consistent hash to a new cluster no matter whether
    its parent hashes were used in the left or right table.
    """
    bytes_records = data.filter(columns).astype(bytes).to_dict("records")

    hashed_records = []

    for record in bytes_records:
        hashed_vals = list_to_value_ordered_sha1(record.values())
        hashed_records.append(hashed_vals)

    return Series(hashed_records)
