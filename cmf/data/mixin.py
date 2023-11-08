from __future__ import annotations

from typing import List, TYPE_CHECKING
from pydantic import BaseModel, model_validator
from abc import ABC


if TYPE_CHECKING:
    from cmf.data.db import DB
    from cmf.data.table import Table


class DBMixin(BaseModel, ABC):
    db: DB


class TableMixin(BaseModel, ABC):
    db_table: Table
    _db_expected_fields: List[str]

    @model_validator(mode="after")
    def check_table(self) -> Table:
        if self.db_table.exists:
            table_fields = set(self.db_table.db_fields)
            expected_fields = set(self._db_expected_fields)

            if table_fields != expected_fields:
                raise ValueError(
                    f"""\
                    Expected {expected_fields}.
                    Found {table_fields}.
                """
                )

            return self
        else:
            return self
