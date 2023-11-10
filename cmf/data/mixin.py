from __future__ import annotations

from typing import List, TYPE_CHECKING
from pydantic import BaseModel, model_validator
from abc import ABC


if TYPE_CHECKING:
    from cmf.data.table import Table
    from cmf.data.datasets import Datasets


class DatasetsMixin(BaseModel, ABC):
    db_datasets: Datasets


class TableMixin(BaseModel, ABC):
    db_table: Table
    _db_expected_fields: List[str]

    @model_validator(mode="after")
    def check_table(self) -> Table:
        if self.db_table is not None:
            if self.db_table.exists:
                table_fields = sorted(self.db_table.db_fields)
                expected_fields = sorted(self._db_expected_fields)

                if table_fields != expected_fields:
                    raise ValueError(
                        f"""\
                        Expected {expected_fields}.
                        Found {table_fields}.
                    """
                    )

        return self
