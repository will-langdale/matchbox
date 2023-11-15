from __future__ import annotations

from typing import List, TYPE_CHECKING
from pydantic import BaseModel, model_validator, ConfigDict
from abc import ABC


if TYPE_CHECKING:
    from cmf.data.table import Table
    from pandas import DataFrame


class TableMixin(BaseModel, ABC):
    db_table: Table
    _expected_fields: List[str]

    @model_validator(mode="after")
    def check_table(self) -> Table:
        if self.db_table is not None:
            if self.db_table.exists:
                table_fields = sorted(self.db_table.db_fields)
                expected_fields = sorted(self._expected_fields)

                if table_fields != expected_fields:
                    raise ValueError(
                        f"""\
                        Expected {expected_fields}.
                        Found {table_fields}.
                    """
                    )

        return self


class DataFrameMixin(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: DataFrame
    _expected_fields: List[str]

    @model_validator(mode="after")
    def check_dataframe(self) -> Table:
        if self.dataframe is not None:
            table_fields = sorted(self.dataframe.columns)
            expected_fields = sorted(self._expected_fields)

            if table_fields != expected_fields:
                raise ValueError(
                    f"""\
                    Expected {expected_fields}.
                    Found {table_fields}.
                """
                )

        return self
