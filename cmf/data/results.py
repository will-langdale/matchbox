from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy.orm import Session, select

from cmf.data.db import ENGINE
from cmf.data.models import Models
from cmf.data.utils import get_schema_table_names, string_to_table

if TYPE_CHECKING:
    from pandas import DataFrame
    from sqlalchemy import Engine, Table


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


class Results(TableMixin, DataFrameMixin):
    dataframe: Optional[DataFrame] = None
    db_table: Optional[Table] = None
    run_name: str
    description: str
    left: str
    right: str

    _expected_fields: List[str]

    def _validate_tables(self, engine: Engine = ENGINE) -> bool:
        db_left_schema, db_left_table = get_schema_table_names(
            full_name=self.left, validate=True
        )
        db_left = string_to_table(
            db_schema=db_left_schema, db_table=db_left_table, engine=engine
        )
        db_right_schema, db_right_table = get_schema_table_names(
            full_name=self.right, validate=True
        )
        db_right = string_to_table(
            db_schema=db_right_schema, db_table=db_right_table, engine=engine
        )

        if db_left.exists and db_right.exists:
            return True
        else:
            return False

    def _validate_sources(self, engine: Engine) -> bool:
        stmt_left = select(Models.name).where(Models.name == self.left)
        stmt_right = select(Models.name).where(Models.name == self.right)

        with Session(engine) as session:
            res_left = session.execute(stmt_left).scalar()
        with Session(engine) as session:
            res_right = session.execute(stmt_right).scalar()

        if res_left is not None and res_right is not None:
            return True
        else:
            return False


class ProbabilityResults(Results):
    pass


class ClusterResults(Results):
    pass
