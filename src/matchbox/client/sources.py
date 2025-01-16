from typing import TypeVar

from pandas import DataFrame
from pyarrow import Table as ArrowTable
from pydantic import (
    model_validator,
)
from sqlalchemy import (
    LABEL_STYLE_TABLENAME_PLUS_COL,
    ColumnElement,
    Engine,
    MetaData,
    String,
    Table,
    cast,
    select,
)
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.hash import HASH_FUNC
from matchbox.common.sources import SourceBase, SourceColumn

T = TypeVar("T")


class Source(SourceBase):
    """A client-side dataset."""

    alias: str
    db_columns: list[SourceColumn] = []
    engine: Engine

    @model_validator(mode="after")
    def hash_columns(self) -> "Source":
        """
        Validates db_columns if specified, and sets to all remote otherwise.
        """
        table = self.to_table()

        remote_columns = {
            col.name: col.type for col in table.columns if col.name not in self.db_pk
        }
        if not self.db_columns:
            self.db_columns = [
                SourceColumn(literal=col_name, type=str(col_type))
                for col_name, col_type in remote_columns
            ]
        else:
            for col in self.db_columns:
                if col.literal not in remote_columns:
                    raise ValueError(
                        f"Column {col.literal} not available in {self.full_name}"
                    )

        return self

    def to_table(self) -> Table:
        """Returns the dataset as a SQLAlchemy Table object."""
        metadata = MetaData(schema=self.db_schema)
        table = Table(self.db_table, metadata, autoload_with=self.database.engine)
        return table

    def _select(
        self,
        fields: list[str] | None,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> Select:
        """Returns a SQLAlchemy Select object to retrieve data from the dataset."""
        table = self.to_table()

        def _get_column(col_name: str) -> ColumnElement:
            """Helper to get a column with proper casting and labeling for PKs"""
            col = table.columns[col_name]
            if col_name == self.db_pk:
                return cast(col, String).label(
                    f"{table.schema}_{table.name}_{col_name}"
                )
            return col

        # Determine which columns to select
        if fields:
            select_cols = [_get_column(field) for field in fields]
        else:
            select_cols = [_get_column(col.name) for col in table.columns]

        stmt = select(*select_cols)

        if pks:
            string_pks = [str(pk) for pk in pks]
            pk_col = table.columns[self.db_pk]
            stmt = stmt.where(cast(pk_col, String).in_(string_pks))

        if limit:
            stmt = stmt.limit(limit)

        return stmt.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)

    def to_hash(self) -> bytes:
        """Generate a unique hash based on the table's columns and datatypes."""
        schema_representation = f"{self.alias}: " + ",".join(
            f"{col.alias.name}:{col.type}" for col in self.db_columns
        )
        return HASH_FUNC(schema_representation.encode("utf-8")).digest()

    def to_arrow(
        self,
        fields: list[str] | None = None,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> ArrowTable:
        """Returns the dataset as a PyArrow Table."""
        stmt = self._select(fields=fields, pks=pks, limit=limit)
        return sql_to_df(stmt, self.database.engine, return_type="arrow")

    def to_pandas(
        self,
        fields: list[str] | None,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> DataFrame:
        """Returns the dataset as a pandas DataFrame."""
        stmt = self._select(fields=fields, pks=pks, limit=limit)
        return sql_to_df(stmt, self.database.engine, return_type="pandas")
