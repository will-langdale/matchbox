import json
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import pyarrow as pa
from pandas import DataFrame
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import (
    LABEL_STYLE_TABLENAME_PLUS_COL,
    ColumnElement,
    Engine,
    MetaData,
    String,
    Table,
    cast,
    func,
    select,
)
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.exceptions import SourceEngineError
from matchbox.common.hash import HASH_FUNC, hash_data

T = TypeVar("T")


class SourceColumn(BaseModel):
    """A column in a dataset that can be indexed in the Matchbox database."""

    name: str
    alias: str = Field(default_factory=lambda data: data["name"])
    type: str | None = Field(
        default=None, description="The type to cast the column to before hashing data."
    )


class SourceAddress(BaseModel):
    full_name: str
    warehouse_hash: bytes

    @classmethod
    def compose(cls, engine: Engine, full_name: str) -> "SourceAddress":
        """Generate a SourceAddress from a SQLAlchemy Engine and full source name."""
        url = engine.url
        components = {
            "dialect": url.get_dialect().name,
            "database": url.database or "",
            "host": url.host or "",
            "port": url.port or "",
            "schema": getattr(url, "schema", "") or "",
            "service_name": url.query.get("service_name", ""),
        }

        stable_str = json.dumps(components, sort_keys=True).encode()

        hash = HASH_FUNC(stable_str).digest()
        return SourceAddress(full_name=full_name, warehouse_hash=hash)


P = ParamSpec("P")
R = TypeVar("R")


def needs_engine(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to ensure Engine is available to object."""

    @wraps(func)
    def wrapper(self: "Source", *args: P.args, **kwargs: P.kwargs) -> R:
        if not self.engine:
            raise SourceEngineError
        return func(self, *args, **kwargs)

    return wrapper


class Source(BaseModel):
    """A dataset that can, or has been indexed on the backend."""

    address: SourceAddress
    alias: str = Field(default_factory=lambda data: data["address"].full_name)
    db_pk: str
    columns: list[SourceColumn] = []

    _engine: Engine

    @property
    def engine(self) -> Engine | None:
        return self._engine

    def set_engine(self, engine: Engine):
        self._engine = engine
        return self

    @property
    def signature(self) -> bytes:
        """Generate a unique hash based on the table's metadata."""
        schema_representation = f"{self.alias}: " + ",".join(
            f"{col.alias}:{col.type}" for col in self.columns
        )
        return HASH_FUNC(schema_representation.encode("utf-8")).digest()

    def _split_full_name(self) -> tuple[str | None, str]:
        schema_name_list = self.address.full_name.replace('"', "").split(".")

        if len(schema_name_list) == 1:
            db_schema = None
            db_table = schema_name_list[0]
        elif len(schema_name_list) == 2:
            db_schema = schema_name_list[0]
            db_table = schema_name_list[1]
        else:
            raise ValueError(
                f"Could not identify schema and table in {self.address.full_name}."
            )
        return db_schema, db_table

    def format_column(self, column: str) -> str:
        """Outputs a full SQLAlchemy column representation.

        Args:
            column: the name of the column

        Returns:
            A string representing the table name and column
        """
        db_schema, db_table = self._split_full_name()
        if db_schema:
            return f"{db_schema}_{db_table}_{column}"
        return f"{db_table}_{column}"

    @needs_engine
    def index_columns(self, columns: list[SourceColumn] | None = None) -> "Source":
        """Adds columns to usend to Matchbox server, overwriting previous value.

        If no columns are specified, all columns from the source table will be used.
        """
        table = self.to_table()

        remote_columns = {
            col.name: col.type for col in table.columns if col.name not in self.db_pk
        }

        if not columns:
            self.columns = [
                SourceColumn(name=col_name, type=str(col_type))
                for col_name, col_type in remote_columns.items()
            ]
        else:
            for col in columns:
                if col.name not in remote_columns:
                    raise ValueError(
                        f"Column {col.name} not available in {self.address.full_name}"
                    )
            self.columns = columns

        return self

    @needs_engine
    def to_table(self) -> Table:
        """Returns the dataset as a SQLAlchemy Table object."""
        db_schema, db_table = self._split_full_name()
        metadata = MetaData(schema=db_schema)
        table = Table(db_table, metadata, autoload_with=self.engine)
        return table

    def _select(
        self,
        fields: list[str] | None,
        include_pk_column: bool = True,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> Select:
        """Returns a SQLAlchemy Select object to retrieve data from the dataset."""
        table = self.to_table()

        if include_pk_column and self.db_pk not in fields:
            fields.append(self.db_pk)

        def _get_column(col_name: str) -> ColumnElement:
            """Helper to get a column with proper casting and labeling for PKs"""
            col = table.columns[col_name]
            if col_name == self.db_pk:
                return cast(col, String).label(self.format_column(col_name))
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

    @needs_engine
    def to_arrow(
        self,
        fields: list[str] | None = None,
        pks: list[T] | None = None,
        limit: int | None = None,
        include_pk_column: bool = True,
    ) -> pa.Table:
        """Returns the dataset as a PyArrow Table."""
        stmt = self._select(
            fields=fields, pks=pks, limit=limit, include_pk_column=include_pk_column
        )
        return sql_to_df(stmt, self._engine, return_type="arrow")

    @needs_engine
    def to_pandas(
        self,
        fields: list[str] | None,
        pks: list[T] | None = None,
        limit: int | None = None,
        include_pk_column: bool = True,
    ) -> DataFrame:
        """Returns the dataset as a pandas DataFrame."""
        stmt = self._select(
            fields=fields, pks=pks, limit=limit, include_pk_column=include_pk_column
        )
        return sql_to_df(stmt, self._engine, return_type="pandas")

    @needs_engine
    def hash_data(self) -> pa.Table:
        """Retrieve and hash a dataset from its warehouse, ready to be inserted."""
        source_table = self.to_table()
        cols_to_index = tuple([col.name for col in self.columns])

        slct_stmt = select(
            func.concat(*source_table.c[cols_to_index]).label("raw"),
            source_table.c[self.db_pk].cast(String).label("source_pk"),
        )

        raw_result = sql_to_df(slct_stmt, self._engine, "arrow")
        grouped = raw_result.group_by("raw").aggregate([("source_pk", "list")])
        grouped_data = grouped["raw"]
        grouped_keys = grouped["source_pk_list"]

        return pa.table(
            {
                "source_pk": grouped_keys,
                "hash": pa.array(
                    [hash_data(d) for d in grouped_data.to_pylist()], type=pa.binary()
                ),
            }
        )


class Match(BaseModel):
    """A match between primary keys in the Matchbox database."""

    cluster: int | None
    source: SourceAddress
    source_id: set[str] = Field(default_factory=set)
    target: SourceAddress
    target_id: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def found_or_none(self) -> "Match":
        if self.target_id and not (self.source_id and self.cluster):
            raise ValueError(
                "A match must have sources and a cluster if target was found."
            )
        if self.cluster and not self.source_id:
            raise ValueError("A match must have source if cluster is set.")
        return self
