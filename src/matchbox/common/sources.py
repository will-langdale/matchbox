import json
from typing import TypeVar

from pandas import DataFrame
from pyarrow import Table as ArrowTable
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
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
from matchbox.common.hash import HASH_FUNC, hash_to_base64
from matchbox.server import MatchboxDBAdapter, inject_backend

T = TypeVar("T")


class SourceColumnName(BaseModel):
    """A column name in the Matchbox database."""

    name: str

    @property
    def hash(self) -> bytes:
        """Generate a unique hash based on the column name."""
        return HASH_FUNC(self.name.encode("utf-8")).digest()

    @property
    def base64(self) -> str:
        """Generate a base64 encoded hash based on the column name."""
        return hash_to_base64(self.hash)


class SourceColumn(BaseModel):
    """A column in a dataset that can be indexed in the Matchbox database."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    literal: SourceColumnName = Field(
        description="The literal name of the column in the database."
    )
    alias: SourceColumnName = Field(
        default_factory=lambda data: SourceColumnName(name=data["literal"].name),
        description="The alias to use when hashing the dataset in Matchbox.",
    )
    type: str | None = Field(
        default=None, description="The type to cast the column to before hashing data."
    )

    def __eq__(self, other: object) -> bool:
        """Compare SourceColumn with another SourceColumn or bytes object.

        Two SourceColumns are equal if:

        * Their literal names match, or
        * Their alias names match, or
        * The hash of either their literal or alias matches the other object's
        corresponding hash

        A SourceColumn is equal to a bytes object if:

        * The hash of either its literal or alias matches the bytes object

        Args:
            other: Another SourceColumn or a bytes object to compare against

        Returns:
            bool: True if the objects are considered equal, False otherwise
        """
        if isinstance(other, SourceColumn):
            if self.literal == other.literal or self.alias == other.alias:
                return True

            self_hashes = {self.literal.hash, self.alias.hash}
            other_hashes = {other.literal.hash, other.alias.hash}

            return bool(self_hashes & other_hashes)

        if isinstance(other, bytes):
            return other in {self.literal.hash, self.alias.hash}

        return NotImplemented

    @field_validator("literal", "alias", mode="before")
    def string_to_name(cls: "SourceColumn", value: str) -> SourceColumnName:
        if isinstance(value, str):
            return SourceColumnName(name=value)
        else:
            raise ValueError("Column name must be a string.")


class SourceNameAddress(BaseModel):
    full_name: str
    warehouse_hash: bytes

    @classmethod
    def compose(cls, engine: Engine, full_name: str) -> "SourceNameAddress":
        """
        Generate a SourceNameAddress from a SQLAlchemy Engine and full source name.
        """
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
        return SourceNameAddress(full_name=full_name, warehouse_hash=hash)


class Source(BaseModel):
    """A client-side dataset."""

    alias: str
    columns: list[SourceColumn] = []
    engine: Engine
    name_address: SourceNameAddress
    db_pk: str
    alias: str

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


class Selector(BaseModel):
    source_name_address: SourceNameAddress
    fields: list[str]

    @classmethod
    @inject_backend
    def verify(cls, backend: MatchboxDBAdapter, engine: Engine, full_name, fields):
        source_address = SourceNameAddress.compose(engine, full_name)
        source = backend.get_source(source_address)

        warehouse_cols = set(source.to_table().columns.keys())
        selected_cols = set(fields)
        if not selected_cols <= warehouse_cols:
            raise ValueError(
                f"{selected_cols - warehouse_cols} not found in {source_address}"
            )

        indexed_cols = set(col.literal for col in source.columns)
        if not selected_cols <= indexed_cols:
            pass
            # TODO raise warning

        return Selector(source_address, fields)
