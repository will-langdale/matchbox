from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Optional, TypeVar

from pandas import DataFrame
from pyarrow import Table as ArrowTable
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from sqlalchemy import (
    LABEL_STYLE_TABLENAME_PLUS_COL,
    ColumnElement,
    MetaData,
    String,
    Table,
    cast,
    create_engine,
    select,
)
from sqlalchemy import text as sqltext
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.hash import HASH_FUNC, hash_to_base64

T = TypeVar("T")


class WarehouseType(StrEnum):
    """The available warehouse types for Matchbox."""

    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"

    @classmethod
    def get(cls, db_type: str) -> Optional["WarehouseType"]:
        """
        Get a

        Args:
            db_type: a string representing a database type

        Returns:
            a member of this StrEnum
        """
        try:
            return cls(db_type)
        except ValueError:
            return None


class Cluster(BaseModel):
    """A cluster of data in the Matchbox database.

    A cluster describes a single entity resolved at the specified probability
    threshold or higher.
    """

    parent: bytes
    children: set[bytes]
    threshold: float = Field(default=None, ge=0, le=1)


class SourceWarehouse(BaseModel, ABC):
    """A warehouse where source data for datasets in Matchbox can be found."""

    alias: str
    db_type: WarehouseType = WarehouseType.POSTGRESQL
    _engine: Engine | None = None

    @property
    @abstractmethod
    def engine(self) -> Engine: ...

    def test_connection(self):
        try:
            with self.engine.connect() as connection:
                connection.execute(sqltext("SELECT 1"))
        except SQLAlchemyError:
            self._engine = None
            raise

    @classmethod
    @abstractmethod
    def from_engine(cls, engine: Engine, alias: str | None = None) -> "SourceWarehouse":
        db_type = engine.url.get_backend_name()
        db_class = cls.get_warehouse_class(db_type)
        return db_class.from_engine(engine, alias)

    @classmethod
    def get_warehouse_class(cls, db_type: str) -> type["SourceWarehouse"]:
        """
        Fetches a warehouse class from a string describing the database type

        Args:
            db_type: string corresponding to SQLAlchemy backend name

        Returns:
            a sub-class of `SourceWarehouse`, or N
        """

        warehouse_type = WarehouseType.get(db_type)

        if not warehouse_type:
            raise ValueError(f"A warehouse of type {db_type} is not supported")

        class_mapping = {
            WarehouseType.POSTGRESQL: PostgresWarehouse,
            WarehouseType.SQLITE: SQLiteWarehouse,
        }

        return class_mapping.get(warehouse_type)


class SQLiteWarehouse(SourceWarehouse):
    """A SQLite-backed warehouse."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    database: str

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            connection_string = f"sqlite:///{self.database}"
            self._engine = create_engine(connection_string)
            self.test_connection()
        return self._engine

    def __eq__(self, other):
        if not isinstance(other, SQLiteWarehouse):
            return False
        return self.alias == other.alias and self.database == other.database

    @classmethod
    def from_engine(cls, engine: Engine, alias: str | None = None) -> "SQLiteWarehouse":
        url = engine.url

        warehouse = cls(
            alias=alias or url.database,
            database=url.database,
        )
        _ = warehouse.engine

        return warehouse


class PostgresWarehouse(SourceWarehouse):
    """A Postgres-backed warehouse."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    user: str
    password: SecretStr
    host: str
    port: int
    database: str

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            connection_string = f"postgresql://{self.user}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.database}"
            self._engine = create_engine(connection_string)
            self.test_connection()
        return self._engine

    def __eq__(self, other):
        if not isinstance(other, PostgresWarehouse):
            return False
        return (
            self.alias == other.alias
            and self.db_type == other.db_type
            and self.user == other.user
            and self.password == other.password
            and self.host == other.host
            and self.port == other.port
            and self.database == other.database
        )

    @classmethod
    def from_engine(
        cls, engine: Engine, alias: str | None = None
    ) -> "PostgresWarehouse":
        url = engine.url

        warehouse = cls(
            alias=alias or url.database,
            user=url.username,
            password=url.password,
            host=url.host,
            port=url.port or 0,
            database=url.database,
        )
        _ = warehouse.engine

        return warehouse


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
    indexed: bool = Field(description="Whether the column is indexed in the database.")

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


class Source(BaseModel):
    """A dataset that can be indexed in the Matchbox database."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    database: SourceWarehouse
    db_pk: str
    db_schema: str = None
    db_table: str
    db_columns: list[SourceColumn] = Field(..., min_length=1)
    alias: str = Field(
        default_factory=lambda data: f"{data['db_schema']}.{data['db_table']}"
    )

    def __str__(self) -> str:
        return f"{self.db_schema}.{self.db_table}"

    def __hash__(self) -> int:
        return hash(
            (type(self), self.db_pk, self.db_schema, self.db_table, self.database.alias)
        )

    @model_validator(mode="before")
    @classmethod
    def hash_columns(cls, data: dict[str, Any]) -> "Source":
        """Shapes indices data from either the backend or TOML.

        Handles three scenarios:
            1. No columns specified - all columns except primary key are indexed
            2. Indices from database - uses existing column hash information
            3. Columns specified in TOML - specified columns are indexed
        """
        # Initialise warehouse and get table metadata
        if isinstance(data["database"], SourceWarehouse):
            warehouse = data["database"]
        else:
            db_type = data["database"].pop("db_type", None)
            warehouse_class = SourceWarehouse.get_warehouse_class(db_type)
            warehouse = warehouse_class(**data["database"])

        optional_schema = None if "db_schema" not in data else data["db_schema"]
        metadata = MetaData(schema=optional_schema)
        table = Table(data["db_table"], metadata, autoload_with=warehouse.engine)

        # Get all columns except primary key
        remote_columns = [
            SourceColumn(literal=col.name, type=str(col.type), indexed=False)
            for col in table.columns
            if col.name not in data["db_pk"]
        ]

        index_data = data.get("index")

        # Case 1: No columns specified - index everything
        if not index_data:
            data["db_columns"] = [
                SourceColumn(literal=col.literal.name, type=col.type, indexed=True)
                for col in remote_columns
            ]
            return data

        # Case 2: Columns from database
        if isinstance(index_data, dict):
            data["db_columns"] = [
                SourceColumn(
                    literal=col.literal.name,
                    type=col.type,
                    indexed=col in index_data["literal"] + index_data["alias"],
                )
                for col in remote_columns
            ]
            return data

        # Case 3: Columns from TOML
        local_columns = []

        # Process TOML column specifications
        for column in index_data:
            local_columns.append(
                SourceColumn(
                    literal=column["literal"],
                    alias=column.get("alias", column["literal"]),
                    indexed=True,
                )
            )

        # Match remote columns with local specifications
        indexed_columns = []
        non_indexed_columns = []

        for remote_col in remote_columns:
            matched = False
            for local_col in local_columns:
                if remote_col.literal == local_col.literal:
                    indexed_columns.append(local_col)
                    matched = True
                    break
            if not matched:
                non_indexed_columns.append(remote_col)

        data["db_columns"] = indexed_columns + non_indexed_columns

        return data

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
            f"{col.alias.name}:{col.type}" for col in self.db_columns if col.indexed
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
