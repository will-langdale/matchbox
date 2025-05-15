"""Classes and functions for working with data sources in Matchbox."""

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import wraps
from typing import (
    Annotated,
    Any,
    Callable,
    Iterator,
    Literal,
    ParamSpec,
    Self,
    TypeVar,
    Union,
)

import polars as pl
from pandas import DataFrame as PandasDataframe
from polars import DataFrame as PolarsDataFrame
from pyarrow import RecordBatch as ArrowRecordBatch
from pyarrow import Table as ArrowTable
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    WithJsonSchema,
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
from sqlalchemy.exc import OperationalError

from matchbox.common.db import (
    fullname_to_prefix,
    get_schema_table_names,
    sql_to_df,
    validate_sql_for_data_extraction,
)
from matchbox.common.dtos import DataTypes, SourceResolutionName
from matchbox.common.exceptions import (
    MatchboxSourceEngineError,
    MatchboxSourceFieldError,
)
from matchbox.common.hash import (
    HASH_FUNC,
    HashMethod,
    base64_to_hash,
    hash_rows,
    hash_to_base64,
)

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


LocationType = Union["RelationalDBLocation"]
"""Union type alias for Location class. Currently only supports RelationalDBLocation."""


def requires_credentials(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator that checks if credentials are set before executing a method.

    A helper method for Location subclasses.

    Raises:
        AttributeError: If the credentials are not set.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs) -> T:
        if self.credentials is None:
            raise AttributeError(
                f"Credentials are required for {method.__name__}. "
                "Use add_credentials() method to set credentials for "
                f"this {self.type} location."
            )
        return method(self, *args, **kwargs)

    return wrapper


class Location(ABC, BaseModel):
    """A location for a data source."""

    type: LocationType
    uri: AnyUrl
    credentials: Any | None = Field(exclude=True, default=None)

    @abstractmethod
    def add_credentials(self, credentials: Any) -> None:
        """Adds credentials to the location."""
        ...

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data location.

        Raises:
            AttributeError: If the credentials are not set.
        """
        ...

    @abstractmethod
    def validate_extract_transform(self, extract_transform: str) -> bool:
        """Validate SQL ET logic against this location's query language.

        Raises:
            MatchboxSourceExtractTransformError: If the ET logic is invalid.
        """
        ...

    @abstractmethod
    def execute(
        self, extract_transform: str, batch_size: int
    ) -> Iterator[ArrowRecordBatch]:
        """Execute ET logic against this location and return Arrow RecordBatches.

        Raises:
            AttributeError: If the credentials are not set.
        """
        ...

    @classmethod
    def create(cls, data: dict[str, Any]) -> "Location":
        """Factory method to create the appropriate Location subclass.

        Examples:
            ```python
            loc = Location.create({"type": "rdbms", "uri": "postgresql://..."})
            isisntance(loc, RelationalDBLocation)  # True
            ```
        """
        location_type = data.get("type")

        if location_type == "rdbms":
            return RelationalDBLocation.model_validate(data)

        raise ValueError(f"Unknown location type: {location_type}")


class RelationalDBLocation(Location):
    """A location for a relational database."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["rdbms"] = "rdbms"
    uri: AnyUrl
    credentials: Engine | None = Field(
        exclude=True,
        default=None,
        description=(
            "The credentials for a relational database are a SQLAlchemy Engine."
        ),
    )

    @field_validator("uri", mode="after")
    @classmethod
    def validate_uri(cls, value: AnyUrl) -> AnyUrl:
        """Ensure no credentials, query params, or fragments are in the URI."""
        if value.username or value.password:
            raise ValueError("Credentials should not be in the URI.")
        if value.query or value.fragment:
            raise ValueError("Query params and fragments should not be in the URI.")
        if "+" in value.scheme:
            raise ValueError("Driver should not be in the URI.")
        return value

    def _validate_engine(self, credentials: Engine) -> None:
        """Validate an engine matches the URI.

        Raises:
            ValueError: If the Engine and URI do not match.
        """
        uri = AnyUrl(str(credentials.url))
        scheme_without_driver = uri.scheme.split("+")[0]

        if any(
            [
                scheme_without_driver != self.uri.scheme,
                uri.host != self.uri.host,
                uri.port != self.uri.port,
                uri.path != self.uri.path,
            ]
        ):
            raise ValueError(
                "The Engine location URI does not match the location model URI. \n"
                f"Scheme: {scheme_without_driver}, {self.uri.scheme} \n"
                f"Host: {uri.host}, {self.uri.host} \n"
                f"Port: {uri.port}, {self.uri.port} \n"
                f"Path: {uri.path}, {self.uri.path} \n"
            )

    def add_credentials(self, credentials: Engine) -> None:  # noqa: D102
        self._validate_engine(credentials)
        self.credentials = credentials

    @requires_credentials
    def connect(self) -> bool:  # noqa: D102
        try:
            _ = sql_to_df(stmt="select 1", connection=self.credentials)
            return True
        except OperationalError:
            return False

    def validate_extract_transform(self, extract_transform: str) -> bool:  # noqa: D102
        # We are NOT attempting a full sanitisation of the SQL statement
        # Validation is done purely to stop accidental mistakes, not malicious actors
        # Users should only run indexing using SourceConfigs they trust and have read,
        # using least privilege credentials
        return validate_sql_for_data_extraction(extract_transform)

    @requires_credentials
    def execute(  # noqa: D102
        self, extract_transform: str, batch_size: int
    ) -> Iterator[PolarsDataFrame]:
        yield from sql_to_df(
            stmt=extract_transform,
            connection=self.credentials,
            batch_size=batch_size,
            return_batches=True,
            return_type="polars",
        )

    @classmethod
    def from_engine(cls, engine: Engine) -> "RelationalDBLocation":
        """Create a RelationalDBLocation from a SQLAlchemy Engine."""
        cleaned_url = engine.url.set(
            username=None,
            password=None,
            query={},
        )
        location = cls(uri=str(cleaned_url))
        location.add_credentials(engine)
        return location


class SourceField(BaseModel):
    """A field in a source that can be indexed in the Matchbox database."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description=(
            "The name of the field in the source after the "
            "extract/transform logic has been applied."
        )
    )
    type: DataTypes = Field(
        description="The cached field type. Used to ensure a stable hash.",
    )


def b64_bytes_validator(val: bytes | str) -> bytes:
    """Ensure that a value is a base64 encoded string or bytes."""
    if isinstance(val, bytes):
        return val
    elif isinstance(val, str):
        return base64_to_hash(val)
    raise ValueError(f"Value {val} could not be converted to bytes")


SerialisableBytes = Annotated[
    bytes,
    PlainValidator(b64_bytes_validator),
    PlainSerializer(lambda v: hash_to_base64(v)),
    WithJsonSchema(
        {"type": "string", "format": "base64", "description": "Base64 encoded bytes"}
    ),
]


class SourceAddress(BaseModel):
    """A unique identifier for a dataset in a warehouse."""

    model_config = ConfigDict(frozen=True)

    full_name: str
    warehouse_hash: SerialisableBytes

    def __str__(self) -> str:
        """Convert to a string."""
        return self.full_name + "@" + self.warehouse_hash_b64

    @property
    def pretty(self) -> str:
        """Return a pretty representation of the address."""
        return self.full_name + "@" + self.warehouse_hash_b64[:5] + "..."

    @property
    def warehouse_hash_b64(self) -> str:
        """Return warehouse hash as a base64 encoded string."""
        return hash_to_base64(self.warehouse_hash)

    @classmethod
    def compose(cls, engine: Engine, full_name: str) -> "SourceAddress":
        """Generate a SourceAddress from a SQLAlchemy Engine and schema.table name."""
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

    def format_field(self, field: str) -> str:
        """Outputs a full SQLAlchemy field representation.

        Args:
            field: the name of the field

        Returns:
            A string representing the table name and field
        """
        return fullname_to_prefix(self.full_name) + field


def needs_engine(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to check that engine is set."""

    @wraps(func)
    def wrapper(self: "SourceConfig", *args: P.args, **kwargs: P.kwargs) -> R:
        if not self.engine:
            raise MatchboxSourceEngineError

        return func(self, *args, **kwargs)

    return wrapper


class SourceConfig(BaseModel):
    """Data that can, or has been indexed on the backend.

    The configuration produces a source that maps to a "noun" of the business,
    such as a "customer" or "product". It contains fields that are attributes of
    the noun, and the unique key that identifies it in the source system.
    """

    model_config = ConfigDict(frozen=True)

    address: SourceAddress
    name: SourceResolutionName = Field(
        default_factory=lambda data: str(data["address"])
    )
    key_field: SourceField
    # Index fields need to be set at creation, or initialised with `.default_fields()`
    index_fields: tuple[SourceField, ...] | None = None

    _engine: Engine | None = None

    @field_validator("key_field", mode="before")
    @classmethod
    def validate_key_field(
        cls: type[Self], key_field: str | dict[str, str] | SourceField
    ) -> SourceField:
        """Validate key field as valid SourceField."""
        if isinstance(key_field, str):
            return SourceField(name=key_field, type=DataTypes.STRING)
        elif isinstance(key_field, dict):
            key_field = SourceField.model_validate(key_field)
        elif not isinstance(key_field, SourceField):
            raise ValueError(
                f"Key field must be a string, dict, or SourceField, but got {key_field}"
            )

        if key_field.type != DataTypes.STRING:
            raise ValueError(
                f"Key field must be a string type, but got {key_field.type}"
            )

        return key_field

    @property
    def engine(self) -> Engine | None:
        """The SQLAlchemy Engine used to connect to the data."""
        return self._engine

    def __eq__(self, other: Any) -> bool:
        """Custom equality which ignores engine."""
        return (self.address, self.name, self.key_field, self.index_fields) == (
            other.address,
            other.name,
            other.key_field,
            other.index_fields,
        )

    def __hash__(self) -> int:
        """Custom hash which ignores engine."""
        return hash((self.address, self.name, self.key_field, self.index_fields))

    def __deepcopy__(self, memo=None):
        """Create a deep copy of the SourceConfig object."""
        if memo is None:
            memo = {}

        obj_copy = SourceConfig(
            address=deepcopy(self.address, memo),
            name=deepcopy(self.name, memo),
            key_field=deepcopy(self.key_field, memo),
            index_fields=deepcopy(self.index_fields, memo),
        )

        # Both objects should share the same engine
        if self._engine is not None:
            obj_copy._engine = self._engine

        return obj_copy

    def set_engine(self, engine: Engine):
        """Adds engine, and use it to validate current fields."""
        implied_address = SourceAddress.compose(
            full_name=self.address.full_name, engine=engine
        )
        if implied_address != self.address:
            raise ValueError("The engine does not match the source address.")

        self._engine = engine

        return self

    @needs_engine
    def get_remote_fields(self, exclude_key=False) -> dict[str, DataTypes]:
        """Returns a dictionary of field names and Matchbox DataTypes."""
        table = self.to_table()
        return {
            field.name: DataTypes.from_pytype(field.type.python_type)
            for field in table.columns
            if (field.name != self.key_field.name) or (not exclude_key)
        }

    @needs_engine
    def default_fields(self) -> "SourceConfig":
        """Returns a new source with default fields.

        Default fields are all from the source warehouse other than `self.key_field`.
        All other attributes are copied, and its engine (if present) is set.
        """
        remote_fields = self.get_remote_fields(exclude_key=True)
        index_fields = (
            SourceField(name=field_name, type=field_type)
            for field_name, field_type in remote_fields.items()
        )

        new_source = SourceConfig(
            address=self.address,
            name=self.name,
            key_field=self.key_field,
            index_fields=index_fields,
        )

        if self.engine:
            new_source.set_engine(self.engine)

        return new_source

    @needs_engine
    def to_table(self) -> Table:
        """Returns the source as a SQLAlchemy Table object."""
        db_schema, db_table = get_schema_table_names(self.address.full_name)
        metadata = MetaData(schema=db_schema)
        table = Table(db_table, metadata, autoload_with=self.engine)
        return table

    @needs_engine
    def check_fields(self, fields: list[str] | None = None) -> None:
        """Check that fields are available in the warehouse and correctly typed.

        Args:
            fields: List of field names to check. If None, it will check
                self.index_fields
        """
        remote_fields = self.get_remote_fields()

        if self.key_field.name not in remote_fields:
            raise MatchboxSourceFieldError(
                f"Key field {self.key_field.name} not available in {self.address}"
            )

        if fields:
            fields = set(fields)
            remote_names = set(remote_fields.keys())
            if not fields <= remote_names:
                raise MatchboxSourceFieldError(
                    f"Fields {fields - remote_names} not in {self.address}"
                )
        else:
            if not self.index_fields:
                raise ValueError("No fields passed, and none set on the SourceConfig.")

            for field in self.index_fields:
                if field.name not in remote_fields:
                    raise MatchboxSourceFieldError(
                        f"Field {field.name} not available in {self.address.full_name}"
                    )
                actual_type = remote_fields[field.name]
                if actual_type != field.type:
                    raise MatchboxSourceFieldError(
                        f"Type {actual_type} != {field.type} for {field.name}"
                    )

    def _select(
        self,
        fields: list[str] | None = None,
        keys: list[T] | None = None,
        limit: int | None = None,
    ) -> str:
        """Returns a SQL query to retrieve data from the source."""
        table = self.to_table()

        # Ensure all set fields are available and the expected type
        self.check_fields(fields=fields)
        if not fields:
            fields = [field.name for field in self.index_fields]

        if self.key_field.name not in fields:
            fields.append(self.key_field.name)

        def _get_field(field_name: str) -> ColumnElement:
            """Helper to get a field with proper casting and labeling for keys."""
            field = table.columns[field_name]
            if field_name == self.key_field:
                return cast(field, String).label(self.address.format_field(field_name))
            return field

        # Determine which fields to select
        if fields:
            select_fields = [_get_field(field) for field in fields]
        else:
            select_fields = [_get_field(field.name) for field in table.columns]

        stmt = select(*select_fields)

        if keys:
            string_keys = [str(key) for key in keys]
            key_field = table.columns[self.key_field]
            stmt = stmt.where(cast(key_field, String).in_(string_keys))

        if limit:
            stmt = stmt.limit(limit)

        stmt = stmt.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)

        return stmt.compile(
            dialect=self.engine.dialect, compile_kwargs={"literal_binds": True}
        )

    @needs_engine
    def to_arrow(
        self,
        fields: list[str] | None = None,
        keys: list[T] | None = None,
        limit: int | None = None,
        *,
        return_batches: bool = False,
        batch_size: int | None = None,
        schema_overrides: dict[str, Any] | None = None,
        execute_options: dict[str, Any] | None = None,
    ) -> ArrowTable | Iterator[ArrowTable]:
        """Returns the source as a PyArrow Table or an iterator of PyArrow Tables.

        Args:
            fields: List of field names to retrieve. If None, retrieves all fields.
            keys: List of primary keys to filter by. If None, retrieves all rows.
            limit: Maximum number of rows to retrieve. If None, retrieves all rows.
            return_batches:
                * If True, return an iterator that yields each batch separately
                * If False, return a single Table with all results
            batch_size: Indicate the size of each batch when processing data in batches.
            schema_overrides: A dictionary mapping field names to dtypes.
            execute_options: These options will be passed through into the underlying
                query execution method as kwargs.

        Returns:
            The requested data in PyArrow format.

                * If return_batches is False: a PyArrow Table
                * If return_batches is True: an iterator of PyArrow Tables
        """
        stmt = self._select(fields=fields, keys=keys, limit=limit)
        return sql_to_df(
            stmt,
            self._engine,
            return_type="arrow",
            return_batches=return_batches,
            batch_size=batch_size,
            schema_overrides=schema_overrides,
            execute_options=execute_options,
        )

    @needs_engine
    def to_polars(
        self,
        fields: list[str] | None = None,
        keys: list[T] | None = None,
        limit: int | None = None,
        *,
        return_batches: bool = False,
        batch_size: int | None = None,
        schema_overrides: dict[str, Any] | None = None,
        execute_options: dict[str, Any] | None = None,
    ) -> PolarsDataFrame | Iterator[PolarsDataFrame]:
        """Returns the source as a PyArrow Table or an iterator of PyArrow Tables.

        Args:
            fields: List of field names to retrieve. If None, retrieves all fields.
            keys: List of primary keys to filter by. If None, retrieves all rows.
            limit: Maximum number of rows to retrieve. If None, retrieves all rows.
            return_batches:
                * If True, return an iterator that yields each batch separately
                * If False, return a single Table with all results
            batch_size: Indicate the size of each batch when processing data in batches.
            schema_overrides: A dictionary mapping field names to dtypes.
            execute_options: These options will be passed through into the underlying
                query execution method as kwargs.

        Returns:
            The requested data in Polars format.

                * If return_batches is False: a Polars DataFrame
                * If return_batches is True: an iterator of Polars DataFrames
        """
        stmt = self._select(fields=fields, keys=keys, limit=limit)
        return sql_to_df(
            stmt,
            self._engine,
            return_type="polars",
            return_batches=return_batches,
            batch_size=batch_size,
            schema_overrides=schema_overrides,
            execute_options=execute_options,
        )

    @needs_engine
    def to_pandas(
        self,
        fields: list[str] | None = None,
        keys: list[T] | None = None,
        limit: int | None = None,
        *,
        return_batches: bool = False,
        batch_size: int | None = None,
        schema_overrides: dict[str, Any] | None = None,
        execute_options: dict[str, Any] | None = None,
    ) -> PandasDataframe | Iterator[PandasDataframe]:
        """Returns the source as a pandas DataFrame or an iterator of DataFrames.

        Args:
            fields: List of field names to retrieve. If None, retrieves all fields.
            keys: List of primary keys to filter by. If None, retrieves all rows.
            limit: Maximum number of rows to retrieve. If None, retrieves all rows.
            return_batches:
                * If True, return an iterator that yields each batch separately
                * If False, return a single Table with all results
            batch_size: Indicate the size of each batch when processing data in batches.
            schema_overrides: A dictionary mapping field names to dtypes.
            execute_options: These options will be passed through into the underlying
                query execution method as kwargs.

        Returns:
            The requested data in Pandas format.

                * If return_batches is False: a Pandas DataFrame
                * If return_batches is True: an iterator of Pandas DataFrames
        """
        stmt = self._select(fields=fields, keys=keys, limit=limit)
        return sql_to_df(
            stmt,
            self._engine,
            return_type="pandas",
            return_batches=return_batches,
            batch_size=batch_size,
            schema_overrides=schema_overrides,
            execute_options=execute_options,
        )

    @needs_engine
    def hash_data(
        self,
        *,
        batch_size: int | None = None,
        schema_overrides: dict[str, Any] | None = None,
        execute_options: dict[str, Any] | None = None,
    ) -> ArrowTable:
        """Retrieve and hash the source from its warehouse, ready to be inserted.

        Args:
            batch_size: If set, process data in batches internally. Indicates the
                size of each batch.
            schema_overrides: A dictionary mapping field names to dtypes.
            execute_options: These options will be passed through into the underlying
                query execution method as kwargs.

        Returns:
            A PyArrow Table containing source primary keys and their hashes.
        """
        # Ensure all set fields are available and the expected type
        self.check_fields()

        source_table = self.to_table()
        fields_to_index = tuple([field.name for field in self.index_fields])

        slct_stmt = select(
            *[source_table.c[field] for field in fields_to_index],
            source_table.c[self.key_field.name].cast(String).label("keys"),
        )

        def _process_batch(
            batch: PolarsDataFrame,
            fields_to_index: tuple,
        ) -> PolarsDataFrame:
            """Process a single batch of data using Polars.

            Args:
                batch: Polars DataFrame containing the data
                fields_to_index: fields to include in the hash

            Returns:
                Polars DataFrame with hash and key fields

            Raises:
                ValueError: If any key values are null
            """
            if batch["keys"].is_null().any():
                raise ValueError("key field contains null values")

            row_hashes = hash_rows(
                df=batch,
                columns=list(sorted(fields_to_index)),
                method=HashMethod.SHA256,
            )

            result = batch.with_columns(row_hashes.alias("hash")).select(
                ["hash", "keys"]
            )

            return result

        if bool(batch_size):
            # Process in batches
            raw_batches = sql_to_df(
                slct_stmt,
                self._engine,
                return_type="polars",
                return_batches=True,
                batch_size=batch_size,
                schema_overrides=schema_overrides,
                execute_options=execute_options,
            )

            all_results = []
            for batch in raw_batches:
                batch_result = _process_batch(batch, fields_to_index)
                all_results.append(batch_result)

            processed_df = pl.concat(all_results)

        else:
            # Non-batched processing
            raw_result = sql_to_df(
                slct_stmt,
                self._engine,
                return_type="polars",
                schema_overrides=schema_overrides,
                execute_options=execute_options,
            )

            processed_df = _process_batch(raw_result, fields_to_index)

        return processed_df.group_by("hash").agg(pl.col("keys")).to_arrow()


class Match(BaseModel):
    """A match between primary keys in the Matchbox database."""

    cluster: int | None
    source: SourceAddress
    source_id: set[str] = Field(default_factory=set)
    target: SourceAddress
    target_id: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def found_or_none(self) -> "Match":
        """Ensure that a match has sources and a cluster if target was found."""
        if self.target_id and not (self.source_id and self.cluster):
            raise ValueError(
                "A match must have sources and a cluster if target was found."
            )
        if self.cluster and not self.source_id:
            raise ValueError("A match must have source if cluster is set.")
        return self
