"""Utilities to interact with a data warehouse."""

from typing import Any, Iterator, TypeVar

import polars as pl
import pyarrow as pa
from pandas import DataFrame as PandasDataFrame
from sqlalchemy import (
    LABEL_STYLE_TABLENAME_PLUS_COL,
    ColumnElement,
    Engine,
    MetaData,
    String,
    Table,
    cast,
    create_engine,
)
from sqlalchemy import (
    select as sqlselect,
)
from sqlalchemy.sql.selectable import Select

from matchbox.client._settings import settings
from matchbox.common.db import (
    fullname_to_prefix,
    get_schema_table_names,
    sql_to_df,
)
from matchbox.common.hash import hash_data
from matchbox.common.logging import logger
from matchbox.common.sources import SourceAddress, SourceColumn, SourceConfig


def engine_fallback(engine: Engine | None = None):
    """Returns passed engine or looks for a default one."""
    if not engine:
        if default_engine := settings.default_warehouse:
            engine = create_engine(default_engine)
            logger.warning("Using default engine")
        else:
            raise ValueError(
                "An engine needs to be provided if "
                "`MB__CLIENT__DEFAULT_WAREHOUSE` is unset"
            )
    return engine


T = TypeVar("T")


class SourceReader:
    """An interface to a source in a warehouse."""

    def __init__(
        self,
        engine: Engine,
        full_name: str,
        db_pk: str,
        fields: list[str] | None = None,
    ):
        """Create new `SourceReader`.

        Args:
            engine: Engine to use in `SourceAddress` and to connect to warehouse.
            full_name: Full name of the source in the warehouse.
            db_pk: name of the field corresponding to the primary key for each row.
            fields: optional sub-set of fields in the source.
        """
        self.engine = self.engine
        self.address = SourceAddress.compose(engine=engine, full_name=full_name)
        self.db_pk = db_pk
        self.fields = fields

        warehouse_fields = set(self.remote_fields().keys())
        selected_fields = set(self.fields)
        if not selected_fields <= warehouse_fields:
            raise ValueError(
                f"{selected_fields - warehouse_fields} not in {str(self.address)}"
            )

    def to_table(self) -> Table:
        """Returns the dataset as a SQLAlchemy Table object."""
        db_schema, db_table = get_schema_table_names(self.address.full_name)
        metadata = MetaData(schema=db_schema)
        table = Table(db_table, metadata, autoload_with=self.engine)
        return table

    def format_field(self, field: str) -> str:
        """Map field name in the source to prefixed name returned by `SourceReader`.

        Args:
            field: Field name in the source

        Returns:
            Prefixed field name
        """
        return fullname_to_prefix(self.address.full_name) + field

    def _select(
        self,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> Select:
        """Returns a SQLAlchemy Select object to retrieve data from the dataset."""
        table = self.to_table()

        def _get_field(col_name: str) -> ColumnElement:
            """Helper to get a column with proper casting and labeling for PKs."""
            col = table.columns[col_name]
            if col_name == self.db_pk:
                label = self.format_field(col_name)
                return cast(col, String).label(label)
            return col

        # Determine which columns to select
        if self.fields:
            fields = set(self.fields + self.db_pk)
            select_cols = [_get_field(field) for field in fields]
        else:
            select_cols = [_get_field(col.name) for col in table.columns]

        stmt = sqlselect(*select_cols)

        if pks:
            string_pks = [str(pk) for pk in pks]
            pk_col = table.columns[self.db_pk]
            stmt = stmt.where(cast(pk_col, String).in_(string_pks))

        if limit:
            stmt = stmt.limit(limit)

        return stmt.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)

    def to_arrow(
        self,
        pks: list[T] | None = None,
        limit: int | None = None,
        *,
        iter_batches: bool = False,
        batch_size: int | None = None,
        schema_overrides: dict[str, Any] | None = None,
        execute_options: dict[str, Any] | None = None,
    ) -> pa.Table | Iterator[pa.Table]:
        """Returns the dataset as a PyArrow Table or an iterator of PyArrow Tables.

        Args:
            pks: List of primary keys to filter by. If None, retrieves all rows.
            limit: Maximum number of rows to retrieve. If None, retrieves all rows.
            iter_batches: If True, return an iterator that yields each batch separately.
                If False, return a single Table with all results.
            batch_size: Indicate the size of each batch when processing data in batches.
            schema_overrides: A dictionary mapping column names to dtypes.
            execute_options: These options will be passed through into the underlying
                query execution method as kwargs.

        Returns:
            If iter_batches is False: A PyArrow Table containing the requested data.
            If iter_batches is True: An iterator of PyArrow Tables.
        """
        stmt = self._select(pks=pks, limit=limit)
        return sql_to_df(
            stmt,
            self.engine,
            return_type="arrow",
            iter_batches=iter_batches,
            batch_size=batch_size,
            schema_overrides=schema_overrides,
            execute_options=execute_options,
        )

    def to_pandas(
        self,
        pks: list[T] | None = None,
        limit: int | None = None,
        *,
        iter_batches: bool = False,
        batch_size: int | None = None,
        schema_overrides: dict[str, Any] | None = None,
        execute_options: dict[str, Any] | None = None,
    ) -> PandasDataFrame | Iterator[PandasDataFrame]:
        """Returns the dataset as a pandas DataFrame or an iterator of DataFrames.

        Args:
            pks: List of primary keys to filter by. If None, retrieves all rows.
            limit: Maximum number of rows to retrieve. If None, retrieves all rows.
            iter_batches: If True, return an iterator that yields each batch separately.
                If False, return a single DataFrame with all results.
            batch_size: Indicate the size of each batch when processing data in batches.
            schema_overrides: A dictionary mapping column names to dtypes.
            execute_options: These options will be passed through into the underlying
                query execution method as kwargs.

        Returns:
            If iter_batches is False: A pandas DataFrame containing the requested data.
            If iter_batches is True: An iterator of pandas DataFrames.
        """
        stmt = self._select(pks=pks, limit=limit)
        return sql_to_df(
            stmt,
            self.engine,
            return_type="pandas",
            iter_batches=iter_batches,
            batch_size=batch_size,
            schema_overrides=schema_overrides,
            execute_options=execute_options,
        )

    def hash_data(
        self,
        *,
        iter_batches: bool = False,
        batch_size: int | None = None,
        schema_overrides: dict[str, Any] | None = None,
        execute_options: dict[str, Any] | None = None,
    ) -> pa.Table:
        """Retrieve and hash a dataset from its warehouse, ready to be inserted.

        Args:
            iter_batches: If True, process data in batches internally.
                This is only used for internal processing and will still return a
                single table.
            batch_size: Indicate the size of each batch when processing data in batches.
            schema_overrides: A dictionary mapping column names to dtypes.
            execute_options: These options will be passed through into the underlying
                query execution method as kwargs.

        Returns:
            A PyArrow Table containing source primary keys and their hashes.
        """
        source_table = self.to_table()
        cols_to_index = tuple(self.fields)

        slct_stmt = sqlselect(
            *[source_table.c[col] for col in cols_to_index],
            source_table.c[self.db_pk].cast(String).label("source_pk"),
        )

        def _process_batch(batch: pl.DataFrame, cols_to_index: tuple) -> pl.DataFrame:
            """Process a single batch of data using Polars.

            Args:
                batch: Polars DataFrame containing the data
                cols_to_index: Columns to include in the hash

            Returns:
                Polars DataFrame with hash and source_pk columns
            """
            for col_name in cols_to_index:
                batch = batch.with_columns(pl.col(col_name).cast(pl.Utf8))

            batch = batch.with_columns(
                pl.concat_str([pl.col(c) for c in cols_to_index]).alias("raw_value")
            )

            # TODO: add column names
            batch = batch.with_columns((pl.col("raw_value")).alias("value_with_sig"))

            batch = batch.with_columns(
                pl.col("value_with_sig")
                .map_elements(lambda x: hash_data(x), return_dtype=pl.Binary)
                .alias("hash")
            )

            return batch.select(["hash", "source_pk"])

        if iter_batches:
            # Process in batches
            raw_batches = sql_to_df(
                slct_stmt,
                self.engine,
                return_type="polars",
                iter_batches=True,
                batch_size=batch_size,
                schema_overrides=schema_overrides,
                execute_options=execute_options,
            )

            all_results = []
            for batch in raw_batches:
                batch_result = _process_batch(batch, cols_to_index)
                all_results.append(batch_result)

            processed_df = pl.concat(all_results)

        else:
            # Non-batched processing
            raw_result = sql_to_df(
                slct_stmt,
                self.engine,
                return_type="polars",
                schema_overrides=schema_overrides,
                execute_options=execute_options,
            )

            processed_df = _process_batch(raw_result, cols_to_index)

        return processed_df.group_by("hash").agg(pl.col("source_pk")).to_arrow()

    def remote_fields(self) -> dict[str, str]:
        """Returns a dictionary representing fields avaiable on the source.

        Keys are field names and values field types.
        """
        table = self.to_table()
        return {
            col.name: col.type for col in table.columns if col.name not in self.db_pk
        }

    def source_config(self) -> SourceConfig:
        """Returns a new SourceConfig.

        If fields are set, it will use them to generate `SourceColumn`s.
        Otherwise, it will use all fields in the source table except `self.db_pk`.
        """
        columns = (
            SourceColumn(name=col_name, type=str(col_type))
            for col_name, col_type in self.remote_fields().items()
        )
        if self.fields:
            columns = (c for c in columns if c.name in self.fields)

        return SourceConfig(
            address=self.address,
            db_pk=self.db_pk,
            columns=columns,
        )
