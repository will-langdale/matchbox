from src.locations import DATA_SUBDIR

import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text as sql_text

import os
from contextlib import closing

DEFAULT_BATCH_SIZE = 10000
DEFAULT_DF_FORMAT = "fea"  # can be switched to csv


sql_engine = sqlalchemy.create_engine("postgresql://")


def query(sql, params=None):
    """
    Read full results set from Data Workspace based on arbitrary query

    Parameters:
        sql: a valid Postgres-SQL query
        params: a dictionary of parameters to format the SQL string
            See https://docs.sqlalchemy.org/en/14/core/tutorial.html#using-textual-sql

    Returns:
        df: pandas dataframe read from Data Workspace
    """

    with sql_engine.connect() as connection:
        return pd.read_sql(sql_text(sql), connection, params=params)


def query_iter(sql, params=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Read lazily (in chunks) from Data Workspace based on arbitrary query.
    Usage:
    ```
    sql_query = ...
    with query_iter(sql_query) as qiter:
        for chunk in qiter:
            print(chunk)
    ```

    Parameters:
        sql: a valid Postgres-SQL query
        params: a dictionary of parameters to format the SQL string
            See https://docs.sqlalchemy.org/en/14/core/tutorial.html#using-textual-sql
        batch_size: the number of rows to be read at a time

    Returns:
        a context manager yielding results in batch
    """

    def query_iter_logic():
        # The server-side cursor allows to read individual chunks through a generator
        # (see SQLAlchemy documentation)
        with sql_engine.connect().execution_options(
            stream_results=True
        ) as server_side_cursor:
            yield from pd.read_sql(
                sql_text(sql), server_side_cursor, params=params, chunksize=batch_size
            )

    # This returns a context manager that closes the connection at the end of the block
    return closing(query_iter_logic())


def query_nonreturn(sql, params=None):
    """
    Execute an arbitrary query and don't return the result

    Parameters:
        sql: a valid Postgres-SQL query
        params: a dictionary of parameters to format the SQL string
            See https://docs.sqlalchemy.org/en/14/core/tutorial.html#using-textual-sql

    """
    with sql_engine.begin() as connection:
        connection.execute(sql_text(sql), params)


def dataset(dataset_name):
    """
    Read dataset from Data Workspace

    Parameters:
        dataset_name: specified in the format 'schema.table'

    Returns:
        df: pandas dataframe read from Data Workspace
    """
    sql = f"SELECT * FROM {dataset_name};"
    return query(sql)


def dataset_iter(dataset_name, batch_size=DEFAULT_BATCH_SIZE):
    """
    Read dataset lazily (in chunks) from Data Workspace.
    Usage:
    ```
    dataset_name = ...
    with dataset_iter(dataset_name) as diter:
        for chunk in diter:
            print(chunk)
    ```

    Parameters:
        dataset_name: specified in the format 'schema.table'
        batch_size: the number of rows to be read at a time

    Returns:
        a context manager yielding results in batch
    """
    sql = f"SELECT * FROM {dataset_name};"
    return query_iter(sql, batch_size=batch_size)


def data_workspace_write(schema, table, df, if_exists="fail", index=False, **kwargs):
    """
    Persist dataset as table in Data Workspace.

    Parameters:
        schema: (str) the name of the schema where 'table' will be held
        table: (str) the name of the table to use
        df: (Pandas DataFrame) the data to be stored in the table
        if_exists: (str) argument that will be passed to the underlying
            Pandas function. See Pandas docs for DataFrame.to_sql
        index: (bool) whether to write the index of the dataframe
        **kwargs: all other key-value arguments will be passed to the
            pandas.to_sql function

    NOTE: this function will fail if the DataFrame has individual cells
    that contain complex objects (e.g. numpy arrays). In this case, it may
    be possible to convert the cells to another format (e.g. a list-of-lists,
    or a JSON object)
    """
    with sql_engine.connect() as connection:
        df.to_sql(
            table,
            con=connection,
            schema=schema,
            index=index,
            if_exists=if_exists,
            **kwargs,
        )


def _get_df_path(data_subdir, ds_name, extension=DEFAULT_DF_FORMAT):
    if data_subdir not in DATA_SUBDIR:
        raise ValueError("The data location specified is invalid")

    return os.path.join(DATA_SUBDIR[data_subdir], ds_name + "." + extension)


def persist_df(df, data_subdir, ds_name, extension=DEFAULT_DF_FORMAT):
    """
    Store a dataframe in one of the data folders

    Raises:
        ValueError: when an unsupported extension is specified

    Parameters:
        df: pandas dataframe to be persisted
        data_subdir: subfolder within /data to use, e.g. "raw"
        ds_name: extension-less name of the file to be stored
        extension: Whether the file to be written is a feather or csv file
    """

    file_path = _get_df_path(data_subdir, ds_name, extension)
    if extension == "fea":
        df.to_feather(file_path)
    elif extension == "csv":
        df.to_csv(file_path, index=False)
    else:
        raise ValueError("The format specified is not supported")


def load_df(data_subdir, ds_name, extension=DEFAULT_DF_FORMAT, **kwargs):
    """
    Reads a dataframe from one of the data folders

    Parameters:
        data_subdir: subfolder within /data to use, e.g. "raw"
        ds_name: extension-less name of the file to be read
        extension: Whether the file to be read is a feather or csv file
        **kwargs: keyword args passed to the underlying pandas function

    Raises:
        ValueError: when an unsupported extension is specified

    Returns:
        pandas dataframe read from disk
    """

    file_path = _get_df_path(data_subdir, ds_name, extension)

    if extension == "fea":
        return pd.read_feather(file_path, **kwargs)
    if extension == "csv":
        return pd.read_csv(file_path, low_memory=False, **kwargs)

    raise ValueError("The format specified is not supported")
