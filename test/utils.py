from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import TypeEngine


def dataset_parquet_to_local_warehouse(
    parquetpath: Path, schema: str, table: str, dtype: dict[str, TypeEngine]
) -> None:
    """Load dataset parquet file to local warehouse instance.

    Args:
        parquetpath: Path to parquet file.
        schema: Name of schema in local database.
        table: Name of new table in local database.
        dtype: Dictionary of dataset column names and sqlalchemy types.
    """

    user = "warehouse_user"
    password = "warehouse_password"
    host = "localhost"
    database = "warehouse"
    port = 7654

    engine = create_engine(
        f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
    )
    with engine.connect() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        connection.commit()

    dataset = pd.read_parquet(parquetpath, dtype_backend="pyarrow")
    dataset.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists="replace",
        dtype=dtype,
        chunksize=100000,
        index=False,
    )
    engine.dispose()


def company_name(column: str) -> str:
    """Returns DuckDB SQL for standard company name cleaning.

    * Lower case, remove punctuation & tokenise the company name into an array
    * Remove stopwords and expand abbreviations
    * Join back to a string

    Args:
        column: The name of the column to clean

    Returns:
        String containing DuckDB SQL for company name cleaning
    """

    # Hard-coded stopwords and abbreviations from the original code
    stopwords = [
        "limited",
        "uk",
        "company",
        "international",
        "group",
        "of",
        "the",
        "inc",
        "and",
        "plc",
        "corporation",
        "llp",
        "pvt",
        "gmbh",
        "u k",
        "pte",
        "usa",
        "bank",
        "b v",
        "bv",
    ]

    # Convert stopwords list to SQL array format
    stopwords_sql = "[" + ", ".join([f"'{word}'" for word in stopwords]) + "]"

    # Build the nested SQL transformation
    sql = f"""
    list_aggr(
        array_filter(
            array(
                select distinct unnest(
                    regexp_split_to_array(
                        trim(
                            regexp_replace(
                                regexp_replace(
                                    regexp_replace(
                                        lower(
                                            regexp_replace(
                                                regexp_replace(
                                                    {column},
                                                    '[^a-zA-Z0-9 ]+',
                                                    ' ',
                                                    'g'
                                                ),
                                                '[.]+',
                                                '',
                                                'g'
                                            )
                                        ),
                                        '\\b(co)\\b',
                                        'company',
                                        'g'
                                    ),
                                    '\\b(ltd)\\b',
                                    'limited',
                                    'g'
                                ),
                                '\\s+',
                                ' ',
                                'g'
                            )
                        ),
                        '[^a-zA-Z0-9]+'
                    )
                ) tokens
                order by tokens
            ),
            x -> not array_contains({stopwords_sql}, x)
        ),
        'string_agg',
        ' '
    )"""

    return sql


def company_number(column: str) -> str:
    """Returns DuckDB SQL for company number cleaning.

    Remove non-numbers, and then leading zeroes.

    Args:
        column: The name of the column to clean

    Returns:
        String containing DuckDB SQL for company number cleaning
    """

    sql = f"""
    regexp_replace(
        regexp_replace(
            {column},
            '[^0-9]',
            '',
            'g'
        ),
        '^0+',
        ''
    )"""

    return sql
