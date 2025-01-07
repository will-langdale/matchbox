from textwrap import dedent

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    Sources,
)


def empty_schema() -> str:
    schema = MBDB.MatchboxBase.metadata.schema
    sql = dedent(f"""
        DROP SCHEMA IF EXISTS {schema} CASCADE;
        CREATE SCHEMA {schema};   
    """)

    return sql


def create_tables() -> str:
    sql = ""
    # Order matters
    for table_class in (
        Resolutions,
        ResolutionFrom,
        Sources,
        Clusters,
        Contains,
        Probabilities,
    ):
        sql += (
            str(
                CreateTable(table_class.__table__).compile(dialect=postgresql.dialect())
            )
            + "; \n"
        )

    return sql


if __name__ == "__main__":
    print(empty_schema())
    print(create_tables())
