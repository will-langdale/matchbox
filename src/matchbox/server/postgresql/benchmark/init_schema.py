from textwrap import dedent

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    Sources,
)

if __name__ == "__main__":
    print(
        dedent("""
        DROP SCHEMA mb CASCADE;
        CREATE SCHEMA mb;   
    """)
    )

    # Order matters
    for table_class in (
        Resolutions,
        ResolutionFrom,
        Sources,
        Clusters,
        Contains,
        Probabilities,
    ):
        print(
            str(
                CreateTable(table_class.__table__).compile(dialect=postgresql.dialect())
            )
            + ";"
        )
