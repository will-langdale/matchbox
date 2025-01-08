from typing import Iterable

from sqlalchemy import text

from matchbox.server.postgresql.benchmark.generate_tables import generate_all_tables
from matchbox.server.postgresql.benchmark.init_schema import create_tables, empty_schema
from matchbox.server.postgresql.db import MBDB


def test_benchmark_init_schema():
    schema = MBDB.MatchboxBase.metadata.schema
    count_tables = text(f"""
        select count(*)
        from information_schema.tables
        where table_schema = '{schema}';
    """)

    with MBDB.get_engine().connect() as con:
        con.execute(text(empty_schema()))
        con.commit()
        n_tables = int(con.execute(count_tables).scalar())
        assert n_tables == 0

        con.execute(text(create_tables()))
        con.commit()
        n_tables_expected = len(MBDB.MatchboxBase.metadata.tables)
        n_tables = int(con.execute(count_tables).scalar())
        assert n_tables == n_tables_expected


def test_benchmark_generate_tables():
    schema = MBDB.MatchboxBase.metadata.schema

    def array_encode(array: Iterable[str]):
        if not array:
            return None
        escaped_l = [f'"{s}"' for s in array]
        list_rep = ", ".join(escaped_l)
        return "{" + list_rep + "}"

    with MBDB.get_engine().connect() as con:
        con.execute(text(empty_schema()))
        con.commit()

        results = generate_all_tables(20, 5, 25, 5, 25)

        assert len(results) == len(MBDB.MatchboxBase.metadata.tables)

        for table_name, table_arrow in results.items():
            df = table_arrow.to_pandas()
            # Pandas' `to_sql` dislikes arrays
            if "source_pk" in df.columns:
                df["source_pk"] = df["source_pk"].apply(array_encode)
            # Pandas' `to_sql` dislikes large unsigned ints
            for c in df.columns:
                if df[c].dtype == "uint64":
                    df[c] = df[c].astype("int64")
            df.to_sql(table_name, con, schema)
