from typing import Iterable

import pytest
from sqlalchemy import text

from matchbox.server.postgresql.benchmark.generate_tables import (
    generate_all_tables,
    generate_result_tables,
)
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


@pytest.mark.parametrize(
    "left_ids, right_ids, next_id, n_components, n_probs",
    (
        [range(10_000), None, 10_000, 8000, 2000],
        [range(8000), range(8000, 16_000), 16_000, 6000, 10_000],
    ),
    ids=["dedupe", "link"],
)
def test_benchmark_result_tables(left_ids, right_ids, next_id, n_components, n_probs):
    resolution_id = None

    (top_clusters, _, _, _, _) = generate_result_tables(
        left_ids, right_ids, resolution_id, next_id, n_components, n_probs
    )

    assert len(top_clusters) == n_components


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
            df.to_sql(name=table_name, con=con, schema=schema)
