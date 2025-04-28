import polars as pl
import pyarrow as pa
from httpx import Response
from polars.testing import assert_frame_equal
from respx import MockRouter
from sqlalchemy import Engine, create_engine

from matchbox.client.extract import primary_keys_map
from matchbox.common.arrow import SCHEMA_MB_IDS, table_to_buffer
from matchbox.common.factories.sources import source_from_tuple


def test_sql_interface(
    sqlite_warehouse: Engine,
    matchbox_api: MockRouter,
):
    # Make dummy data
    foo = source_from_tuple(
        full_name="foo",
        engine=sqlite_warehouse,
        data_pks=[1, 2, 3],
        data_tuple=({"col": 0}, {"col": 1}, {"col": 2}),
    )
    bar = source_from_tuple(
        full_name="bar",
        engine=sqlite_warehouse,
        data_pks=["a", "b", "c"],
        data_tuple=({"col": 10}, {"col": 11}, {"col": 12}),
    )
    different_engine_source = source_from_tuple(
        full_name="baz",
        engine=create_engine("sqlite:///:memory:"),
        data_pks=["a", "b", "c"],
        data_tuple=({"col": 10}, {"col": 11}, {"col": 12}),
    )

    foo.to_warehouse(sqlite_warehouse)
    bar.to_warehouse(sqlite_warehouse)

    # Because of FULL OUTER JOIN, we expect some values to be null, and some explosions
    expected_mapping = pl.DataFrame(
        [
            {"id": 1, "foo_pk": "1", "bar_pk": "a"},
            {"id": 2, "foo_pk": "2", "bar_pk": None},
            {"id": 3, "foo_pk": "3", "bar_pk": "b"},
            {"id": 3, "foo_pk": "3", "bar_pk": "c"},
        ]
    )

    # Mock API
    matchbox_api.get("/sources").mock(
        return_value=Response(
            200,
            json=[
                foo.source.model_dump(),
                bar.source.model_dump(),
                different_engine_source.source.model_dump(),
            ],
        )
    )

    # We don't expect a query call for the third source
    matchbox_api.get("/query", params={"full_name": "foo"}).mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"source_pk": "1", "id": 1},
                        {"source_pk": "2", "id": 2},
                        {"source_pk": "3", "id": 3},
                    ],
                    schema=SCHEMA_MB_IDS,
                )
            ).read(),
        )
    )

    matchbox_api.get("/query", params={"full_name": "bar"}).mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"source_pk": "a", "id": 1},
                        {"source_pk": "b", "id": 3},
                        {"source_pk": "c", "id": 3},
                    ],
                    schema=SCHEMA_MB_IDS,
                )
            ).read(),
        )
    )

    # Call extracting method
    mapping = primary_keys_map(resolution_name="companies", engine=sqlite_warehouse)

    assert_frame_equal(
        pl.from_arrow(mapping),
        expected_mapping,
        check_row_order=False,
        check_column_order=False,
    )
