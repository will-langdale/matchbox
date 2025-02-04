import logging
from unittest.mock import Mock, patch

import pyarrow as pa
import pytest
import respx
from dotenv import find_dotenv, load_dotenv
from httpx import Response
from pandas import DataFrame
from sqlalchemy import Engine

from matchbox import index, match, process, query
from matchbox.client._handler import url
from matchbox.client.clean import company_name, company_number
from matchbox.client.helpers import cleaner, cleaners, comparison, select
from matchbox.client.helpers.selector import Match, Selector
from matchbox.common.arrow import SCHEMA_MB_IDS, table_to_buffer
from matchbox.common.dtos import BackendRetrievableType, NotFoundError
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.hash import hash_to_base64
from matchbox.common.sources import Source, SourceAddress

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)


def test_cleaners():
    cleaner_name = cleaner(function=company_name, arguments={"column": "company_name"})
    cleaner_number = cleaner(
        function=company_number, arguments={"column": "company_number"}
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    assert cleaner_name_number is not None


def test_process(
    warehouse_data: list[Source],
):
    crn = warehouse_data[0].to_arrow()

    cleaner_name = cleaner(
        function=company_name,
        arguments={"column": "test_crn_company_name"},
    )
    cleaner_number = cleaner(
        function=company_number,
        arguments={"column": "test_crn_crn"},
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    df_name_cleaned = process(data=crn, pipeline=cleaner_name_number)

    assert isinstance(df_name_cleaned, DataFrame)
    assert df_name_cleaned.shape[0] == 3000


def test_comparisons():
    comparison_name_id = comparison(
        sql_condition=(
            "l.company_name = r.company_name and l.data_hub_id = r.data_hub_id"
        )
    )

    assert comparison_name_id is not None


@patch("matchbox.server.base.BackendManager.get_backend")
def test_select_non_indexed_columns(get_backend: Mock, warehouse_engine: Engine):
    """Selecting columns not declared to backend generates warning."""
    source = Source(
        address=SourceAddress.compose(engine=warehouse_engine, full_name="test.foo"),
        db_pk="pk",
    )

    mock_backend = Mock()
    mock_backend.get_source = Mock(return_value=source)
    get_backend.return_value = mock_backend

    df = DataFrame([{"pk": 0, "a": 1, "b": "2"}, {"pk": 1, "a": 10, "b": "20"}])
    with warehouse_engine.connect() as conn:
        df.to_sql(
            name="foo",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )

    with pytest.warns(Warning):
        select({"test.foo": ["a", "b"]}, warehouse_engine)


@patch("matchbox.server.base.BackendManager.get_backend")
def test_select_missing_columns(get_backend: Mock, warehouse_engine: Engine):
    """Selecting columns not in the warehouse errors."""
    source = Source(
        address=SourceAddress.compose(engine=warehouse_engine, full_name="test.foo"),
        db_pk="pk",
    )

    mock_backend = Mock()
    mock_backend.get_source = Mock(return_value=source)
    get_backend.return_value = mock_backend

    df = DataFrame([{"pk": 0, "a": 1, "b": "2"}, {"pk": 1, "a": 10, "b": "20"}])
    with warehouse_engine.connect() as conn:
        df.to_sql(
            name="foo",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )
    with pytest.raises(ValueError):
        select({"test.foo": ["a", "c"]}, warehouse_engine)


def test_query_no_resolution_fail():
    """Querying with multiple selectors and no resolution is not allowed."""
    sels = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="foo",
                    warehouse_hash="bar",
                ),
                db_pk="i",
            ),
            fields=["a", "b"],
        ),
        Selector(
            source=Source(
                address=SourceAddress(full_name="foo2", warehouse_hash="bar2"),
                db_pk="j",
            ),
            fields=["x", "y"],
        ),
    ]

    with pytest.raises(ValueError, match="resolution name"):
        query(sels)


@respx.mock
def test_query_no_resolution_ok_various_params():
    """Tests that we can avoid passing resolution name, with a variety of parameters."""
    with (
        patch("matchbox.server.base.BackendManager.get_backend") as get_backend,
        patch.object(Source, "to_arrow") as to_arrow,
    ):
        # Mock backend (temporary)
        mock_backend = Mock()
        get_resolution_id = Mock(return_value=42)
        mock_backend.get_resolution_id = get_resolution_id
        get_backend.return_value = mock_backend

        # Mock API
        query_route = respx.get(url("/query")).mock(
            return_value=Response(
                200,
                content=table_to_buffer(
                    pa.Table.from_pylist(
                        [
                            {"source_pk": "0", "id": 1},
                            {"source_pk": "1", "id": 2},
                        ],
                        schema=SCHEMA_MB_IDS,
                    )
                ).read(),
            )
        )

        # Mock `Source.to_arrow`
        to_arrow.return_value = pa.Table.from_pandas(
            DataFrame(
                [
                    {"foo_pk": "0", "foo_a": 1, "foo_b": "2"},
                    {"foo_pk": "1", "foo_a": 10, "foo_b": "20"},
                ]
            )
        )

        # Well-formed selector for these mocks
        sels = [
            Selector(
                source=Source(
                    address=SourceAddress(
                        full_name="foo",
                        warehouse_hash=b"bar",
                    ),
                    db_pk="pk",
                ),
                fields=["a", "b"],
            )
        ]

        # Tests with no optional params
        results = query(sels)
        assert len(results) == 2
        assert {"foo_a", "foo_b", "id"} == set(results.columns)

        get_resolution_id.assert_not_called()
        assert dict(query_route.calls.last.request.url.params) == {
            "full_name": sels[0].source.address.full_name,
            "warehouse_hash_b64": hash_to_base64(sels[0].source.address.warehouse_hash),
        }
        to_arrow.assert_called_once()
        assert set(to_arrow.call_args.kwargs["fields"]) == {"a", "b"}
        assert set(to_arrow.call_args.kwargs["pks"]) == {"0", "1"}

        # Tests with optional params
        results = query(sels, return_type="arrow", threshold=0.5, limit=2).to_pandas()
        assert len(results) == 2
        assert {"foo_a", "foo_b", "id"} == set(results.columns)

        assert dict(query_route.calls.last.request.url.params) == {
            "full_name": sels[0].source.address.full_name,
            "warehouse_hash_b64": hash_to_base64(sels[0].source.address.warehouse_hash),
            "threshold": "0.5",
            "limit": "2",
        }


@respx.mock
def test_query_multiple_sources_with_limits():
    """Tests that we can query multiple sources and distribute the limit among them."""
    with (
        patch("matchbox.server.base.BackendManager.get_backend") as get_backend,
        patch.object(Source, "to_arrow") as to_arrow,
    ):
        # Mock backend (temporary)
        mock_backend = Mock()
        get_resolution_id = Mock(return_value=42)
        mock_backend.get_resolution_id = get_resolution_id
        get_backend.return_value = mock_backend

        # Mock API
        query_route = respx.get(url("/query")).mock(
            side_effect=[
                Response(
                    200,
                    content=table_to_buffer(
                        pa.Table.from_pylist(
                            [
                                {"source_pk": "0", "id": 1},
                                {"source_pk": "1", "id": 2},
                            ],
                            schema=SCHEMA_MB_IDS,
                        )
                    ).read(),
                ),
                Response(
                    200,
                    content=table_to_buffer(
                        pa.Table.from_pylist(
                            [
                                {"source_pk": "2", "id": 1},
                                {"source_pk": "3", "id": 2},
                            ],
                            schema=SCHEMA_MB_IDS,
                        )
                    ).read(),
                ),
            ]
            * 2  # 2 calls to `query()` in this test, each querying server twice
        )

        # Mock `Source.to_arrow`
        to_arrow.side_effect = [
            pa.Table.from_pandas(
                DataFrame(
                    [
                        {"foo_pk": "0", "foo_a": 1, "foo_b": "2"},
                        {"foo_pk": "1", "foo_a": 10, "foo_b": "20"},
                    ]
                )
            ),
            pa.Table.from_pandas(
                DataFrame(
                    [
                        {"foo2_pk": "2", "foo2_c": "val"},
                        {"foo2_pk": "3", "foo2_c": "val"},
                    ]
                )
            ),
        ] * 2  # 2 calls to `query()` in this test, each dealing with 2 sources

        # Well-formed select from these mocks
        sels = [
            Selector(
                source=Source(
                    address=SourceAddress(
                        full_name="foo",
                        warehouse_hash="bar",
                    ),
                    db_pk="pk",
                ),
                fields=["a", "b"],
            ),
            Selector(
                source=Source(
                    address=SourceAddress(full_name="foo2", warehouse_hash="bar2"),
                    db_pk="pk",
                ),
                fields=["c"],
            ),
        ]

        # Validate results
        results = query(sels, resolution_name="link", limit=7)
        assert len(results) == 4
        assert {"foo_a", "foo_b", "foo2_c", "id"} == set(results.columns)

        get_resolution_id.assert_called_with("link")
        assert dict(query_route.calls[-2].request.url.params) == {
            "full_name": sels[0].source.address.full_name,
            "warehouse_hash_b64": hash_to_base64(sels[0].source.address.warehouse_hash),
            "resolution_id": "42",
            "limit": "4",
        }
        assert dict(query_route.calls[-1].request.url.params) == {
            "full_name": sels[1].source.address.full_name,
            "warehouse_hash_b64": hash_to_base64(sels[1].source.address.warehouse_hash),
            "resolution_id": "42",
            "limit": "3",
        }

        # It also works with the selectors specified separately
        query([sels[0]], [sels[1]], resolution_name="link", limit=7)


@respx.mock
@patch("matchbox.server.base.BackendManager.get_backend")
def test_query_404_resolution(get_backend: Mock):
    # Mock backend (temporary)
    mock_backend = Mock()
    get_resolution_id = Mock(return_value=42)
    mock_backend.get_resolution_id = get_resolution_id
    get_backend.return_value = mock_backend

    # Mock API
    respx.get(url("/query")).mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Resolution 42 not found",
                entity=BackendRetrievableType.RESOLUTION,
            ).model_dump(),
        )
    )

    # Well-formed selector for these mocks
    sels = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="foo",
                    warehouse_hash="bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    # Tests with no optional params
    with pytest.raises(MatchboxResolutionNotFoundError, match="42"):
        query(sels)


@respx.mock
@patch("matchbox.server.base.BackendManager.get_backend")
def test_query_404_source(get_backend: Mock):
    # Mock backend (temporary)
    mock_backend = Mock()
    get_resolution_id = Mock(return_value=42)
    mock_backend.get_resolution_id = get_resolution_id
    get_backend.return_value = mock_backend

    # Mock API
    respx.get(url("/query")).mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Resolution 42 not found",
                entity=BackendRetrievableType.SOURCE,
            ).model_dump(),
        )
    )

    # Well-formed selector for these mocks
    sels = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="foo",
                    warehouse_hash="bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    # Tests with no optional params
    with pytest.raises(MatchboxSourceNotFoundError, match="42"):
        query(sels)


@patch("matchbox.server.base.BackendManager.get_backend")
def test_index_default(get_backend: Mock, warehouse_engine: Engine):
    """Test the index function with default columns."""
    # Setup
    mock_backend = Mock()
    get_backend.return_value = mock_backend
    # Mock Source methods
    mock_source = Mock(spec=Source)
    # Ensure each method returns the same mock object
    mock_source.set_engine.return_value = mock_source
    mock_source.default_columns.return_value = mock_source
    mock_source.hash_data.return_value = "test_hash"
    with patch("matchbox.client.helpers.index.Source", return_value=mock_source):
        # Execute
        index("test.table", "id", engine=warehouse_engine)
        # Assert
        mock_backend.index.assert_called_once_with(mock_source, "test_hash")
        mock_source.set_engine.assert_called_once_with(warehouse_engine)
        mock_source.default_columns.assert_called_once()


@patch("matchbox.server.base.BackendManager.get_backend")
def test_index_list(get_backend: Mock, warehouse_engine: Engine):
    """Test the index function with a list of columns."""
    # Setup
    mock_backend = Mock()
    get_backend.return_value = mock_backend
    columns = ["name", "age"]
    # Mock Source methods
    mock_source = Mock(spec=Source)
    # Ensure each method returns the same mock object
    mock_source.set_engine.return_value = mock_source
    mock_source.hash_data.return_value = "test_hash"
    with patch("matchbox.client.helpers.index.Source", return_value=mock_source):
        # Execute
        index("test.table", "id", engine=warehouse_engine, columns=columns)
        # Assert
        mock_backend.index.assert_called_once_with(mock_source, "test_hash")
        mock_source.set_engine.assert_called_once_with(warehouse_engine)
        mock_source.default_columns.assert_not_called()


@patch("matchbox.server.base.BackendManager.get_backend")
def test_index_dict(get_backend: Mock, warehouse_engine: Engine):
    """Test the index function with a dictionary of columns."""
    # Setup
    mock_backend = Mock()
    get_backend.return_value = mock_backend
    columns = [
        {"name": "name", "alias": "person_name", "type": "TEXT"},
        {"name": "age", "alias": "person_age", "type": "BIGINT"},
    ]
    # Mock Source methods
    mock_source = Mock(spec=Source)
    # Ensure each method returns the same mock object
    mock_source.set_engine.return_value = mock_source
    mock_source.hash_data.return_value = "test_hash"
    with patch("matchbox.client.helpers.index.Source", return_value=mock_source):
        # Execute
        index("test.table", "id", engine=warehouse_engine, columns=columns)
        # Assert
        mock_backend.index.assert_called_once_with(mock_source, "test_hash")
        mock_source.set_engine.assert_called_once_with(warehouse_engine)
        mock_source.default_columns.assert_not_called()


@patch("matchbox.server.base.BackendManager.get_backend")
def test_match_calls_backend(get_backend: Mock):
    """The client can perform the right call for matching."""
    mock_backend = Mock()
    mock_backend.match = Mock(
        return_value=Match(
            cluster=1,
            source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
            source_id={"a"},
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
            target_id={"b"},
        )
    )
    get_backend.return_value = mock_backend

    res = match("pk1", "test.source", "test.target", resolution="foo")
    assert isinstance(res, Match)
