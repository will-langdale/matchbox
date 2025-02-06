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
from matchbox.common.sources import Source, SourceAddress, SourceColumn

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
def test_select_mixed_style(get_backend: Mock, warehouse_engine: Engine):
    """We can select select specific columns from some of the sources"""
    # Set up mocks and test data
    source1 = Source(
        address=SourceAddress.compose(engine=warehouse_engine, full_name="test.foo"),
        db_pk="pk",
        columns=[SourceColumn(name="a", type="BIGINT")],
    )
    source2 = Source(
        address=SourceAddress.compose(engine=warehouse_engine, full_name="test.bar"),
        db_pk="pk",
    )

    mock_backend = Mock()
    mock_backend.get_source = Mock(side_effect=[source1, source2])
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

        df.to_sql(
            name="bar",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )

    # Select sources
    selection = select({"test.foo": ["a"]}, "test.bar", engine=warehouse_engine)

    # Check they contain what we expect
    assert selection[0].fields == ["a"]
    assert not selection[1].fields
    assert selection[0].source == source1
    assert selection[1].source == source2


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
        select({"test.foo": ["a", "b"]}, engine=warehouse_engine)


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
        select({"test.foo": ["a", "c"]}, engine=warehouse_engine)


def test_query_no_resolution_fail():
    """Querying with multiple selectors and no resolution is not allowed."""
    sels = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="foo",
                    warehouse_hash=b"bar",
                ),
                db_pk="i",
            ),
            fields=["a", "b"],
        ),
        Selector(
            source=Source(
                address=SourceAddress(full_name="foo2", warehouse_hash=b"bar2"),
                db_pk="j",
            ),
            fields=["x", "y"],
        ),
    ]

    with pytest.raises(ValueError, match="resolution name"):
        query(sels)


@respx.mock
@patch.object(Source, "to_arrow")
def test_query_no_resolution_ok_various_params(to_arrow: Mock):
    """Tests that we can avoid passing resolution name, with a variety of parameters."""
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

    assert dict(query_route.calls.last.request.url.params) == {
        "full_name": sels[0].source.address.full_name,
        "warehouse_hash_b64": hash_to_base64(sels[0].source.address.warehouse_hash),
    }
    to_arrow.assert_called_once()
    assert set(to_arrow.call_args.kwargs["fields"]) == {"a", "b"}
    assert set(to_arrow.call_args.kwargs["pks"]) == {"0", "1"}

    # Tests with optional params
    results = query(sels, return_type="arrow", threshold=50, limit=2).to_pandas()
    assert len(results) == 2
    assert {"foo_a", "foo_b", "id"} == set(results.columns)

    assert dict(query_route.calls.last.request.url.params) == {
        "full_name": sels[0].source.address.full_name,
        "warehouse_hash_b64": hash_to_base64(sels[0].source.address.warehouse_hash),
        "threshold": "50",
        "limit": "2",
    }


@respx.mock
@patch.object(Source, "to_arrow")
def test_query_multiple_sources_with_limits(to_arrow: Mock):
    """Tests that we can query multiple sources and distribute the limit among them."""
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
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
        ),
        Selector(
            source=Source(
                address=SourceAddress(full_name="foo2", warehouse_hash=b"bar2"),
                db_pk="pk",
            ),
            fields=["c"],
        ),
    ]

    # Validate results
    results = query(sels, resolution_name="link", limit=7)
    assert len(results) == 4
    assert {
        # All columns automatically selected for `foo`
        "foo_pk",
        "foo_a",
        "foo_b",
        # Only one column selected for `foo2`
        "foo2_c",
        # The id always comes back
        "id",
    } == set(results.columns)

    assert dict(query_route.calls[-2].request.url.params) == {
        "full_name": sels[0].source.address.full_name,
        "warehouse_hash_b64": hash_to_base64(sels[0].source.address.warehouse_hash),
        "resolution_name": "link",
        "limit": "4",
    }
    assert dict(query_route.calls[-1].request.url.params) == {
        "full_name": sels[1].source.address.full_name,
        "warehouse_hash_b64": hash_to_base64(sels[1].source.address.warehouse_hash),
        "resolution_name": "link",
        "limit": "3",
    }

    # It also works with the selectors specified separately
    query([sels[0]], [sels[1]], resolution_name="link", limit=7)


@respx.mock
def test_query_404_resolution():
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
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    # Test with no optional params
    with pytest.raises(MatchboxResolutionNotFoundError, match="42"):
        query(sels)


@respx.mock
def test_query_404_source():
    # Mock API
    respx.get(url("/query")).mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Source 42 not found",
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
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    # Test with no optional params
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


@respx.mock
def test_match_ok():
    """The client can perform the right call for matching."""
    # Set up mocks
    mock_match1 = Match(
        cluster=1,
        source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
        source_id={"a"},
        target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
        target_id={"b"},
    )
    mock_match2 = Match(
        cluster=1,
        source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
        source_id={"a"},
        target=SourceAddress(full_name="test.target2", warehouse_hash=b"bar"),
        target_id={"b"},
    )
    # The standard JSON serialiser does not handle Pydantic objects
    serialised_matches = (
        f"[{mock_match1.model_dump_json()}, {mock_match2.model_dump_json()}]"
    )

    match_route = respx.get(url("/match")).mock(
        return_value=Response(200, content=serialised_matches)
    )

    # Use match function
    source = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.source",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]
    target1 = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.target1",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    target2 = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.target2",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    res = match(
        target1,
        target2,
        source=source,
        source_pk="pk1",
        resolution_name="foo",
    )

    # Verify results
    assert len(res) == 2
    assert isinstance(res[0], Match)
    param_set = sorted(match_route.calls.last.request.url.params.multi_items())
    assert param_set == sorted(
        [
            ("target_full_names", "test.target1"),
            ("target_full_names", "test.target2"),
            ("target_warehouse_hashes_b64", hash_to_base64(b"bar")),
            ("target_warehouse_hashes_b64", hash_to_base64(b"bar")),
            ("source_full_name", "test.source"),
            ("source_warehouse_hash_b64", hash_to_base64(b"bar")),
            ("source_pk", "pk1"),
            ("resolution_name", "foo"),
        ]
    )


@respx.mock
def test_match_404_resolution():
    """The client can handle a resolution not found error."""
    # Set up mocks
    respx.get(url("/match")).mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Resolution 42 not found",
                entity=BackendRetrievableType.RESOLUTION,
            ).model_dump(),
        )
    )

    # Use match function
    source = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.source",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]
    target = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.target1",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    with pytest.raises(MatchboxResolutionNotFoundError, match="42"):
        match(
            target,
            source=source,
            source_pk="pk1",
            resolution_name="foo",
        )


@respx.mock
def test_match_404_source():
    """The client can handle a source not found error."""
    # Set up mocks
    respx.get(url("/match")).mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Source 42 not found",
                entity=BackendRetrievableType.SOURCE,
            ).model_dump(),
        )
    )

    # Use match function
    source = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.source",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]
    target = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.target1",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    with pytest.raises(MatchboxSourceNotFoundError, match="42"):
        match(
            target,
            source=source,
            source_pk="pk1",
            resolution_name="foo",
        )
