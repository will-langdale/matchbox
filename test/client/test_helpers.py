import logging
from unittest.mock import Mock, call, patch

import pyarrow as pa
import pytest
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame
from sqlalchemy import Engine

from matchbox import index, match, process, query
from matchbox.client.clean import company_name, company_number
from matchbox.client.helpers import cleaner, cleaners, comparison, select
from matchbox.client.helpers.selector import Match, Selector
from matchbox.common.sources import Source, SourceAddress
from matchbox.server.postgresql import MatchboxPostgres

from ..fixtures.db import AddIndexedDataCallable

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
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
):
    # Setup
    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    crn_source = warehouse_data[0]

    crn = query(
        select(
            {crn_source.address.full_name: ["crn", "company_name"]},
            engine=crn_source.engine,
        ),
        return_type="pandas",
    )

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


def test_select_non_indexed_columns(warehouse_engine: Engine):
    with patch("matchbox.server.base.BackendManager.get_backend") as get_backend:
        source = Source(
            address=SourceAddress.compose(
                engine=warehouse_engine, full_name="test.foo"
            ),
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


def test_query_no_resolution_ok_various_params():
    """Tests that we can avoid passing resolution name, with a variety of parameters."""
    with (
        patch("matchbox.server.base.BackendManager.get_backend") as get_backend,
        patch.object(Source, "to_arrow") as to_arrow,
    ):
        # Mock backend's `get_resolution` and `query`
        mock_backend = Mock()

        get_resolution_id = Mock(return_value=42)
        mock_backend.get_resolution_id = get_resolution_id

        query_mock = Mock(
            return_value=pa.Table.from_arrays(
                [pa.array([0, 1]), pa.array([10, 11])],
                names=["source_pk", "final_parent"],
            )
        )
        mock_backend.query = query_mock

        get_backend.return_value = mock_backend

        # Mock `Source.to_arrow`
        to_arrow.return_value = pa.Table.from_pandas(
            DataFrame(
                [
                    {"foo_pk": 0, "foo_a": 1, "foo_b": "2"},
                    {"foo_pk": 1, "foo_a": 10, "foo_b": "20"},
                ]
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
        results = query(sels)
        assert len(results) == 2
        assert {"foo_a", "foo_b"} == set(results.columns)

        get_resolution_id.assert_not_called()
        query_mock.assert_called_once_with(
            source_address=sels[0].source.address,
            resolution_id=None,
            threshold=None,
            limit=None,
        )
        to_arrow.assert_called_once()
        assert set(to_arrow.call_args.kwargs["fields"]) == {"a", "b"}
        assert set(to_arrow.call_args.kwargs["pks"]) == {0, 1}

        # Tests with optional params
        results = query(sels, return_type="arrow", threshold=0.5, limit=2).to_pandas()
        assert len(results) == 2
        assert {"foo_a", "foo_b"} == set(results.columns)

        query_mock.assert_called_with(
            source_address=sels[0].source.address,
            resolution_id=None,
            threshold=0.5,
            limit=2,
        )


def test_query_multiple_sources_with_limits():
    """Tests that we can query multiple sources and distribute the limit among them"""
    with (
        patch("matchbox.server.base.BackendManager.get_backend") as get_backend,
        patch.object(Source, "to_arrow") as to_arrow,
    ):
        # Mock backend's `get_resolution` and `query`
        mock_backend = Mock()

        get_resolution_id = Mock(return_value=42)
        mock_backend.get_resolution_id = get_resolution_id

        query_mock = Mock(
            side_effect=[
                pa.Table.from_arrays(
                    [pa.array([0, 1]), pa.array([10, 11])],
                    names=["source_pk", "final_parent"],
                ),
                pa.Table.from_arrays(
                    [pa.array([2, 3]), pa.array([10, 11])],
                    names=["source_pk", "final_parent"],
                ),
            ]
            * 2  # 2 calls to `query()` in this test, each querying server twice
        )
        mock_backend.query = query_mock

        get_backend.return_value = mock_backend

        # Mock `Source.to_arrow`
        to_arrow.side_effect = [
            pa.Table.from_pandas(
                DataFrame(
                    [
                        {"foo_pk": 0, "foo_a": 1, "foo_b": "2"},
                        {"foo_pk": 1, "foo_a": 10, "foo_b": "20"},
                    ]
                )
            ),
            pa.Table.from_pandas(
                DataFrame(
                    [
                        {"foo2_pk": 2, "foo2_c": "val"},
                        {"foo2_pk": 3, "foo2_c": "val"},
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
        assert {"foo_a", "foo_b", "foo2_c"} == set(results.columns)

        get_resolution_id.assert_called_with("link")
        assert query_mock.call_args_list[0] == call(
            source_address=sels[0].source.address,
            resolution_id=42,
            threshold=None,
            limit=4,
        )
        assert query_mock.call_args_list[1] == call(
            source_address=sels[1].source.address,
            resolution_id=42,
            threshold=None,
            limit=3,
        )

        # It also works with the selectors specified separately
        query([sels[0]], [sels[1]], resolution_name="link", limit=7)


def test_index_default(warehouse_engine: Engine):
    """Test the index function with default columns."""
    with patch("matchbox.server.base.BackendManager.get_backend") as get_backend:
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


def test_index_list(warehouse_engine: Engine):
    """Test the index function with a list of columns."""
    with patch("matchbox.server.base.BackendManager.get_backend") as get_backend:
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


def test_index_dict(warehouse_engine: Engine):
    """Test the index function with a dictionary of columns."""
    with patch("matchbox.server.base.BackendManager.get_backend") as get_backend:
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
