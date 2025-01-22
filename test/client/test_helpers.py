import logging
from unittest.mock import Mock, patch

import pytest
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame
from sqlalchemy import Engine

from matchbox import match, process, query
from matchbox.client.clean import company_name, company_number
from matchbox.client.helpers import cleaner, cleaners, comparison, select
from matchbox.client.helpers.selector import Match
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


def test_query_no_resolution():
    # TODO
    pass


def test_query_limit():
    # TODO
    pass


def test_query_multiple_sources():
    # TODO
    pass


def test_query_multiple_engines():
    # TODO
    pass


def test_query_return_type():
    # TODO
    pass


def test_query_threshold():
    # TODO
    pass


def test_index():
    # TODO
    pass


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
