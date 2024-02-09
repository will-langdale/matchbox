from test.fixtures.models import linker_test_params, merge_test_params

import pytest
from pandas import DataFrame

from cmf import make_linker


@pytest.mark.parametrize("data", merge_test_params)
@pytest.mark.parametrize("linker", linker_test_params)
def test_linkers(
    # Fixtures
    db_engine,
    db_clear_models,
    db_add_dedupe_models,
    # Parameterised data classes
    data,
    linker,
    # Pytest
    request,
):
    """Runs all linker methodologies over exemplar tables.

    Tests:
        1. That the input data is as expected
        2. That the data is linked correctly
        3. That the linked probabilities are inserted correctly
        4. That the correct number of clusters are resolved
        5. That the resolved clusters are inserted correctly
    """
    # i. Ensure database is clean, collect fixtures

    db_clear_models(db_engine)
    db_add_dedupe_models(db_engine, request)

    df_l = request.getfixturevalue(data.fixture_l)
    df_r = request.getfixturevalue(data.fixture_r)

    # 1. Input data is as expected

    assert isinstance(df_l, DataFrame)
    assert df_l.shape[0] == data.curr_n_l

    assert isinstance(df_r, DataFrame)
    assert df_r.shape[0] == data.curr_n_r

    # 2. Data is linked correctly

    linker_name = f"{linker.name}_{data.source_l}_{data.source_r}"
    linker_settings = linker.build_settings(data)

    linker = make_linker(
        link_run_name=linker_name,
        description=(
            f"Testing link of {data.source_l} and {data.source_r} "
            f"with {linker.name} method."
        ),
        linker=linker.cls,
        linker_settings=linker_settings,
        left_data=df_l,
        left_source=data.source_l,
        right_data=df_r,
        right_source=data.source_r,
    )

    linked = linker()

    linked_df = linked.to_df()
    linked_df_with_source = linked.inspect_with_source(
        left_data=df_l,
        left_key="cluster_sha1",
        right_data=df_r,
        right_key="cluster_sha1",
    )

    # print(linked_df_with_source.drop(columns=["cluster_sha1", "data_sha1"]))
    assert linked.dataframe == ""
    assert linked_df == ""
    assert linked_df_with_source == ""

    assert isinstance(linked_df, DataFrame)
    assert linked_df.shape[0] == data.tgt_prob_n

    assert isinstance(linked_df_with_source, DataFrame)
    for field_l, field_r in zip(data.fields_l, data.fields_r):
        if field_l == field_r:
            assert linked_df_with_source[field_l + "_x"].equals(
                linked_df_with_source[field_r + "_y"]
            )
        else:
            assert linked_df_with_source[field_l].equals(linked_df_with_source[field_r])

    # 3. Linked probabilities are inserted correctly

    # 4. Correct number of clusters are resolved

    # 5. Resolved clusters are inserted correctly

    # i. Clean up after ourselves

    db_clear_models(db_engine)
