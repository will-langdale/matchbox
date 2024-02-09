from test.fixtures.models import linker_test_params, merge_test_params

import pytest
from pandas import DataFrame
from sqlalchemy.orm import Session

from cmf import make_linker, to_clusters
from cmf.data import Models


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

    assert isinstance(linked_df, DataFrame)
    assert linked_df.shape[0] == data.tgt_prob_n

    assert isinstance(linked_df_with_source, DataFrame)
    for field_l, field_r in zip(data.fields_l, data.fields_r):
        # Drop NA because in data where entities are imbalanced we include the
        # unmatched rows from the source data. This is not a fair check
        linked_df_with_source_no_na = linked_df_with_source.dropna()
        assert linked_df_with_source_no_na[field_l].equals(
            linked_df_with_source_no_na[field_r]
        )

    # 3. Linked probabilities are inserted correctly

    linked.to_cmf(engine=db_engine[1])

    with Session(db_engine[1]) as session:
        model = session.query(Models).filter_by(name=linker_name).first()
        proposed_links = model.proposes_links

    assert len(proposed_links) == data.tgt_prob_n

    # 4. Correct number of clusters are resolved

    clusters_links = to_clusters(results=linked, key="cluster_sha1", threshold=0)

    clusters_links_df = clusters_links.to_df()
    clusters_links_df_with_source = clusters_links.inspect_with_source(
        left_data=df_l,
        left_key="cluster_sha1",
        right_data=df_r,
        right_key="cluster_sha1",
    )

    assert isinstance(clusters_links_df, DataFrame)
    assert clusters_links_df.parent.nunique() == data.tgt_clus_n

    assert isinstance(clusters_links_df_with_source, DataFrame)
    for field_l, field_r in zip(data.fields_l, data.fields_r):
        # We don't drop NA here because this is speficially matched clusters
        assert clusters_links_df_with_source[field_l].equals(
            clusters_links_df_with_source[field_r]
        )

    clusters_all = to_clusters(
        df_l, df_r, results=linked, key="cluster_sha1", threshold=0
    )

    clusters_all_df = clusters_all.to_df()
    clusters_all_df_with_source = clusters_all.inspect_with_source(
        left_data=df_l,
        left_key="cluster_sha1",
        right_data=df_r,
        right_key="cluster_sha1",
    )

    assert isinstance(clusters_all_df, DataFrame)
    assert clusters_all_df.parent.nunique() == data.unique_n

    assert isinstance(clusters_all_df_with_source, DataFrame)
    for field_l, field_r in zip(data.fields_l, data.fields_r):
        assert clusters_all_df_with_source[field_l].equals(
            clusters_all_df_with_source[field_r]
        )

    # 5. Resolved clusters are inserted correctly

    clusters_all.to_cmf(engine=db_engine[1])

    with Session(db_engine[1]) as session:
        model = session.query(Models).filter_by(name=linker_name).first()
        created_clusters = model.creates

    assert len(created_clusters) == data.unique_n

    # i. Clean up after ourselves

    db_clear_models(db_engine)
