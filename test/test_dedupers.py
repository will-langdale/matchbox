from test.fixtures.models import data_test_params, deduper_test_params

import pytest
from pandas import DataFrame
from sqlalchemy.orm import Session

from cmf import make_deduper, to_clusters
from cmf.data import Models


@pytest.mark.parametrize("data", data_test_params)
@pytest.mark.parametrize("ddupe", deduper_test_params)
def test_dedupers(
    # Fixtures
    db_engine,
    db_clear_models,
    # Parameterised data classes
    data,
    ddupe,
    # Pytest
    request,
):
    """Runs all deduper methodologies over exemplar tables.

    Tests:
        1. That the input data is as expected
        2. That the data is deduplicated correctly
        3. That the deduplicated probabilities are inserted correctly
        4. The the correct number of clusters are resolved
        5. That the resolved clusters are inserted correctly
    """
    df = request.getfixturevalue(data.fixture)

    # 1. Input data is as expected

    assert isinstance(df, DataFrame)
    assert df.shape[0] == data.curr_n

    # 2. Data is deduplicated correctly

    deduper_name = f"{ddupe.name}_{data.source}"
    deduper_settings = ddupe.build_settings(data)

    deduper = make_deduper(
        dedupe_run_name=deduper_name,
        description=f"Testing dedupe of {data.source} with {ddupe.name} method",
        deduper=ddupe.deduper,
        deduper_settings=deduper_settings,
        data_source=data.source,
        data=df,
    )

    deduped = deduper()

    deduped_df = deduped.to_df()
    deduped_df_with_source = deduped.inspect_with_source(
        left_data=df, left_key="data_sha1", right_data=df, right_key="data_sha1"
    )

    assert isinstance(deduped_df, DataFrame)
    assert deduped_df.shape[0] == data.tgt_prob_n

    assert isinstance(deduped_df_with_source, DataFrame)
    for field in data.fields:
        assert deduped_df_with_source[field + "_x"].equals(
            deduped_df_with_source[field + "_y"]
        )

    # 3. Deduplicated probabilities are inserted correctly

    deduped.to_cmf(engine=db_engine[1])

    with Session(db_engine[1]) as session:
        model = session.query(Models).filter_by(name=deduper_name).first()
        proposed_dedupes = model.proposes_dedupes

    assert len(proposed_dedupes) == data.tgt_prob_n

    # 4. Correct number of clusters are resolved

    clusters_dupes = to_clusters(results=deduped, key="data_sha1", threshold=0)

    clusters_dupes_df = clusters_dupes.to_df()
    clusters_dupes_df_with_source = clusters_dupes.inspect_with_source(
        left_data=df, left_key="data_sha1", right_data=df, right_key="data_sha1"
    )

    assert isinstance(clusters_dupes_df, DataFrame)
    assert clusters_dupes_df.parent.nunique() == data.tgt_clus_n

    assert isinstance(clusters_dupes_df_with_source, DataFrame)
    for field in data.fields:
        assert clusters_dupes_df_with_source[field + "_x"].equals(
            clusters_dupes_df_with_source[field + "_y"]
        )

    clusters_all = to_clusters(df, results=deduped, key="data_sha1", threshold=0)

    clusters_all_df = clusters_all.to_df()
    clusters_all_df_with_source = clusters_all.inspect_with_source(
        left_data=df, left_key="data_sha1", right_data=df, right_key="data_sha1"
    )

    assert isinstance(clusters_all_df, DataFrame)
    assert clusters_all_df.parent.nunique() == data.unique_n

    assert isinstance(clusters_all_df_with_source, DataFrame)
    for field in data.fields:
        assert clusters_all_df_with_source[field + "_x"].equals(
            clusters_all_df_with_source[field + "_y"]
        )

    # 5. Resolved clusters are inserted correctly

    clusters_all.to_cmf(engine=db_engine[1])

    with Session(db_engine[1]) as session:
        model = session.query(Models).filter_by(name=deduper_name).first()
        created_clusters = model.creates

    assert len(created_clusters) == data.unique_n

    # i. Clean up

    db_clear_models(db_engine)
