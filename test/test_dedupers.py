import pytest
from pandas import DataFrame
from sqlalchemy.orm import Session

from matchbox import make_deduper, to_clusters
from matchbox.data import Models

from .fixtures.models import dedupe_data_test_params, dedupe_model_test_params


@pytest.mark.parametrize("fx_data", dedupe_data_test_params)
@pytest.mark.parametrize("fx_deduper", dedupe_model_test_params)
def test_dedupers(
    # Fixtures
    db_engine,
    db_clear_models,
    # Parameterised data classes
    fx_data,
    fx_deduper,
    # Pytest
    request,
):
    """Runs all deduper methodologies over exemplar tables.

    Tests:
        1. That the input data is as expected
        2. That the data is deduplicated correctly
        3. That the deduplicated probabilities are inserted correctly
        4. That the correct number of clusters are resolved
        5. That the resolved clusters are inserted correctly
    """
    # i. Ensure database is clean, collect fixtures, perform any special
    # deduper cleaning

    db_clear_models(db_engine)

    df = request.getfixturevalue(fx_data.fixture)

    fields = list(fx_data.fields.keys())

    if fx_deduper.rename_fields:
        df_renamed = df.copy().rename(columns=fx_data.fields)
        fields_renamed = list(fx_data.fields.values())
        df_renamed = df_renamed.filter(["data_sha1"] + fields_renamed)

    # 1. Input data is as expected

    if fx_deduper.rename_fields:
        assert isinstance(df_renamed, DataFrame)
        assert df_renamed.shape[0] == fx_data.curr_n
    else:
        assert isinstance(df, DataFrame)
        assert df.shape[0] == fx_data.curr_n

    # 2. Data is deduplicated correctly

    deduper_name = f"{fx_deduper.name}_{fx_data.source}"
    deduper_settings = fx_deduper.build_settings(fx_data)

    deduper = make_deduper(
        dedupe_run_name=deduper_name,
        description=f"Testing dedupe of {fx_data.source} with {fx_deduper.name} method",
        deduper=fx_deduper.cls,
        deduper_settings=deduper_settings,
        data=df_renamed if fx_deduper.rename_fields else df,
        data_source=fx_data.source,
    )

    deduped = deduper()

    deduped_df = deduped.to_df()
    deduped_df_with_source = deduped.inspect_with_source(
        left_data=df, left_key="data_sha1", right_data=df, right_key="data_sha1"
    )

    assert isinstance(deduped_df, DataFrame)
    assert deduped_df.shape[0] == fx_data.tgt_prob_n

    assert isinstance(deduped_df_with_source, DataFrame)
    for field in fields:
        assert deduped_df_with_source[field + "_x"].equals(
            deduped_df_with_source[field + "_y"]
        )

    # 3. Deduplicated probabilities are inserted correctly

    deduped.to_cmf(engine=db_engine)

    with Session(db_engine) as session:
        model = session.query(Models).filter_by(name=deduper_name).first()
        assert session.scalar(model.dedupes_count()) == fx_data.tgt_prob_n

    # 4. Correct number of clusters are resolved

    clusters_dupes = to_clusters(results=deduped, key="data_sha1", threshold=0)

    clusters_dupes_df = clusters_dupes.to_df()
    clusters_dupes_df_with_source = clusters_dupes.inspect_with_source(
        left_data=df, left_key="data_sha1", right_data=df, right_key="data_sha1"
    )

    assert isinstance(clusters_dupes_df, DataFrame)
    assert clusters_dupes_df.parent.nunique() == fx_data.tgt_clus_n

    assert isinstance(clusters_dupes_df_with_source, DataFrame)
    for field in fields:
        assert clusters_dupes_df_with_source[field + "_x"].equals(
            clusters_dupes_df_with_source[field + "_y"]
        )

    clusters_all = to_clusters(df, results=deduped, key="data_sha1", threshold=0)

    clusters_all_df = clusters_all.to_df()
    clusters_all_df_with_source = clusters_all.inspect_with_source(
        left_data=df, left_key="data_sha1", right_data=df, right_key="data_sha1"
    )

    assert isinstance(clusters_all_df, DataFrame)
    assert clusters_all_df.parent.nunique() == fx_data.unique_n

    assert isinstance(clusters_all_df_with_source, DataFrame)
    for field in fields:
        assert clusters_all_df_with_source[field + "_x"].equals(
            clusters_all_df_with_source[field + "_y"]
        )

    # 5. Resolved clusters are inserted correctly

    clusters_all.to_cmf(engine=db_engine)

    with Session(db_engine) as session:
        model = session.query(Models).filter_by(name=deduper_name).first()
        assert session.scalar(model.creates_count()) == fx_data.unique_n

    # i. Clean up after ourselves

    db_clear_models(db_engine)
