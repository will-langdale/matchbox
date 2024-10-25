import pytest
from matchbox import make_model, query
from matchbox.helpers import selector
from matchbox.server.models import Source, SourceWarehouse
from matchbox.server.postgresql import MatchboxPostgres
from pandas import DataFrame

from ..fixtures.db import AddIndexedDataCallable
from ..fixtures.models import (
    DedupeTestParams,
    ModelTestParams,
    dedupe_data_test_params,
    dedupe_model_test_params,
)


@pytest.mark.parametrize("fx_data", dedupe_data_test_params)
@pytest.mark.parametrize("fx_deduper", dedupe_model_test_params)
def test_dedupers(
    # Fixtures
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse: SourceWarehouse,
    warehouse_data: list[Source],
    # Parameterised data classes
    fx_data: DedupeTestParams,
    fx_deduper: ModelTestParams,
    # Pytest
    request: pytest.FixtureRequest,
):
    """Runs all deduper methodologies over exemplar tables.

    Tests:
        1. That the input data is as expected
        2. That the data is deduplicated correctly
        3. That the deduplicated probabilities are inserted correctly
        4. That the correct number of clusters are resolved and inserted correctly
    """
    # i. Ensure database is ready, collect fixtures, perform any special
    # deduper cleaning

    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    df: DataFrame = request.getfixturevalue(fx_data.fixture)

    fields = list(fx_data.fields.keys())

    if fx_deduper.rename_fields:
        df_renamed = df.copy().rename(columns=fx_data.fields)
        fields_renamed = list(fx_data.fields.values())
        df_renamed = df_renamed.filter(["hash"] + fields_renamed)

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

    model = make_model(
        model_name=deduper_name,
        description=f"Testing dedupe of {fx_data.source} with {fx_deduper.name} method",
        model_class=fx_deduper.cls,
        model_settings=deduper_settings,
        left_data=df_renamed if fx_deduper.rename_fields else df,
        left_source=fx_data.source,
    )

    deduped = model.run()

    deduped_df = deduped.to_df()
    deduped_df_with_source = deduped.inspect_with_source(
        left_data=df, left_key="hash", right_data=df, right_key="hash"
    )

    assert isinstance(deduped_df, DataFrame)
    assert deduped_df.shape[0] == fx_data.tgt_prob_n

    assert isinstance(deduped_df_with_source, DataFrame)
    for field in fields:
        assert deduped_df_with_source[field + "_x"].equals(
            deduped_df_with_source[field + "_y"]
        )

    # 3. Deduplicated probabilities are inserted correctly

    deduped.to_matchbox(backend=matchbox_postgres)

    model = matchbox_postgres.get_model(model=deduper_name)
    assert model.probabilities.count() == fx_data.tgt_prob_n

    # 4. Correct number of clusters are resolved and inserted correctly

    model.set_truth_threshold(probability=0.0)

    clusters = query(
        selector=selector(
            table=fx_data.source,
            fields=list(fx_data.fields.values()),
            engine=warehouse.engine,
        ),
        backend=matchbox_postgres,
        return_type="pandas",
        model=deduper_name,
    )

    assert isinstance(clusters, DataFrame)
    assert clusters.hash.nunique() == fx_data.tgt_clus_n

    model = matchbox_postgres.get_model(model=deduper_name)
    assert model.clusters.count() == fx_data.unique_n
