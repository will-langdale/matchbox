import pyarrow as pa
import pyarrow.compute as pc
import pytest
from pandas import DataFrame

from matchbox import make_model, query
from matchbox.client.helpers.selector import Selector
from matchbox.common.sources import Source
from matchbox.server.postgresql import MatchboxPostgres

from ..fixtures.db import AddIndexedDataCallable
from ..fixtures.models import (
    DedupeTestParams,
    ModelTestParams,
    dedupe_data_test_params,
    dedupe_model_test_params,
)


@pytest.mark.parametrize("fx_data", dedupe_data_test_params)
@pytest.mark.parametrize("fx_deduper", dedupe_model_test_params)
@pytest.mark.docker
def test_dedupers(
    # Fixtures
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
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

    select: list[Selector]
    df: DataFrame

    select, df = request.getfixturevalue(fx_data.fixture)

    fields = list(fx_data.fields.keys())

    if fx_deduper.rename_fields:
        df_renamed = df.copy().rename(columns=fx_data.fields)
        fields_renamed = list(fx_data.fields.values())
        df_renamed = df_renamed.filter(["id"] + fields_renamed)

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
        left_resolution=fx_data.source,
    )

    results = model.run()

    result_with_source = results.inspect_probabilities(
        left_data=df, left_key="id", right_data=df, right_key="id"
    )

    assert isinstance(results.probabilities, pa.Table)
    assert results.probabilities.shape[0] == fx_data.tgt_prob_n

    assert isinstance(result_with_source, DataFrame)
    for field in fields:
        assert result_with_source[field + "_x"].equals(result_with_source[field + "_y"])

    # 3. Correct number of clusters are resolved

    clusters_with_source = results.inspect_clusters(
        left_data=df, left_key="id", right_data=df, right_key="id"
    )

    assert isinstance(results.clusters, pa.Table)
    assert pc.count_distinct(results.clusters["parent"]).as_py() == fx_data.tgt_clus_n

    assert isinstance(clusters_with_source, DataFrame)
    for field in fields:
        assert clusters_with_source[field + "_x"].equals(
            clusters_with_source[field + "_y"]
        )

    # 4. Probabilities and clusters are inserted correctly

    results.to_matchbox()

    retrieved_results = matchbox_postgres.get_model_results(model=deduper_name)
    assert retrieved_results.shape[0] == fx_data.tgt_prob_n

    matchbox_postgres.set_model_truth(model=deduper_name, truth=0.0)

    clusters = query(
        select,
        resolution_name=deduper_name,
        return_type="pandas",
    )

    assert isinstance(clusters, DataFrame)
    assert clusters.id.nunique() == fx_data.unique_n
