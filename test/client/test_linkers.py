import pytest
from matchbox import make_model, query
from matchbox.helpers import selector, selectors
from matchbox.server.models import Source, SourceWarehouse
from matchbox.server.postgresql import MatchboxPostgres
from pandas import DataFrame

from ..fixtures.db import AddDedupeModelsAndDataCallable, AddIndexedDataCallable
from ..fixtures.models import (
    LinkTestParams,
    ModelTestParams,
    dedupe_data_test_params,
    dedupe_model_test_params,
    link_data_test_params,
    link_model_test_params,
)


@pytest.mark.parametrize("fx_data", link_data_test_params)
@pytest.mark.parametrize("fx_linker", link_model_test_params)
def test_linkers(
    # Fixtures
    matchbox_postgres: MatchboxPostgres,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse: SourceWarehouse,
    warehouse_data: list[Source],
    # Parameterised data classes
    fx_data: LinkTestParams,
    fx_linker: ModelTestParams,
    # Pytest
    request: pytest.FixtureRequest,
):
    """Runs all linker methodologies over exemplar tables.

    Tests:
        1. That the input data is as expected
        2. That the data is linked correctly
        3. That the linked probabilities are inserted correctly
        4. That the correct number of clusters are resolved
        4. That the correct number of clusters are resolved and inserted correctly
    """
    # i. Ensure database is ready, collect fixtures, perform any special linker cleaning

    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=matchbox_postgres,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        request=request,
    )

    df_l: DataFrame = request.getfixturevalue(fx_data.fixture_l)
    df_r: DataFrame = request.getfixturevalue(fx_data.fixture_r)

    fields_l = list(fx_data.fields_l.keys())
    fields_r = list(fx_data.fields_r.keys())

    if fx_linker.rename_fields:
        df_l_renamed = df_l.copy().rename(columns=fx_data.fields_l)
        df_r_renamed = df_r.copy().rename(columns=fx_data.fields_r)
        fields_l_renamed = list(fx_data.fields_l.values())
        fields_r_renamed = list(fx_data.fields_r.values())
        df_l_renamed = df_l_renamed.filter(["hash"] + fields_l_renamed)
        df_r_renamed = df_r_renamed.filter(["hash"] + fields_r_renamed)
        assert set(df_l_renamed.columns) == set(df_r_renamed.columns)
        assert df_l_renamed.dtypes.equals(df_r_renamed.dtypes)

    # 1. Input data is as expected

    if fx_linker.rename_fields:
        assert isinstance(df_l_renamed, DataFrame)
        assert df_l_renamed.shape[0] == fx_data.curr_n_l
    else:
        assert isinstance(df_l, DataFrame)
        assert df_l.shape[0] == fx_data.curr_n_l

    if fx_linker.rename_fields:
        assert isinstance(df_r_renamed, DataFrame)
        assert df_r_renamed.shape[0] == fx_data.curr_n_r
    else:
        assert isinstance(df_r, DataFrame)
        assert df_r.shape[0] == fx_data.curr_n_r

    # 2. Data is linked correctly

    linker_name = f"{fx_linker.name}_{fx_data.source_l}_{fx_data.source_r}"
    linker_settings = fx_linker.build_settings(fx_data)

    model = make_model(
        model_name=linker_name,
        description=(
            f"Testing link of {fx_data.source_l} and {fx_data.source_r} "
            f"with {fx_linker.name} method."
        ),
        model_class=fx_linker.cls,
        model_settings=linker_settings,
        left_data=df_l_renamed if fx_linker.rename_fields else df_l,
        left_source=fx_data.source_l,
        right_data=df_r_renamed if fx_linker.rename_fields else df_r,
        right_source=fx_data.source_r,
    )

    linked = model.run()

    linked_df = linked.to_df()

    linked_df_with_source = linked.inspect_with_source(
        left_data=df_l,
        left_key="hash",
        right_data=df_r,
        right_key="hash",
    )

    assert isinstance(linked_df, DataFrame)
    assert linked_df.shape[0] == fx_data.tgt_prob_n

    assert isinstance(linked_df_with_source, DataFrame)
    for field_l, field_r in zip(fields_l, fields_r, strict=True):
        assert linked_df_with_source[field_l].equals(linked_df_with_source[field_r])

    # 3. Linked probabilities are inserted correctly

    linked.to_matchbox(backend=matchbox_postgres)

    model = matchbox_postgres.get_model(model=linker_name)
    assert model.probabilities.count() == fx_data.tgt_prob_n

    # 4. Correct number of clusters are resolved and inserted correctly

    model.set_truth_threshold(probability=0.0)

    l_r_selector = selectors(
        selector(
            table=fx_data.source_l,
            fields=list(fx_data.fields_l.values()),
            engine=warehouse.engine,
        ),
        selector(
            table=fx_data.source_r,
            fields=list(fx_data.fields_r.values()),
            engine=warehouse.engine,
        ),
    )

    clusters = query(
        selector=l_r_selector,
        backend=matchbox_postgres,
        return_type="pandas",
        model=linker_name,
    )

    assert isinstance(clusters, DataFrame)
    assert clusters.hash.nunique() == fx_data.tgt_clus_n

    model = matchbox_postgres.get_model(model=linker_name)

    assert model.clusters.count() == fx_data.unique_n
