from dotenv import find_dotenv, load_dotenv
from matchbox.helpers.selector import query, selector, selectors
from matchbox.server.models import Source
from matchbox.server.postgresql import MatchboxPostgres
from pandas import DataFrame
from pytest import FixtureRequest

from ..fixtures.db import (
    AddDedupeModelsAndDataCallable,
    AddIndexedDataCallable,
    AddLinkModelsAndDataCallable,
)
from ..fixtures.models import (
    dedupe_data_test_params,
    dedupe_model_test_params,
    link_data_test_params,
    link_model_test_params,
)

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def test_index(
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    crn_companies: DataFrame,
    duns_companies: DataFrame,
    cdms_companies: DataFrame,
):
    """Test that indexing data works."""
    assert matchbox_postgres.data.count() == 0

    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    def count_deduplicates(df: DataFrame) -> int:
        return df.drop(columns=["id"]).drop_duplicates().shape[0]

    unique = sum(
        count_deduplicates(df) for df in [crn_companies, duns_companies, cdms_companies]
    )

    assert matchbox_postgres.data.count() == unique


def test_query_single_table(
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
):
    """Test querying data from the database."""
    # Setup
    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    # Test
    crn = warehouse_data[0]

    select_crn = selector(
        table=str(crn),
        fields=["id", "crn"],
        engine=crn.database.engine,
    )

    df_crn_sample = query(
        selector=select_crn,
        backend=matchbox_postgres,
        model=None,
        return_type="pandas",
        limit=10,
    )

    assert isinstance(df_crn_sample, DataFrame)
    assert df_crn_sample.shape[0] == 10

    df_crn_full = query(
        selector=select_crn,
        backend=matchbox_postgres,
        model=None,
        return_type="pandas",
    )

    assert df_crn_full.shape[0] == 3000
    assert set(df_crn_full.columns) == {
        "data_hash",
        "test_crn_id",
        "test_crn_crn",
    }


def test_query_multi_table(
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
):
    """Test querying data from multiple tables from the database."""
    # Setup
    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    # Test
    crn = warehouse_data[0]
    duns = warehouse_data[1]

    select_crn = selector(
        table=str(crn),
        fields=["id", "crn"],
        engine=crn.database.engine,
    )
    select_duns = selector(
        table=str(duns),
        fields=["id", "duns"],
        engine=duns.database.engine,
    )
    select_crn_duns = selectors(select_crn, select_duns)

    df_crn_duns_full = query(
        selector=select_crn_duns,
        backend=matchbox_postgres,
        model=None,
        return_type="pandas",
    )

    assert df_crn_duns_full.shape[0] == 3500
    assert df_crn_duns_full[df_crn_duns_full["test_duns_id"].notnull()].shape[0] == 500
    assert df_crn_duns_full[df_crn_duns_full["test_crn_id"].notnull()].shape[0] == 3000

    assert set(df_crn_duns_full.columns) == {
        "data_hash",
        "test_crn_id",
        "test_crn_crn",
        "test_duns_id",
        "test_duns_duns",
    }


def test_query_with_dedupe_model(
    matchbox_postgres: MatchboxPostgres,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
    request: FixtureRequest,
):
    """Test querying data from a deduplication point of truth."""
    # Setup
    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=matchbox_postgres,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        request=request,
    )

    # Test
    crn = warehouse_data[0]

    select_crn = selector(
        table=str(crn),
        fields=["company_name", "crn"],
        engine=crn.database.engine,
    )

    df_crn = query(
        selector=select_crn,
        backend=matchbox_postgres,
        model="naive_test.crn",
        return_type="pandas",
    )

    assert isinstance(df_crn, DataFrame)
    assert df_crn.shape[0] == 3000
    assert set(df_crn.columns) == {
        "cluster_hash",
        "data_hash",
        "test_crn_crn",
        "test_crn_company_name",
    }
    assert df_crn.data_hash.nunique() == 3000
    assert df_crn.cluster_hash.nunique() == 1000


def test_query_with_link_model(
    matchbox_postgres: MatchboxPostgres,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    db_add_link_models_and_data: AddLinkModelsAndDataCallable,
    warehouse_data: list[Source],
    request: FixtureRequest,
):
    """Test querying data from a link point of truth."""
    db_add_link_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        db_add_dedupe_models_and_data=db_add_dedupe_models_and_data,
        backend=matchbox_postgres,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        link_data=link_data_test_params,
        link_models=[link_model_test_params[0]],  # Deterministic linker,
        request=request,
    )


def test_validate_hashes(matchbox_postgres):
    def test_validate_data_hashes():
        pass

    def test_validate_cluster_hashes():
        pass

    def test_validate_nonexistent_hashes():
        pass


def test_get_dataset(matchbox_postgres):
    # Test getting an existing model
    pass


def test_get_model_subgraph(matchbox_postgres):
    # Test getting the model subgraph
    pass


def test_get_model(matchbox_postgres):
    # Test getting an existing model
    pass


def test_delete_model(matchbox_postgres):
    def test_delete_existing_model():
        pass

    def test_delete_nonexistent_model():
        pass

    def test_delete_model_without_confirmation():
        pass


def test_insert_model(matchbox_postgres):
    def test_insert_deduper_model():
        pass

    def test_insert_linker_model():
        pass

    def test_insert_duplicate_model():
        pass


# Additional tests for other properties and methods


def test_datasets_property(matchbox_postgres):
    pass


def test_models_property(matchbox_postgres):
    pass


def test_models_from_property(matchbox_postgres):
    pass


def test_data_property(matchbox_postgres):
    pass


def test_clusters_property(matchbox_postgres):
    pass


def test_creates_property(matchbox_postgres):
    pass


def test_merges_property(matchbox_postgres):
    pass


def test_proposes_property(matchbox_postgres):
    pass


# def test_add_dedupers_and_data(
#     db_engine, db_clear_models, db_add_dedupe_models_and_data, request
# ):
#     """
#     Test that adding models and generated data for deduplication processes works.
#     """
#     db_clear_models(db_engine)
#     db_add_dedupe_models_and_data(
#         db_engine=db_engine,
#         dedupe_data=dedupe_data_test_params,
#         dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper
#         request=request,
#     )

#     dedupe_test_params_dict = {
#         test_param.source: test_param for test_param in dedupe_data_test_params
#     }

#     with Session(db_engine) as session:
#         model_list = session.query(Models).all()

#         assert len(model_list) == len(dedupe_data_test_params)

#         for model in model_list:
#             deduplicates = (
#                 session.query(SourceDataset.db_schema, SourceDataset.db_table)
#                 .filter(SourceDataset.uuid == model.deduplicates)
#                 .first()
#             )

#             test_param = dedupe_test_params_dict[
#                 f"{deduplicates[0]}.{deduplicates[1]}"
#             ]

#             assert session.scalar(model.dedupes_count()) == test_param.tgt_prob_n
#             # We assert unique_n rather than tgt_clus_n because tgt_clus_n
#             # checks what the deduper found, not what was inserted
#             assert session.scalar(model.creates_count()) == test_param.unique_n

#     db_clear_models(db_engine)


# def test_add_linkers_and_data(
#     db_engine,
#     db_clear_models,
#     db_add_dedupe_models_and_data,
#     db_add_link_models_and_data,
#     request,
# ):
#     """
#     Test that adding models and generated data for link processes works.
#     """
#     naive_deduper_params = [dedupe_model_test_params[0]]  # Naive deduper
#     deterministic_linker_params = [link_model_test_params[0]]  # Deterministic linker

#     db_clear_models(db_engine)
#     db_add_link_models_and_data(
#         db_engine=db_engine,
#         db_add_dedupe_models_and_data=db_add_dedupe_models_and_data,
#         dedupe_data=dedupe_data_test_params,
#         dedupe_models=naive_deduper_params,
#         link_data=link_data_test_params,
#         link_models=deterministic_linker_params,
#         request=request,
#     )

#     with Session(db_engine) as session:
#         model_list = session.query(Models).filter(Models.deduplicates == None).all()  # NoQA E711

#         assert len(model_list) == len(link_data_test_params)

#     for fx_linker, fx_data in itertools.product(
#         deterministic_linker_params, link_data_test_params
#     ):
#         linker_name = f"{fx_linker.name}_{fx_data.source_l}_{fx_data.source_r}"

#         with Session(db_engine) as session:
#             model = session.query(Models).filter(Models.name == linker_name).first()

#             assert session.scalar(model.links_count()) == fx_data.tgt_prob_n
#             # We assert unique_n rather than tgt_clus_n because tgt_clus_n
#             # checks what the linker found, not what was inserted
#             assert session.scalar(model.creates_count()) == fx_data.unique_n

#     db_clear_models(db_engine)
