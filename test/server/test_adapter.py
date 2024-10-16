from dotenv import find_dotenv, load_dotenv
from matchbox.server.base import IndexableDataset
from matchbox.server.postgresql import MatchboxPostgres

from ..fixtures.db import AddIndexedDataCallable

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def test_index(
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[IndexableDataset],
):
    # Test indexing a dataset
    pass


def test_query(matchbox_postgres, db_add_link_models_and_data):
    # Test with different combinations of parameters
    def test_basic_query():
        pass

    def test_query_with_model():
        pass

    def test_query_with_limit():
        pass

    def test_query_return_pandas():
        pass

    def test_query_return_sqlalchemy():
        pass


def test_validate_hashes(matchbox_postgres):
    def test_validate_data_hashes():
        pass

    def test_validate_cluster_hashes():
        pass

    def test_validate_nonexistent_hashes():
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
