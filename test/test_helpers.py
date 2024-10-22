import logging

from dotenv import find_dotenv, load_dotenv
from matchbox import process, query
from matchbox.clean import company_name, company_number
from matchbox.helpers import (
    cleaner,
    cleaners,
    comparison,
    draw_model_tree,
    selector,
    selectors,
)
from matchbox.server.models import Source
from matchbox.server.postgresql import MatchboxPostgres
from matplotlib.figure import Figure
from pandas import DataFrame

from .fixtures.db import AddIndexedDataCallable

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)


def test_selectors(
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
):
    # Setup
    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    crn_wh = warehouse_data[0]
    select_crn = selector(
        table=str(crn_wh),
        fields=["id", "crn"],
        engine=crn_wh.database.engine,
    )

    duns_wh = warehouse_data[1]
    select_duns = selector(
        table=str(duns_wh),
        fields=["id", "duns"],
        engine=duns_wh.database.engine,
    )

    select_crn_duns = selectors(select_crn, select_duns)

    assert select_crn_duns is not None


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

    crn_wh = warehouse_data[0]
    select_crn = selector(
        table=str(crn_wh),
        fields=["crn", "company_name"],
        engine=crn_wh.database.engine,
    )
    crn = query(
        selector=select_crn,
        backend=matchbox_postgres,
        model=None,
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
            "l.company_name = r.company_name" " and l.data_hub_id = r.data_hub_id"
        )
    )

    assert comparison_name_id is not None


def test_draw_model_tree(matchbox_postgres: MatchboxPostgres):
    plt = draw_model_tree(backend=matchbox_postgres)
    assert isinstance(plt, Figure)


# def test_model_deletion(
#     matchbox_postgres,
#     db_clear_models,
#     db_add_dedupe_models_and_data,
#     db_add_link_models_and_data,
#     request,
# ):
#     """
#     Tests the deletion of:

#     * The model from the model table
#     * The creates edges the model made
#     * Any models that depended on this model, and their creates edges
#     * Any probability values associated with the model
#     * All of the above for all parent models. As every model is defined by
#         its children, deleting a model means cascading deletion to all ancestors
#     """
#     db_clear_models(matchbox_postgres)
#     db_add_link_models_and_data(
#         matchbox_postgres=matchbox_postgres,
#         db_add_dedupe_models_and_data=db_add_dedupe_models_and_data,
#         dedupe_data=dedupe_data_test_params,
#         dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
#         link_data=link_data_test_params,
#         link_models=[link_model_test_params[0]],  # Deterministic linker,
#         request=request,
#     )

#     # Expect it to delete itself, its probabilities,
#     # its parents, and their probabilities
#     deduper_to_delete = f"naive_{os.getenv('MB__POSTGRES__SCHEMA')}.crn"
#     total_models = len(dedupe_data_test_params) + len(link_data_test_params)

#     with Session(matchbox_postgres) as session:
#         model_list_pre_delete = session.query(Models).all()
#         assert len(model_list_pre_delete) == total_models

#         cluster_count_pre_delete = session.query(Clusters).count()
#         cluster_assoc_count_pre_delete = session.query(clusters_association).count()
#         ddupe_count_pre_delete = session.query(Dedupes).count()
#         ddupe_prob_count_pre_delete = session.query(DDupeProbabilities).count()
#         link_count_pre_delete = session.query(Links).count()
#         link_prob_count_pre_delete = session.query(LinkProbabilities).count()

#         assert cluster_count_pre_delete > 0
#         assert cluster_assoc_count_pre_delete > 0
#         assert ddupe_count_pre_delete > 0
#         assert ddupe_prob_count_pre_delete > 0
#         assert link_count_pre_delete > 0
#         assert link_prob_count_pre_delete > 0

#     # Perform deletion
#     delete_model(deduper_to_delete, engine=matchbox_postgres, certain=True)

#     with Session(matchbox_postgres) as session:
#         model_list_post_delete = session.query(Models).all()
#         # Deletes deduper and parent linkers: 3 models gone
#         assert len(model_list_post_delete) == len(model_list_pre_delete) - 3

#         cluster_count_post_delete = session.query(Clusters).count()
#         cluster_assoc_count_post_delete = session.query(clusters_association).count()
#         ddupe_count_post_delete = session.query(Dedupes).count()
#         ddupe_prob_count_post_delete = session.query(DDupeProbabilities).count()
#         link_count_post_delete = session.query(Links).count()
#         link_prob_count_post_delete = session.query(LinkProbabilities).count()

#         # Cluster, dedupe and link count unaffected
#         assert cluster_count_post_delete == cluster_count_pre_delete
#         assert ddupe_count_post_delete == ddupe_count_pre_delete
#         assert link_count_post_delete == link_count_pre_delete

#         # But count of propose and create edges has dropped
#         assert cluster_assoc_count_post_delete < cluster_assoc_count_pre_delete
#         assert ddupe_prob_count_post_delete < ddupe_prob_count_pre_delete
#         assert link_prob_count_post_delete < link_prob_count_pre_delete
