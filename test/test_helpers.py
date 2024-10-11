import logging
import os

from dotenv import find_dotenv, load_dotenv
from matchbox import process, query
from matchbox.clean import company_name, company_number
from matchbox.data import (
    Clusters,
    DDupeProbabilities,
    Dedupes,
    LinkProbabilities,
    Links,
    Models,
    clusters_association,
)
from matchbox.helpers import (
    cleaner,
    cleaners,
    comparison,
    delete_model,
    draw_model_tree,
    selector,
    selectors,
)
from matplotlib.figure import Figure
from pandas import DataFrame
from sqlalchemy.orm import Session

from .fixtures.models import (
    dedupe_data_test_params,
    dedupe_model_test_params,
    link_data_test_params,
    link_model_test_params,
)

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)


def test_selectors(db_engine):
    select_crn = selector(
        table=f"{os.getenv('MB__POSTGRES__SCHEMA')}.crn",
        fields=["id", "crn"],
        engine=db_engine,
    )
    select_duns = selector(
        table=f"{os.getenv('MB__POSTGRES__SCHEMA')}.duns",
        fields=["id", "duns"],
        engine=db_engine,
    )
    select_crn_duns = selectors(select_crn, select_duns)

    assert select_crn_duns is not None


def test_single_table_no_model_query(db_engine):
    """Tests query() on a single table. No point of truth to derive clusters"""
    select_crn = selector(
        table=f"{os.getenv('MB__POSTGRES__SCHEMA')}.crn",
        fields=["id", "crn"],
        engine=db_engine,
    )

    df_crn_sample = query(
        selector=select_crn,
        model=None,
        return_type="pandas",
        engine=db_engine,
        limit=10,
    )

    assert isinstance(df_crn_sample, DataFrame)
    assert df_crn_sample.shape[0] == 10

    df_crn_full = query(
        selector=select_crn, model=None, return_type="pandas", engine=db_engine
    )

    assert df_crn_full.shape[0] == 3000
    assert set(df_crn_full.columns) == {
        "data_sha1",
        f"{os.getenv('MB__POSTGRES__SCHEMA')}_crn_id",
        f"{os.getenv('MB__POSTGRES__SCHEMA')}_crn_crn",
    }


def test_multi_table_no_model_query(db_engine):
    """Tests query() on multiple tables. No point of truth to derive clusters"""
    select_crn = selector(
        table=f"{os.getenv('MB__POSTGRES__SCHEMA')}.crn",
        fields=["id", "crn"],
        engine=db_engine,
    )
    select_duns = selector(
        table=f"{os.getenv('MB__POSTGRES__SCHEMA')}.duns",
        fields=["id", "duns"],
        engine=db_engine,
    )
    select_crn_duns = selectors(select_crn, select_duns)

    df_crn_duns_full = query(
        selector=select_crn_duns, model=None, return_type="pandas", engine=db_engine
    )

    assert df_crn_duns_full.shape[0] == 3500
    assert (
        df_crn_duns_full[
            df_crn_duns_full[f"{os.getenv('MB__POSTGRES__SCHEMA')}_duns_id"].notnull()
        ].shape[0]
        == 500
    )
    assert (
        df_crn_duns_full[
            df_crn_duns_full[f"{os.getenv('MB__POSTGRES__SCHEMA')}_crn_id"].notnull()
        ].shape[0]
        == 3000
    )

    assert set(df_crn_duns_full.columns) == {
        "data_sha1",
        f"{os.getenv('MB__POSTGRES__SCHEMA')}_crn_id",
        f"{os.getenv('MB__POSTGRES__SCHEMA')}_crn_crn",
        f"{os.getenv('MB__POSTGRES__SCHEMA')}_duns_id",
        f"{os.getenv('MB__POSTGRES__SCHEMA')}_duns_duns",
    }


def test_single_table_with_model_query(
    db_engine, db_clear_models, db_add_dedupe_models_and_data, request
):
    """Tests query() on a single table using a model point of truth."""
    # Ensure database is clean, insert deduplication models

    db_clear_models(db_engine)
    db_add_dedupe_models_and_data(
        db_engine=db_engine,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper
        request=request,
    )

    # Query

    select_crn = selector(
        table=f"{os.getenv('MB__POSTGRES__SCHEMA')}.crn",
        fields=["crn", "company_name"],
        engine=db_engine,
    )

    crn = query(
        selector=select_crn,
        model=f"naive_{os.getenv('MB__POSTGRES__SCHEMA')}.crn",
        return_type="pandas",
        engine=db_engine,
    )

    assert isinstance(crn, DataFrame)
    assert crn.shape[0] == 3000
    assert set(crn.columns) == {
        "cluster_sha1",
        "data_sha1",
        f"{os.getenv('MB__POSTGRES__SCHEMA')}_crn_crn",
        f"{os.getenv('MB__POSTGRES__SCHEMA')}_crn_company_name",
    }
    assert crn.data_sha1.nunique() == 3000
    assert crn.cluster_sha1.nunique() == 1000


def test_multi_table_with_model_query(
    db_engine,
    db_clear_models,
    db_add_dedupe_models_and_data,
    db_add_link_models_and_data,
    request,
):
    """Tests query() on multiple tables using a model point of truth."""
    # Ensure database is clean, insert deduplication and linker models

    db_clear_models(db_engine)
    db_add_link_models_and_data(
        db_engine=db_engine,
        db_add_dedupe_models_and_data=db_add_dedupe_models_and_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        link_data=link_data_test_params,
        link_models=[link_model_test_params[0]],  # Deterministic linker,
        request=request,
    )

    # Query

    linker_name = (
        f"deterministic_"
        f"naive_{os.getenv('MB__POSTGRES__SCHEMA')}.crn_"
        f"naive_{os.getenv('MB__POSTGRES__SCHEMA')}.duns"
    )

    select_crn = selector(
        table=f"{os.getenv('MB__POSTGRES__SCHEMA')}.crn",
        fields=["crn"],
        engine=db_engine,
    )
    select_duns = selector(
        table=f"{os.getenv('MB__POSTGRES__SCHEMA')}.duns",
        fields=["duns"],
        engine=db_engine,
    )
    select_crn_duns = selectors(select_crn, select_duns)

    crn_duns = query(
        selector=select_crn_duns,
        model=linker_name,
        return_type="pandas",
        engine=db_engine,
    )

    assert isinstance(crn_duns, DataFrame)
    assert crn_duns.shape[0] == 3500
    assert set(crn_duns.columns) == {
        "cluster_sha1",
        "data_sha1",
        f"{os.getenv('MB__POSTGRES__SCHEMA')}_crn_crn",
        f"{os.getenv('MB__POSTGRES__SCHEMA')}_duns_duns",
    }
    assert crn_duns.data_sha1.nunique() == 3500
    assert crn_duns.cluster_sha1.nunique() == 1000


def test_cleaners():
    cleaner_name = cleaner(function=company_name, arguments={"column": "company_name"})
    cleaner_number = cleaner(
        function=company_number, arguments={"column": "company_number"}
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    assert cleaner_name_number is not None


def test_process(db_engine):
    select_name = selector(
        table=f"{os.getenv('MB__POSTGRES__SCHEMA')}.crn",
        fields=["crn", "company_name"],
        engine=db_engine,
    )

    df_name = query(
        selector=select_name, model=None, return_type="pandas", engine=db_engine
    )

    cleaner_name = cleaner(
        function=company_name,
        arguments={"column": f"{os.getenv('MB__POSTGRES__SCHEMA')}_crn_company_name"},
    )
    cleaner_number = cleaner(
        function=company_number,
        arguments={"column": f"{os.getenv('MB__POSTGRES__SCHEMA')}_crn_crn"},
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    df_name_cleaned = process(data=df_name, pipeline=cleaner_name_number)

    assert isinstance(df_name_cleaned, DataFrame)
    assert df_name_cleaned.shape[0] == 3000


def test_comparisons():
    comparison_name_id = comparison(
        sql_condition=(
            "l.company_name = r.company_name" " and l.data_hub_id = r.data_hub_id"
        )
    )

    assert comparison_name_id is not None


def test_draw_model_tree(db_engine):
    plt = draw_model_tree(db_engine)
    assert isinstance(plt, Figure)


def test_model_deletion(
    db_engine,
    db_clear_models,
    db_add_dedupe_models_and_data,
    db_add_link_models_and_data,
    request,
):
    """
    Tests the deletion of:

    * The model from the model table
    * The creates edges the model made
    * Any models that depended on this model, and their creates edges
    * Any probability values associated with the model
    * All of the above for all parent models. As every model is defined by
        its children, deleting a model means cascading deletion to all ancestors
    """
    db_clear_models(db_engine)
    db_add_link_models_and_data(
        db_engine=db_engine,
        db_add_dedupe_models_and_data=db_add_dedupe_models_and_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        link_data=link_data_test_params,
        link_models=[link_model_test_params[0]],  # Deterministic linker,
        request=request,
    )

    # Expect it to delete itself, its probabilities,
    # its parents, and their probabilities
    deduper_to_delete = f"naive_{os.getenv('MB__POSTGRES__SCHEMA')}.crn"
    total_models = len(dedupe_data_test_params) + len(link_data_test_params)

    with Session(db_engine) as session:
        model_list_pre_delete = session.query(Models).all()
        assert len(model_list_pre_delete) == total_models

        cluster_count_pre_delete = session.query(Clusters).count()
        cluster_assoc_count_pre_delete = session.query(clusters_association).count()
        ddupe_count_pre_delete = session.query(Dedupes).count()
        ddupe_prob_count_pre_delete = session.query(DDupeProbabilities).count()
        link_count_pre_delete = session.query(Links).count()
        link_prob_count_pre_delete = session.query(LinkProbabilities).count()

        assert cluster_count_pre_delete > 0
        assert cluster_assoc_count_pre_delete > 0
        assert ddupe_count_pre_delete > 0
        assert ddupe_prob_count_pre_delete > 0
        assert link_count_pre_delete > 0
        assert link_prob_count_pre_delete > 0

    # Perform deletion
    delete_model(deduper_to_delete, engine=db_engine, certain=True)

    with Session(db_engine) as session:
        model_list_post_delete = session.query(Models).all()
        # Deletes deduper and parent linkers: 3 models gone
        assert len(model_list_post_delete) == len(model_list_pre_delete) - 3

        cluster_count_post_delete = session.query(Clusters).count()
        cluster_assoc_count_post_delete = session.query(clusters_association).count()
        ddupe_count_post_delete = session.query(Dedupes).count()
        ddupe_prob_count_post_delete = session.query(DDupeProbabilities).count()
        link_count_post_delete = session.query(Links).count()
        link_prob_count_post_delete = session.query(LinkProbabilities).count()

        # Cluster, dedupe and link count unaffected
        assert cluster_count_post_delete == cluster_count_pre_delete
        assert ddupe_count_post_delete == ddupe_count_pre_delete
        assert link_count_post_delete == link_count_pre_delete

        # But count of propose and create edges has dropped
        assert cluster_assoc_count_post_delete < cluster_assoc_count_pre_delete
        assert ddupe_prob_count_post_delete < ddupe_prob_count_pre_delete
        assert link_prob_count_post_delete < link_prob_count_pre_delete
