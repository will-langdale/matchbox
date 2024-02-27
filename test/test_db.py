import logging
import os
from test.fixtures.models import (
    dedupe_data_test_params,
    dedupe_model_test_params,
    link_data_test_params,
    link_model_test_params,
)

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import MetaData, Table, insert, inspect
from sqlalchemy.orm import Session

from cmf.admin import add_dataset
from cmf.data import (
    Clusters,
    DDupeProbabilities,
    Dedupes,
    LinkProbabilities,
    Links,
    Models,
    SourceData,
    SourceDataset,
    clusters_association,
)
from cmf.data import utils as du

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)


def test_database(db_engine):
    """
    Test the database contains all the tables we expect.
    """
    tables = set(inspect(db_engine[1]).get_table_names(schema=os.getenv("SCHEMA")))
    to_check = {
        "crn",
        "duns",
        "cdms",
        "cmf__models_create_clusters",
        "cmf__clusters",
        "cmf__cluster_validation",
        "cmf__source_dataset",
        "cmf__source_data",
        "cmf__ddupes",
        "cmf__ddupe_probabilities",
        "cmf__ddupe_contains",
        "cmf__ddupe_validation",
        "cmf__links",
        "cmf__link_probabilities",
        "cmf__link_contains",
        "cmf__link_validation",
        "cmf__models",
        "cmf__models_from",
    }

    assert tables == to_check


def test_add_data(db_engine):
    """
    Test all datasets were inserted.
    """
    with Session(db_engine[1]) as session:
        inserted_tables = session.query(SourceDataset.db_table).all()
        inserted_tables = {t[0] for t in inserted_tables}
        expected_tables = {"crn", "duns", "cdms"}

    assert inserted_tables == expected_tables


def test_inserted_data(db_engine, crn_companies, duns_companies, cdms_companies):
    """
    Test all data was inserted.
    Note we drop duplicates because they're rolled up to arrays.
    """
    with Session(db_engine[1]) as session:
        inserted_rows = session.query(SourceData).count()
        raw_rows = (
            crn_companies.drop(columns=["id"]).drop_duplicates().shape[0]
            + duns_companies.drop(columns=["id"]).drop_duplicates().shape[0]
            + cdms_companies.drop(columns=["id"]).drop_duplicates().shape[0]
        )

    assert inserted_rows == raw_rows


def test_insert_data(db_engine, crn_companies, duns_companies, cdms_companies):
    """
    Test that when new data is added to the system, insertion updates it.
    Adds 5 rows to crn table, then adds the dataset again.
    Should result in the same as test_inserted_data + 5
    """
    new_data = [
        {
            "id": 3001,
            "company_name": "Twitterlist",
            "crn": "01HJ0TY5CRPT6ZWWJMH3K4DXH0",
        },
        {"id": 3002, "company_name": "Avaveo", "crn": "01HJ0TY5CR79KQT423SD5HMCXE"},
        {"id": 3003, "company_name": "Realmix", "crn": "01HJ0TY5CRRQBFQNVANJEPJ29D"},
        {"id": 3004, "company_name": "Eidel", "crn": "01HJ0TY5CRET0YPB0WF2R0DFEB"},
        {"id": 3005, "company_name": "Zoozzy", "crn": "01HJ0TY5CRHDX0NX5RSBJWSSKF"},
    ]
    with Session(db_engine[1]) as session:
        # Reflect the table and insert the data
        db_metadata = MetaData(schema=os.getenv("SCHEMA"))
        crn_table = Table(
            "crn",
            db_metadata,
            schema=os.getenv("SCHEMA"),
            autoload_with=session.get_bind(),
        )
        session.execute(insert(crn_table), new_data)
        session.commit()

        # Add the dataset again
        add_dataset(
            {
                "schema": os.getenv("SCHEMA"),
                "table": "crn",
                "id": "id",
            },
            db_engine[1],
        )

        # Test SourceData now contains 5 more rows
        inserted_rows = session.query(SourceData).count()
        raw_rows_plus5 = (
            crn_companies.drop(columns=["id"]).drop_duplicates().shape[0]
            + duns_companies.drop(columns=["id"]).drop_duplicates().shape[0]
            + cdms_companies.drop(columns=["id"]).drop_duplicates().shape[0]
            + 5
        )

    assert inserted_rows == raw_rows_plus5


def test_model_cluster_association(db_engine, db_clear_models, db_add_models):
    """
    Test that cluster read/write via the association objects works as expected.
    """
    # Refresh model layer
    db_clear_models(db_engine)
    db_add_models(db_engine)

    # Model has six clusters
    with Session(db_engine[1]) as session:
        m = session.query(Models).filter_by(name="l_m1").first()
        clusters_in_db = session.query(Clusters).count()
        creates_in_db = session.query(clusters_association).count()

        assert len(m.creates) == 6
        assert clusters_in_db == 6
        assert creates_in_db == 12  # two models in db, l_m1 and l_m2

        # Clear the edges for the next test
        m.creates.clear()
        session.commit()

    # Model creates no clusters but clusters still exist
    with Session(db_engine[1]) as session:
        m = session.query(Models).filter_by(name="l_m1").first()
        clusters_in_db = session.query(Clusters).count()
        creates_in_db = session.query(clusters_association).count()

        assert len(m.creates) == 0
        assert clusters_in_db == 6  # nothing deleted
        assert creates_in_db == 6  # now one model in db, l_m2


# DDupeProbabilities, Dedupes, LinkProbabilities, Links
def test_model_ddupe_association(db_engine, db_clear_models, db_add_models):
    """
    Test that dedupe read/write via the association objects works as expected.

    Test that model deletion works as expected, removing the model and
    its proposes edges in the association object, but not the deduplications.
    """
    # Refresh model layer
    db_clear_models(db_engine)
    db_add_models(db_engine)

    # Model proposes deduplications across six data nodes, 6**2
    with Session(db_engine[1]) as session:
        m = session.query(Models).filter_by(name="dd_m1").first()
        ddupes_in_db = session.query(Dedupes).count()
        ddupe_probs_in_db = session.query(DDupeProbabilities).count()

        assert len(m.proposes_dedupes) == 36  # 6 * 6 data comparisons
        assert ddupes_in_db == 36
        assert ddupe_probs_in_db == 72  # two models in db, dd_m1 and dd_m2

        # Clear the edges for the next test
        m.proposes_dedupes.clear()
        session.commit()

    # Model proposes no deduplications but dedupes still exist
    with Session(db_engine[1]) as session:
        m = session.query(Models).filter_by(name="dd_m1").first()
        ddupes_in_db = session.query(Dedupes).count()
        ddupe_probs_in_db = session.query(DDupeProbabilities).count()

        assert len(m.proposes_dedupes) == 0
        assert ddupes_in_db == 36  # nothing deleted
        assert ddupe_probs_in_db == 36  # now one model in db, dd_m2


def test_model_link_association(db_engine, db_clear_models, db_add_models):
    """
    Test that link read/write via the association objects works as expected.

    Test that model deletion works as expected, removing the model and
    its proposes edges in the association object, but not the links.
    """
    # Refresh model layer
    db_clear_models(db_engine)
    db_add_models(db_engine)

    # Model proposes links across six cluster nodes, 6**2
    with Session(db_engine[1]) as session:
        m = session.query(Models).filter_by(name="l_m1").first()
        links_in_db = session.query(Links).count()
        link_probs_in_db = session.query(LinkProbabilities).count()

        assert len(m.proposes_links) == 36  # 6 * 6 cluster comparisons
        assert links_in_db == 36
        assert link_probs_in_db == 72  # two models in db, dd_m1 and dd_m2

        # Clear the edges for the next test
        m.proposes_links.clear()
        session.commit()

    # Model proposes no linkings but links still exist
    with Session(db_engine[1]) as session:
        m = session.query(Models).filter_by(name="l_m1").first()
        links_in_db = session.query(Links).count()
        link_probs_in_db = session.query(LinkProbabilities).count()

        assert len(m.proposes_dedupes) == 0
        assert links_in_db == 36  # nothing deleted
        assert link_probs_in_db == 36  # now one model in db, dd_m2


def test_db_delete(
    db_engine, db_clear_data, db_clear_models, db_add_data, db_add_models
):
    """
    Test that the clearing test functions works.
    """
    with Session(db_engine[1]) as session:
        data_before = session.query(SourceData).count()
        models_before = session.query(Models).count()

    db_clear_models(db_engine)
    db_clear_data(db_engine)

    with Session(db_engine[1]) as session:
        data_after = session.query(SourceData).count()
        models_after = session.query(Models).count()

    assert data_before != data_after
    assert models_before != models_after

    # Add it all back
    db_add_data(db_engine)
    db_add_models(db_engine)


def test_add_dedupers_and_data(
    db_engine, db_clear_models, db_add_dedupe_models_and_data, request
):
    """
    Test that adding models and generated data for deduplication processes works.
    """
    db_clear_models(db_engine)
    db_add_dedupe_models_and_data(
        db_engine=db_engine,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper
        request=request,
    )

    dedupe_test_params_dict = {
        test_param.source: test_param for test_param in dedupe_data_test_params
    }

    with Session(db_engine[1]) as session:
        model_list = session.query(Models).all()

        assert len(model_list) == len(dedupe_data_test_params)

        for model in model_list:
            deduplicates = (
                session.query(SourceDataset.db_schema, SourceDataset.db_table)
                .filter(SourceDataset.uuid == model.deduplicates)
                .first()
            )

            test_param = dedupe_test_params_dict[f"{deduplicates[0]}.{deduplicates[1]}"]

            # We assert unique_n rather than tgt_prob_clus because tgt_prob_clus
            # checks what the deduper found, not what was inserted
            assert len(model.proposes_dedupes) == test_param.tgt_prob_n
            assert len(model.creates) == test_param.unique_n

    db_clear_models(db_engine)


def test_add_linkers_and_data(
    db_engine,
    db_clear_models,
    db_add_dedupe_models_and_data,
    db_add_link_models_and_data,
    request,
):
    """
    Test that adding models and generated data for link processes works.
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

    with Session(db_engine[1]) as session:
        model_list = session.query(Models).filter(Models.deduplicates == None).all()  # NoQA E711

        assert len(model_list) == len(link_data_test_params)

        for model in model_list:
            # Fetch linker's source models
            child_l, child_r = model.child_neighbours()

            # Get the relevant test result parameters based on source datasets
            test_param = None

            data_test_lookup = {}

            for data_param in link_data_test_params:
                source_l_sha1 = (
                    session.query(Models.sha1)
                    .filter(Models.name == data_param.source_l)
                    .scalar()
                )
                source_r_sha1 = (
                    session.query(Models.sha1)
                    .filter(Models.name == data_param.source_r)
                    .scalar()
                )
                model_sha1 = du.list_to_value_ordered_sha1(
                    [bytes(model.name, encoding="utf-8"), source_l_sha1, source_r_sha1]
                )
                data_test_lookup[model_sha1] = data_param

            test_param = data_test_lookup.get(model.sha1)

            # for data_param in link_data_test_params:
            # lr_match = (
            #     data_param.source_l == child_l.name
            #     and data_param.source_r == child_r.name
            # )
            # rl_match = (
            #     data_param.source_l == child_r.name
            #     and data_param.source_r == child_l.name
            # )
            # if lr_match or rl_match:
            #     test_param = data_param

            assert test_param is not None

            # We assert unique_n rather than tgt_prob_clus because tgt_prob_clus
            # checks what the linker found, not what was inserted
            assert len(model.proposes_links) == test_param.tgt_prob_n
            assert len(model.creates) == test_param.unique_n

    db_clear_models(db_engine)
