from src import locations as loc
from src.data import utils as du
from src.data.star import Star
from src.data.probabilities import Probabilities
from src.data.validation import Validation
from src.data.clusters import Clusters

import duckdb
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os
import pytest

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

temp_star = "temp_star"
temp_val = "temp_val"
temp_prob = "temp_prob"
temp_clus = "temp_clus"

star = Star(schema=os.getenv("SCHEMA"), table=temp_star)


def validate_against_answer(my_cluster, validated_cluster, n_type="par"):
    clus_check_l = duckdb.sql(
        """
        select
            cluster,
            id,
            source,
            n::int as n
        from
            my_cluster
        order by
            cluster,
            source,
            id,
            n
    """
    )
    clus_check_r = duckdb.sql(
        f"""
        select
            cluster,
            id,
            source,
            n_{n_type}::int as n
        from
            validated_cluster
        order by
            cluster,
            source,
            id,
            n_{n_type}
    """
    )
    return clus_check_l.df().equals(clus_check_r.df())


tests = [
    "unambig_t2_e4",
    "unambig_t3_e2",
    "masked_t3_e3",
    "val_masked_t3_e2",
    "val_unambig_t3_e2",
]


@pytest.mark.parametrize("test_name", tests)
def test_parallel(test_name):
    # Setup test
    prob, clus, val = du.load_test_data(
        Path(loc.PROJECT_DIR, "test", test_name), int_to_uuid=True
    )
    probabilities = Probabilities(
        schema=os.getenv("SCHEMA"), table=temp_prob, star=star
    )
    probabilities.create(overwrite=True)
    probabilities.add_probabilities(prob.drop(["uuid", "link_type"], axis=1))

    validation = Validation(schema=os.getenv("SCHEMA"), table=temp_val)
    validation.create(overwrite=True)
    validation.add_validation(val.drop("uuid", axis=1))

    clusters = Clusters(schema=os.getenv("SCHEMA"), table=temp_clus, star=star)
    clusters.create(overwrite=True)

    du.query_nonreturn(
        f"""
        insert into {clusters.schema_table}
            select
                gen_random_uuid() as uuid,
                cast(
                    lpad(
                        to_hex(
                            row_number() over ()
                        ),
                        32,
                        '0'
                    ) as uuid
                ) as cluster,
                init.id,
                init.source,
                0 as n
            from (
                select
                    *
                from
                    {probabilities.schema_table}
                where
                    source = 1
            ) init
    """
    )

    # Resolve and check

    clusters.add_clusters(
        probabilities=probabilities,
        validation=validation,
        n=1,
        threshold=0.7,
        add_unmatched_dims=False,
    )

    passed = validate_against_answer(clusters.read(), clus, n_type="par")

    assert passed


@pytest.mark.parametrize("test_name", tests)
def test_sequential(test_name):
    # Setup test
    prob, clus, val = du.load_test_data(
        Path(loc.PROJECT_DIR, "test", test_name), int_to_uuid=True
    )
    prob_sequence_dict = {i - 1: g for i, g in prob.groupby("source")}
    val_sequence_dict = {i - 1: g for i, g in val.groupby("source")}

    # Initialise clusters -- involves some messy work with the prob table but nvm
    probabilities = Probabilities(
        schema=os.getenv("SCHEMA"), table=temp_prob, star=star
    )
    probabilities.create(overwrite=True)
    probabilities.add_probabilities(prob.drop(["uuid", "link_type"], axis=1))
    clusters = Clusters(schema=os.getenv("SCHEMA"), table=temp_clus, star=star)
    clusters.create(overwrite=True)

    du.query_nonreturn(
        f"""
        insert into {clusters.schema_table}
            select
                gen_random_uuid() as uuid,
                cast(
                    lpad(
                        to_hex(
                            row_number() over ()
                        ),
                        32,
                        '0'
                    ) as uuid
                ) as cluster,
                init.id,
                init.source,
                0 as n
            from (
                select
                    *
                from
                    {probabilities.schema_table}
                where
                    source = 1
            ) init
    """
    )

    # Iterate through the sequence resolving at each step
    for i in range(len(prob_sequence_dict)):
        # Create probability table at step n
        prob_n = prob_sequence_dict[i]
        probabilities = Probabilities(
            schema=os.getenv("SCHEMA"), table=temp_prob, star=star
        )
        probabilities.create(overwrite=True)
        probabilities.add_probabilities(prob_n.drop(["uuid", "link_type"], axis=1))

        # Create validation table at step n
        try:
            val_n = val_sequence_dict[i]
        except KeyError:
            val_n = val.iloc[0:0]
        validation = Validation(schema=os.getenv("SCHEMA"), table=temp_val)
        validation.create(overwrite=True)
        validation.add_validation(val_n.drop("uuid", axis=1))

        # Resolve at step n
        clusters.add_clusters(
            probabilities=probabilities,
            validation=validation,
            n=i,
            threshold=0.7,
            add_unmatched_dims=False,
        )

    # Check
    passed = validate_against_answer(clusters.read(), clus, n_type="seq")

    assert passed
