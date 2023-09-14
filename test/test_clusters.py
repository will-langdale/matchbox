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


# tests = [
#     "unambig_t2_e4",
#     "unambig_t3_e2",
#     "masked_t3_e3",
#     "val_masked_t3_e2",
#     "val_unambig_t3_e2",
# ]
# class TestParallelClusers:
def test_unambig_t2_e4_parallel():
    # Setup test
    prob, clus, val = du.load_test_data(
        Path(loc.PROJECT_DIR, "test", "unambig_t2_e4"), int_to_uuid=True
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
                row_number() over () as cluster,
                init.id,
                init.source,
                0 as n
            from (
                select
                    *
                from
                    {probabilities.schema_table}
                where
                    cluster = 0
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
