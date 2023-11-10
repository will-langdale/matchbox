from cmf.data.clusters import Clusters
from cmf.data.datasets import Datasets
from cmf.data.probabilities import Probabilities
from cmf.data.validation import Validation
from cmf.data.table import Table

from dotenv import load_dotenv, find_dotenv
import os
from pydantic import BaseModel


class CMFDB(BaseModel):
    """
    The entrypoint to the whole Company Matching Framework database.
    """

    datasets: Datasets
    clusters: Clusters
    probabilities: Probabilities
    validation: Validation


def make_cmf_connection() -> CMFDB:
    load_dotenv(find_dotenv())

    datasets = Datasets(
        db_table=Table(
            db_schema=os.getenv("SCHEMA"), db_table=os.getenv("DATASETS_TABLE")
        ),
    )
    clusters = Clusters(
        db_datasets=datasets,
        db_table=Table(
            db_schema=os.getenv("SCHEMA"), db_table=os.getenv("CLUSTERS_TABLE")
        ),
    )
    probabilities = Probabilities(
        db_table=Table(
            db_schema=os.getenv("SCHEMA"), db_table=os.getenv("PROBABILITIES_TABLE")
        )
    )
    validation = Validation(
        db_table=Table(
            db_schema=os.getenv("SCHEMA"), db_table=os.getenv("VALIDATE_TABLE")
        )
    )

    return CMFDB(
        datasets=datasets,
        clusters=clusters,
        probabilities=probabilities,
        validation=validation,
    )
