from src.config import settings
from src.models import utils as mu
from src.locations import OUTPUTS_HOME
from src.data import utils as du

import click
import logging
import mlflow
import duckdb
from dotenv import find_dotenv, load_dotenv
from os import path, makedirs

from splink.duckdb.linker import DuckDBLinker

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@click.command()
@click.option("--run_name", required=True, type=str, help="Namespace of the run")
@click.option(
    "--input_dir",
    required=True,
    type=str,
    help="Directory of the cleaned parquet files",
)
@click.option(
    "--description",
    default=None,
    type=str,
    help="Description of the training run",
)
@click.option(
    "--dev",
    is_flag=True,
    help="Dev runs allow to run this script with a dirty git repo",
)
def train_model(run_name: str, input_dir: str, description: str, dev: bool):
    """
    Trains the model and stores a JSON of its settings
    """

    logger = logging.getLogger(__name__)

    # Load data
    logger.info("Loading data")

    connection = duckdb.connect()
    data = du.build_alias_path_dict(input_dir)

    # Instantiate linker
    logger.info("Instantiating linker")
    linker = DuckDBLinker(
        list(data.values()),
        settings_dict=settings,
        connection=connection,
        input_table_aliases=list(data.keys()),
    )

    with mu.mlflow_run(run_name=run_name, description=description, dev_mode=dev):
        # Estimate model parameters...

        logger.info("Estimating model parameters")

        # ...that random records match
        linker.estimate_probability_two_random_records_match(
            "l.name_unusual_tokens = r.name_unusual_tokens",
            recall=0.7,
        )

        # ...u
        linker.estimate_u_using_random_sampling(max_pairs=1e7)

        # ...m
        linker.estimate_m_from_label_column("comp_num_clean")
        m_by_name_and_postcode_area = """
            l.name_unusual_tokens = r.name_unusual_tokens
            and l.postcode_area = r.postcode_area
        """
        linker.estimate_parameters_using_expectation_maximisation(
            m_by_name_and_postcode_area
        )

        # Save model and settings

        logger.info("Saving model and settings")

        outdir = path.join(OUTPUTS_HOME, input_dir)
        if not path.exists(outdir):
            makedirs(outdir)

        json_file_path = path.join(outdir, "companies_matching_model.json")
        linker.save_model_to_json(out_path=json_file_path, overwrite=True)
        mlflow.log_artifact(json_file_path, mu.DEFAULT_ARTIFACT_PATH)

        # Generating ROC curve and AOC metric

        # logger.info("Generating ROC curve and AOC metric")

        mlflow.log_metric("datasets", len(list(data.keys())))

        logger.info("Done.")


def main():
    """
    Entrypoint
    """
    train_model()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    load_dotenv(find_dotenv())

    main()
