from src.config import settings
from src.models import utils as mu
from src.data.datasets import CompanyMatchingDatasets
from src.locations import OUTPUTS_HOME

import click
import logging
import mlflow
from dotenv import find_dotenv, load_dotenv
from os import path

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@click.command()
@click.option("--run_name", required=True, type=str, help="Namespace of the run")
@click.option(
    "--description",
    default=None,
    type=str,
    help="Description of the training run",
)
@click.option(
    "--sample",
    default=None,
    show_default=True,
    help="Sample size for data, useful for speedy testing",
)
@click.option(
    "--dev",
    is_flag=True,
    help="Dev runs allow to run this script with a dirty git repo",
)
def train_model(run_name, description, sample, dev):
    """
    Trains the model and stores a JSON of its settings
    """

    logger = logging.getLogger(__name__)

    if sample is not None:
        if not dev:
            raise ValueError("Cannot subsample dataset during production run")

    # Load data
    logger.info("Loading data")
    datasets = CompanyMatchingDatasets(sample=sample)

    # Instantiate linker
    logger.info("Instantiating linker")
    linker = datasets.linker(settings)

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

        json_file_path = path.join(OUTPUTS_HOME, "companies_matching_model.json")
        linker.save_model_to_json(out_path=json_file_path)
        mlflow.log_artifact(json_file_path, mu.DEFAULT_ARTIFACT_PATH)

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
