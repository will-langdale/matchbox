from src.config import settings, datasets_and_readfuncs
from src.models import utils as mu
from src.data import utils as du

import click
import logging
import duckdb
from dotenv import find_dotenv, load_dotenv

from splink.duckdb.linker import DuckDBLinker

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def predict_clusters(linker, threshold: float = 0.7):
    """
    Produce outputs such as predictions table,
    comparison viewer, clusters, linker settings
    Args: linker model
    Returns: None
    """

    predictions = linker.predict(threshold_match_probability=threshold)
    clusters = linker.cluster_pairwise_predictions_at_threshold(  # noqa:F841
        predictions,
        threshold_match_probability=threshold,
        pairwise_formatting=True,
        filter_pairwise_format_for_clusters=False,
    )
    lookup = duckdb.sql(
        """
        select
            source_dataset_l as source,
            unique_id_l as source_id,
            cluster_id_l as source_cluster,
            source_dataset_r as target,
            unique_id_r as target_id,
            cluster_id_r as target_cluster,
            match_probability
        from
            clusters
        union
        select
            source_dataset_r as source,
            unique_id_r as source_id,
            cluster_id_r as source_cluster,
            source_dataset_l as target,
            unique_id_l as target_id,
            cluster_id_l as target_cluster,
            match_probability
        from
            df_clusters
    """
    )

    return lookup


@click.command()
@click.option(
    "--run_id",
    required=True,
    type=str,
    help="Run ID of model to load",
)
@click.option(
    "--output_schema",
    required=True,
    type=str,
    help="Schema name to write the lookup output to",
)
@click.option(
    "--output_table",
    required=True,
    type=str,
    help="Table name to write the lookup output to",
)
def main(run_id, output_schema, output_table):
    """
    Entrypoint
    """
    logger = logging.getLogger(__name__)

    logger.info("Retrieving model")
    json_settings = mu.load_model_from_run(run_id)

    # Load data
    logger.info("Loading data")
    data = []
    for dataset in datasets_and_readfuncs.keys():
        df = datasets_and_readfuncs[dataset]()
        logger.info(f"{dataset}: {len(dataset)} items loaded")
        data.append(df)

    # Instantiate linker and load settings
    logger.info("Configuring linker")
    linker = DuckDBLinker(
        data,
        settings,
        input_table_aliases=list(datasets_and_readfuncs.keys()),
    )

    linker.load_model(json_settings)

    # Predict/cluster/create lookup
    logger.info("Building lookup")
    lookup = predict_clusters(linker)

    # Write lookup to output
    logger.info("Writing lookup")
    du.data_workspace_write(
        schema=output_schema, table=output_table, df=lookup.df(), if_exists="replace"
    )

    logger.info("Done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    load_dotenv(find_dotenv())

    main()
