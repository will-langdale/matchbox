from src.config import settings
from src.models import utils as mu
from src.data import utils as du

from splink.duckdb.linker import DuckDBLinker

import click
import logging
import duckdb
from dotenv import find_dotenv, load_dotenv
import mlflow
import json

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def predict_clusters(linker, threshold: float = 0.7):
    """
    Produce outputs such as predictions table,
    comparison viewer, clusters, linker settings
    Args: linker model
    Returns: None
    """

    predictions = linker.predict(threshold_match_probability=threshold)
    clusters = linker.cluster_pairwise_predictions_at_threshold(
        predictions,
        threshold_match_probability=threshold,
        pairwise_formatting=True,
        filter_pairwise_format_for_clusters=False,
    )
    clusters_df = clusters.as_pandas_dataframe()  # noqa:F841

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
            clusters_df
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
            clusters_df
    """
    )

    return lookup.df()


@click.command()
@click.option(
    "--run",
    is_flag=True,
    help="Predict from run. Enables run_id if set",
)
@click.option(
    "--run_id",
    required=False,
    default=mu.latest_successful_run_id(),
    type=str,
    help="""
        Run ID of model to load.
        Required if model_version not set.
        Defaults to last successful run
    """,
)
@click.option(
    "--model",
    is_flag=True,
    help="Predict from model. Enables model_name and model_version if set",
)
@click.option(
    "--model_name",
    required=False,
    default=mu.DEFAULT_MODEL_NAME,
    type=int,
    help="Model to load. Required if run_id not set. Defaults to mu.DEFAULT_MODEL_NAME",
)
@click.option(
    "--model_version",
    required=False,
    default=None,
    type=int,
    help="Version of model to load. Required if run_id not set",
)
@click.option(
    "--input_dir",
    required=True,
    type=str,
    help="Directory of the cleaned parquet files",
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
def main(
    run,
    run_id,
    model,
    model_name,
    model_version,
    input_dir,
    output_schema,
    output_table,
):
    """
    Entrypoint
    """
    logger = logging.getLogger(__name__)

    if run and model:
        raise ValueError("Can load a run or a model, not both")
    elif not (run and model):
        raise ValueError("Must load either a run or a model")

    if run:
        logger.info(f"Retrieving model from run {run_id}")
        artifact_uri = (
            f"runs:/{run_id}/{mu.DEFAULT_ARTIFACT_PATH}/companies_matching_model.json"
        )
    elif model:
        if model_version is None:
            raise ValueError("If loading a model, version must be declared")

        logger.info(f"Retrieving {model_name} version {model_version} from model API")
        artifact_uri = du.collapse_multiline_string(
            f"""
            models:/
            {model_name}/
            {model_version}/
            {mu.DEFAULT_ARTIFACT_PATH}/
            companies_matching_model.json
        """
        )

    json_raw = mlflow.artifacts.load_text(artifact_uri=artifact_uri)
    json_settings = json.loads(json_raw)

    # Load data
    logger.info("Loading data")

    connection = duckdb.connect()
    data = du.build_alias_path_dict(input_dir)

    # Instantiate linker, load settings
    logger.info("Instantiating linker and loading settings")
    linker = DuckDBLinker(
        list(data.values()),
        settings_dict=settings,
        connection=connection,
        input_table_aliases=list(data.keys()),
    )
    linker.load_model(json_settings)

    # Predict/cluster/create lookup
    logger.info("Building lookup")
    lookup = predict_clusters(linker, threshold=0.7)

    # Write lookup to output
    logger.info("Writing lookup")
    du.data_workspace_write(
        schema=output_schema, table=output_table, df=lookup, if_exists="replace"
    )

    logger.info("Done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    load_dotenv(find_dotenv())

    main()
