# @click.command()
# @click.option(
#     "--input_name", required=True, type=str, help="Name of input data extract"
# )
# @click.option("--run_name", required=True, type=str, help="Name of model run")
# @click.option(
#     "--seed",
#     default=None,
#     type=int,
#     help="Seed used for non-deterministic components of the model",
# )
# @click.option(
#     "--dev",
#     is_flag=True,
#     help="""Dev runs allow to run this script with a dirty git repo""",
# )
# @click.option(
#     "--description",
#     default=None,
#     type=str,
#     help="Description of the training run",
# )
# def train_model(input_name, run_name, seed, dev, description):
#     logger = logging.getLogger(__name__)

#     logger.info("Loading training data")
#     df = du.load_df("processed", input_name)

#     logger.info("Training model")

#     with mu.mlflow_run(
#         run_name=run_name,
#         description=description,
#         dev_mode=dev,
#     ):
#         params = {"n_estimators": 5, "random_state": seed}
#         mlflow.log_params(params)

#         model = RandomForestRegressor(**params)

#         # ...train model...
#         # ...evaluate model and log metrics and artifacts...

#         prediction_dependencies = {
#             "scikit-learn": sklearn.__version__,
#         }

#         logger.info("Done. Serialising model and logging it to MLFlow.")
#         all_model_steps = (model.predict,)
#         mu.log_python_pipeline(all_model_steps, prediction_dependencies)


# def main():
#     """
#     Entrypoint
#     """
#     train_model()


# if __name__ == "__main__":
#     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     main()

import click
import logging
import time
from pathlib import Path

# import pandas as pd

# import mlflow

from src.data.make_dataset import (
    comp_house_read,
    # export_wins_read,
    # hmrc_exporters_read,
    data_hub_read,
)
from src.features.splink_cleaning_complex import clean_numbers_and_names
from src.config import settings

# from src.data import utils as du
import src.locations as loc

from splink.duckdb.linker import DuckDBLinker
from splink.charts import save_offline_chart


def produce_outputs_of_training(linker):
    """
    Produce outputs such as predictions table,
    comparison viewer, clusters, linker settings
    Args: linker model
    Returns: None
    """

    # What match weights have been estimated?
    # linker.match_weights_chart()
    # df_predict = None
    # linker = None
    # https://moj-analytical-services.github.io/splink/topic_guides/querying_splink_results.html
    # generate all predictions with prob > threshold_match_probability
    predictions = linker.predict(threshold_match_probability=0.1)

    # turn the predictions into a dataframe
    # each row is a pairwise comparison
    df_predict = predictions.as_pandas_dataframe()

    # look at how many values are missing in the various columns
    timestr = time.strftime("%Y%m%d-%H%M%S")
    c = linker.completeness_chart()
    file_name = "completeness_chart_" + timestr + ".html"
    logging.info("Saving completeness chart to %s", file_name)
    output_file = Path(loc.OUTPUTS_HOME, "figures", file_name)
    save_offline_chart(c.spec, filename=output_file, overwrite=True)

    c = linker.missingness_chart()
    file_name = "missingness_chart_" + timestr + ".html"
    logging.info("Saving missingness chart to %s", file_name)
    output_file = Path(loc.OUTPUTS_HOME, "figures", file_name)
    save_offline_chart(c.spec, filename=output_file, overwrite=True)

    file_name = "comparison_viewer_" + timestr + ".html"
    logging.info("Saving comparison viewer to %s", file_name)
    output_file = Path(loc.OUTPUTS_HOME, "figures", file_name)
    linker.comparison_viewer_dashboard(predictions, output_file, overwrite=True)

    # Cluster the pairwise match predictions that result from linker.predict()
    # into groups of connected record using the connected components
    # graph clustering algorithm
    # A cluster is considered to be 'the same entity'
    clusters = linker.cluster_pairwise_predictions_at_threshold(
        predictions,
        threshold_match_probability=0.5,
        pairwise_formatting=False,
        filter_pairwise_format_for_clusters=True,
    )
    df_clusters = clusters.as_pandas_dataframe()

    # not producing anything
    file_name = "cluster_studio_" + timestr + ".html"
    logging.info("Saving cluster studio to %s", file_name)
    output_file = Path(loc.OUTPUTS_HOME, "figures", file_name)
    linker.cluster_studio_dashboard(predictions, clusters, out_path=output_file)

    # save the results
    file_name = "linker_settings_" + timestr + ".json"
    logging.info("Saving linker settings to %s", file_name)
    output_file = Path(loc.OUTPUTS_HOME, "linker_settings", file_name)
    linker.save_model_to_json(output_file, overwrite=True)

    file_name = "df_predict_" + timestr + ".pkl"
    logging.info("Saving dataframe of predictions to %s", file_name)
    output_file = Path(loc.OUTPUTS_HOME, "predictions_and_clusters", file_name)
    df_predict.to_pickle(output_file)

    file_name = "df_clusters_" + timestr + ".pkl"
    logging.info("Saving dataframe of clusters to %s", file_name)
    output_file = Path(loc.OUTPUTS_HOME, "predictions_and_clusters", file_name)
    df_clusters.to_pickle(output_file)

    file_name = "m_u_parameters_chart_" + timestr + ".html"
    logging.info("Saving m and u probabilities chart to %s", file_name)
    c = linker.m_u_parameters_chart()
    output_file = Path(loc.OUTPUTS_HOME, "figures", file_name)
    save_offline_chart(c.spec, filename=output_file, overwrite=True)

    return None


@click.command()
@click.option(
    "--retrain/--no-train",
    default=False,
    show_default=True,
    help="Whether to retrain a new model",
)
def main(retrain):
    """
    Entrypoint
    """

    if retrain:
        logging.info("Retraining a new Splink model")

        logging.info("Retrieving the data to link")
        # read in Companies House data, return company_number, postcodes
        # and company_name split into: 'unusual' tokens,
        # most common 3 tokens and most common 4 to 6 tokens
        df_ch = comp_house_read()
        df_ch_clean = clean_numbers_and_names(df_ch)
        df_ch_clean.reset_index(inplace=True)
        df_ch_clean.rename(columns={"index": "unique_id"}, inplace=True)

        # as above but for Data Hub data
        df_dh = data_hub_read()
        df_dh_clean = clean_numbers_and_names(df_dh)
        df_dh_clean.reset_index(inplace=True)
        df_dh_clean.rename(columns={"index": "unique_id"}, inplace=True)

        # instantiate the linker
        linker = DuckDBLinker(
            [df_dh_clean, df_ch_clean],
            settings,
            input_table_aliases=["datahub", "companies_house"],
        )

        # This is how you do a deterministic link
        # It uses whatever rules you specify in 'blocking_rules_to_generate_predictions'
        # linker.deterministic_link().as_pandas_dataframe()

        # Determine probability two random records match i.e. the prior
        # Should admit very few (none if possible) false positives
        # https://moj-analytical-services.github.io/splink/linkerest.html#splink.linker.
        # Linker.estimate_probability_two_random_records_match
        # this assumption is important to what we think 'a company is'
        # if just using equality on name, we - for instance -
        # think astrazeneca cambridge and astrazeneca macclesfield are 'the same' comp
        # may need revisiting / alternative models building
        linker.estimate_probability_two_random_records_match(
            "l.name_unusual_tokens = r.name_unusual_tokens",
            recall=0.7,
        )

        # But let's do probabilistic linkage instead
        # increased max_pairs so that the model more likely
        #  to encounter the required comparison levels
        # NOTE: random sampling and can't set seed anymore
        linker.estimate_u_using_random_sampling(max_pairs=1e7)

        # If we can treat company number as a partially-completed label
        # we can estimate the m values from the numbers
        linker.estimate_m_from_label_column("comp_num_clean")

        produce_outputs_of_training(linker)

    else:
        logging.info(
            """
        This option does not do anything yet.
        Need to work out which outputs of Splink training to persist, and how"""
        )

    return None


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
