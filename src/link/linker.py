from src.data import utils as du
from src.link import model_utils as mu

# import mlflow
import logging

# import json
# from os import path, makedirs

"""
What does ANY linker neeed?

* The left data: cluster data, pivoted wide, with fields to join
    * Call cluster data method from Clusters
* The right data: dim table data
    * Call dim retrieval method from Dataset
* A prepare method. An optional bit for subclasses to fill in
    * Should require dict parameterisation
    * Preprocessing handled here, even if called from new methods
    * Linker training handled here, even if called from new methods
* A link method to output data. A bit subclasses MUST fill in
    * Optional experiment parameter
    * Ouput as df or direct to probabilities table?
        * Add experiment to Probabilities table so we can compare outputs
* An evaluate method
    * With option to use MLFlow
    * With option to output to Probabilities table
    * Against a single experiment ID (MLflow or otherwise)
        * Runs prepare
        * Runs link
        * Runs a standard report

What does ONE linker need?

* The above, but
    * link method must contain more code
    * prepare method might contain more code

"""


class Linker(object):
    """
    A class to build attributes and methods shared by all Linker subclasses.
    Standardises:

    * Retrieving the left table: cluster data
    * Retrieving the right table: dimension data
    * The structure of the core linking methods: prepare() and link()
    * The output shape and location
    * The evaluation and reporting methodology

    Assumes a Linker subclass will be instantiated with:

    * Dicts of settings that configure its process
    * A single step in the link_pipeline in config.py

    Parameters:
        * dataset: An object of class Dataset
        * probabilities: An object of class Probabilities
        * clusters: An object of class Clusters
        * n: The current step in the pipeline process

    Methods:
        * get_data(): retrieves the left and right tables: clusters
        and dimensions
        * prepare(): a method intended for linkers that need to clean data
        and train model parameters. Can output None to be skipped
        * link(): performs linking and returns a match table appropriate
        for Probabilities
        * evaluate(): runs prepare() and link() and returns a report of
        their performance
    """

    def __init__(
        self, dataset: object, probabilities: object, clusters: object, n: int
    ):
        self.dataset = dataset
        self.probabilities = probabilities
        self.clusters = clusters
        self.n = n

        self.dim_raw = None
        self.cluster_raw = None

        self.dim_processed = None
        self.cluster_processed = None

    def get_data(self, dim_fields: list, fact_fields: list):
        self.dim_raw = self.dataset.read_dim(dim_fields)
        self.cluster_raw = self.clusters.get_data(fact_fields)

    def _run_pipeline(self, table_in, pipeline):
        """
        Runs a pipeline of functions against an input table and
        returns the result.

        Arguments:
            table_in: The table to be processed
            pipeline: A dict with where each key is a pipeline step,
            and the value is another dict with keys "function" and
            "arguments". "function"'s value is a callable function
            that returns the input for the next step, and "arguments"
            should contain a dictionary of method and value arguments
            to the function.

        Returns:
            A output table with the pipeline of functions applied

        Examples:
            _run_pipeline(
                table_in = self.table,
                pipeline = {
                    "add_two_columns": {
                        "function": add_columns,
                        "arguments": {
                            "left_column": "q1_q2_profit",
                            "right_column": "q3_q4_profit",
                            "output": "year_profit"
                        }
                    },
                    "delete_columns": {
                        "function": delete_columns,
                        "arguments": {
                            "columns": ["q1_q2_profit", "q3_q4_profit"]
                        }
                    }
                }
            )
        """
        curr = table_in
        for func in pipeline.keys():
            curr = pipeline[func]["function"](curr, **pipeline[func]["arguments"])
        return curr

    def prepare(self):
        """
        An optional method for functions like data cleaning and linker training.
        If you don't use it, must return False. If you do, must return True.

        Returns
            Bool indicating whether code was run.
        """
        return False

    def link(self, log_output: bool = True):
        """
        Runs whatever linking logic the subclass implements. Must finish by
        optionally calling Probabilities.add_probabilities(predictions), and then
        returning those predictions.

        Arguments:
            * log_output: whether to log outputs to the probabilities table
        """
        raise NotImplementedError("method link() must be implemented")

        predictions = None

        if log_output:
            self.probabilities.add_probabilities(predictions)

        return predictions

    def evaluate(self, log_mlflow: bool = False, log_output: bool = False) -> dict:
        """
        Runs the prepare() and link() functions, and records evaluations.

        Arguments:
            * log_mlflow: whether to use MLflow to log this run
            * log_output: whether to log outputs to the probabilities table

        Returns:
            A dict of analysis
        """
        logging.basicConfig(
            level=logging.INFO,
            format=du.LOG_FMT,
        )
        logger = logging.getLogger(__name__)

        logger.info("Running pipeline")

        if log_output:
            logger.info("Logging outputs to the Probabilities table")

        if log_mlflow:
            logger.info("Logging as MLflow experiment")
            with mu.mlflow_run(
                # TODO: configure
                # run_name=None,
                # experiment_name=None,
                # description=None,
                dev_mode=True,
            ):
                logger.info("Running prepare() function")
                self.prepare()

                logger.info("Running link() fnction")
                self.link()

                # TODO: Artifacts and parameters
                # for key, value in kwargs.items():
                #     mlflow.log_param(key=key, value=value)

        else:
            logger.info("Experiment not automatically logged")
            # TODO: non ML-flow logic
            pass

        logger.info("Done!")
