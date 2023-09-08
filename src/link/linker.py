from src.data import utils as du
from src.link import model_utils as mu

import mlflow
import logging

# from pathlib import Path
import io

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

        self.report_artefacts = {}
        self.report_parameters = {}
        self.report_metrics = {}

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

        During a run, use the _add_log_item(item_type='artefact') method to record
        items you want evaluate() to save. Examples are plots, datasets or JSON
        objects.

        During a run, use the _add_log_item(item_type='parameter') method to record
        method parameters you want evaluate() to save. Examples are the
        Jaro-Winkler fuzzy matching value above which you consider something a
        match.

        Returns
            Bool indicating whether code was run.
        """
        return False

    def link(self, log_output: bool = True):
        """
        Runs whatever linking logic the subclass implements. Must finish by
        optionally calling Probabilities.add_probabilities(predictions), and then
        returning those predictions.

        During a run, use the _add_log_item(item_type='artefact') method to record
        items you want evaluate() to save. Examples are plots, datasets or JSON
        objects.

        During a run, use the _add_log_item(item_type='parameter') method to record
        method parameters you want evaluate() to save. Examples are the
        Jaro-Winkler fuzzy matching value above which you consider something a
        match.

        Arguments:
            * log_output: whether to log outputs to the probabilities table
        """
        raise NotImplementedError("method link() must be implemented")

        predictions = None

        if log_output:
            self.probabilities.add_probabilities(predictions)

        return predictions

    def _add_log_item(
        self,
        name: str,
        item: object,
        item_type: str,
        path: str = None,
    ):
        """
        Adds an item to either the artefact, metric or parameter dictionary,
        ready to be recorded as part of a report in evaluate(). When using
        MLflow, this is attached to the run in the specified directory.

        Subclasses should not use item_type='metric', as all matching methods
        should be comparable.

        Arguments:
            name: the unique name the artifact will be keyed to
            path: [Optional] if saving an artefact, the relative path you want it
            saved in, including the name and file extension you want to use
            item: the object you want to save. Requires:
                * object, if item_type is 'artefact'
                * string, if item_type is 'parameter'
                * numeric, if item_type is 'metric'
            item_type: the type of item you're saving. One of 'artefact',
            'parameter' or 'metric'

        Raises:
            ValueError:
                * if one of 'artefact', 'parameter' or 'metric' not passed to
                item_type
                * if path not set when item_type is 'artefact'
            TypeError: if an unacceptable datatype is passed for the item_type
        """
        # TODO: prevent key duplication in same run, allow between runs

        if item_type not in ["artefact", "parameter", "metric"]:
            raise ValueError(
                """
                item_type must be one of 'artefact', 'parameter' or 'metric'
            """
            )

        if item_type == "artefact":
            if path is None:
                raise ValueError(
                    """
                    If item_type is 'artefact', must specify path
                """
                )
            self.report_artefacts[name] = {"path": path, "artefact": item}
        elif item_type == "parameter":
            if not isinstance(item, str):
                raise TypeError("Parameters must be logged as strings")
            self.report_parameters[name] = {"name": name, "value": item}
        elif item_type == "metric":
            if not isinstance(item, int):
                raise TypeError("Metrics must be logged as strings")
            self.report_metrics[name] = {"name": name, "value": item}

    def evaluate(
        self,
        link_experiment: str,
        evaluation_name: str,
        evaluation_description: str,
        log_mlflow: bool = False,
        log_output: bool = False,
    ) -> dict:
        """
        Runs the prepare() and link() functions, and records evaluations.

        Arguments:
            * link_experiment: the experiment for the link, defined in config
            * evaluation_name: the name of this specific evaluation run
            * evaluation_description: a description of this specific
            evaluation run
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
                experiment_name=link_experiment,
                run_name=evaluation_name,
                description=evaluation_description,
                dev_mode=True,
            ):
                logger.info("Running prepare() function")
                self.prepare()

                logger.info("Running link() fnction")
                self.link()

                # TODO: Evaluation method based on validation table
                # Table not yet implemented

                for artefact in self.report_parameters.keys():
                    path = self.report_parameters[artefact]["path"]
                    artefact = self.report_parameters[artefact]["artefact"]

                    with io.BytesIO(artefact) as f:
                        mlflow.log_artifact(local_dir=f.name, artifact_path=path)

                for param in self.report_parameters.keys():
                    mlflow.log_param(
                        key=self.report_parameters[param]["name"],
                        value=self.report_parameters[param]["value"],
                    )

                for metric in self.report_metrics.keys():
                    mlflow.log_metric(
                        key=self.report_parameters[metric]["name"],
                        value=self.report_parameters[metric]["value"],
                    )

                # TODO: Make dict of outputs to return

        else:
            logger.info("Experiment not automatically logged")
            # TODO: non ML-flow logic, potentially with dir to save to
            # Use shutil to save objects to the path specifid
            # https://docs.python.org/3/library/shutil.html
            pass

        logger.info("Done!")

        return None
