from src.link.linker import Linker

from splink.duckdb.linker import DuckDBLinker
from splink.comparison import Comparison

import json
import io


class ComparisonEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return obj.__name__
        elif isinstance(obj, Comparison):
            return obj.as_dict()
        else:
            return json.JSONEncoder.default(self, obj)


class SplinkLinker(Linker):
    """
    A class to handle linking a dataset using Splink. Implements linking
    with DuckDB.

    Parameters:
        * dataset: An object of class Dataset
        * probabilities: An object of class Probabilities
        * clusters: An object of class Clusters
        * n: The current step in the pipeline process

    Methods:
        * get_data(): retrieves the left and right tables: clusters
        and dimensions
        * prepare(linker_settings, cluster_pipeline, dim_pipeline,
        train_pipeline): cleans the data using a data processing dict,
        creates a linker with linker_settings, then trains it with a
        train_pipeline dict
        * link(threshold): performs linking and returns a match table
        appropriate for Probabilities. Drops observations below the specified
        threshold
        * evaluate(): runs prepare() and link() and returns a report of
        their performance
    """

    def __init__(
        self, dataset: object, probabilities: object, clusters: object, n: int
    ):
        super().__init__(dataset, probabilities, clusters, n)

        self.linker = None

    def _clean_data(self, cluster_pipeline: dict, dim_pipeline: dict):
        self.cluster_processed = super()._run_pipeline(
            self.cluster_raw, cluster_pipeline
        )
        self.dim_processed = super()._run_pipeline(self.dim_raw, dim_pipeline)

    def _create_linker(self, linker_settings: dict):
        self.linker = DuckDBLinker(
            input_table_or_tables=[self.cluster_processed, self.dim_processed],
            settings_dict=linker_settings,
            input_table_aliases=[0, self.dataset.id],
        )

    def _train_linker(self, train_pipeline: dict):
        """
        Runs the pipeline of linker functions to train the linker object.

        Similar to _run_pipeline(), expects self.pipeline to be a dict
        of step keys with a value of a dict with "function" and "argument"
        keys. Here, however, the value of "function" should be a string
        corresponding to a method in the linker object. "argument"
        remains the same: a dictionary of method and value arguments to
        the referenced linker method.
        """
        for func in train_pipeline.keys():
            proc_func = getattr(self.linker, train_pipeline[func]["function"])
            proc_func(**train_pipeline[func]["arguments"])

        train_pipeline_json = json.dumps(
            train_pipeline, indent=4, cls=ComparisonEncoder
        )

        super()._add_log_item(
            name="train_pipeline",
            item=train_pipeline_json.encode(),
            item_type="artefact",
            path="config/train_pipeline.json",
        )

        model_json = io.BytesIO()
        self.linker.save_model_to_json(out_path=model_json.name, overwrite=True)

        super()._add_log_item(
            name="model",
            item=model_json.encode(),
            item_type="artefact",
            path="model/model.json",
        )

    def prepare(
        self,
        linker_settings: dict,
        cluster_pipeline: dict,
        dim_pipeline: dict,
        train_pipeline: dict,
    ):
        self._clean_data(cluster_pipeline, dim_pipeline)
        self._create_linker(linker_settings)
        self._train_linker(train_pipeline)

    def link(self, threshold: int, log_output: bool = True):
        predictions = None

        if log_output:
            self.probabilities.add_probabilities(predictions)

        return predictions
