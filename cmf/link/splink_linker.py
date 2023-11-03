from cmf.link.linker import Linker
from cmf.data import utils as du
from cmf.data.datasets import Dataset
from cmf.data.probabilities import Probabilities
from cmf.data.clusters import Clusters

from splink.duckdb.linker import DuckDBLinker
from splink.comparison import Comparison

import json
import pandas as pd


class ComparisonEncoder(json.JSONEncoder):
    """
    Splink functions can't be encoded to JSON by default, but can be
    represented as dictionaries with their own implemented as_dict() method.
    This class allows json.dumps to use that method when encoding.
    """

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

    Uses an internal lookup table unique_id_lookup to minimise memory
    usage during linking. Will create this during a job, and re-join the
    correct data back on afterwards.

    Attributes:
        * linker: The Splink Linker object
        * db_path: The path to the duckDB database this linker uses
        * con: The connection object for the duckDB database
        * id_lookup: A lookup of IDs to minimise strings in memory
        * predictions: The dataset of predictions, once made

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
        self,
        name: str,
        dataset: Dataset,
        probabilities: Probabilities,
        clusters: Clusters,
        n: int,
        overwrite: bool = False,
        db_path: str = ":memory:",
    ):
        """
        Parameters:
            * name: The name of the linker model you're making. Should be unique --
            link outputs are keyed to this name
            * dataset: An object of class Dataset
            * probabilities: An object of class Probabilities
            * clusters: An object of class Clusters
            * n: The current step in the pipeline process
            * overwrite: Whether the link() method should replace existing outputs
            of models with this linker model's name
            * db_path: [Optional] If writing to disk, the location to use
            for duckDB
        """
        super().__init__(name, dataset, probabilities, clusters, n, overwrite)

        self.linker = None
        self.db_path = db_path
        self.con = du.get_duckdb_connection(path=self.db_path)
        self.id_lookup = None
        self.predictions = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Only pickle linker settings
        state["linker"] = state["linker"]._settings_obj.as_dict()
        # Don't pickle connection
        del state["con"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add connection and linker back when loading pickle
        self.con = du.get_duckdb_connection(path=self.db_path)
        if self.cluster_processed is not None and self.dim_processed is not None:
            self._register_tables()
        if self.linker is not None:
            self._create_linker(linker_settings=self.linker)

    def _substitute_ids(self):
        cls_len = self.cluster_processed.shape[0]
        dim_len = self.dim_processed.shape[0]

        self.cluster_processed["duckdb_id"] = range(cls_len)
        self.dim_processed["duckdb_id"] = range(cls_len, cls_len + dim_len)

        self.id_lookup = pd.concat(
            objs=[
                self.cluster_processed[["duckdb_id", "id"]],
                self.dim_processed[["duckdb_id", "id"]],
            ],
            axis=0,
        )
        self.cluster_processed["id"] = self.cluster_processed["duckdb_id"]
        self.cluster_processed.drop("duckdb_id", axis=1, inplace=True)
        self.dim_processed["id"] = self.dim_processed["duckdb_id"]
        self.dim_processed.drop("duckdb_id", axis=1, inplace=True)

    def _register_tables(self):
        self.con.register("cls", self.cluster_processed)
        self.con.register("dim", self.dim_processed)

    def _create_linker(self, linker_settings: dict):
        self.linker = DuckDBLinker(
            input_table_or_tables=["cls", "dim"],
            input_table_aliases=["cls", "dim"],
            connection=self.con,
            settings_dict=linker_settings,
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

        model_json = json.dumps(
            self.linker._settings_obj.as_dict(), indent=4, cls=ComparisonEncoder
        )

        super()._add_log_item(
            name="model",
            item=model_json.encode(),
            item_type="artefact",
            path="model/model.json",
        )

    def prepare(
        self,
        cluster_pipeline: dict,
        dim_pipeline: dict,
        linker_settings: dict,
        train_pipeline: dict,
        low_memory: bool = True,
    ):
        """
        Runs all the linker's private cleaning, shaping and training methods,
        ready for linking.

        When low_memory is true, raw data is purged after processing.
        """
        self._clean_data(cluster_pipeline, dim_pipeline, delete_raw=low_memory)
        self._substitute_ids()
        self._register_tables()
        self._create_linker(linker_settings)
        self._train_linker(train_pipeline)

    def link(self, threshold: float, log_output: bool = False, overwrite: bool = None):
        """
        Runs the linker's link job.

        Note threshold is different to the threshold set in Clusters.add_clusters,
        which represents the threshold at which you believe a link to be a good one.
        It's likely you'll want this to be lower so you can use validation to discover
        the right Clusters.add_clusters threshold.

        Arguments:
            threshold: the probability threshold below which to drop outputs
            log_output: whether to write outputs to the final table. Likely False as
            you refine your methodology, then True when you're happy
            overwrite: whether to overwrite existing outputs keyed to the specified
            model name. Defaults to option set at linker instantiation

        Returns:
            The output of Splink's `predict()` method, but with original IDs rejoined
            and as a pandas dataframe
        """
        if overwrite is None:
            overwrite = self.overwrite

        self.predictions = self.linker.predict(threshold_match_probability=threshold)

        super()._add_log_item(
            name="link_threshold", item=str(threshold), item_type="parameter"
        )

        probabilities = (
            self.predictions.as_pandas_dataframe()
            .merge(
                right=self.id_lookup.rename(columns={"id": "cluster"}),
                how="left",
                left_on="id_l",
                right_on="duckdb_id",
            )
            .merge(
                right=self.id_lookup, how="left", left_on="id_r", right_on="duckdb_id"
            )
            .drop(columns=["id_l", "id_r"])
        )
        probabilities["source"] = self.dataset.db_id

        super()._add_log_item(
            name="match_pct",
            item=probabilities.id.nunique() / self.dim_processed.shape[0],
            item_type="metric",
        )

        if log_output:
            out = (probabilities.rename(columns={"match_probability": "probability"}))[
                ["cluster", "id", "probability", "source"]
            ]

            self.probabilities.add_probabilities(
                probabilities=out, model=self.name, overwrite=overwrite
            )

        return probabilities
