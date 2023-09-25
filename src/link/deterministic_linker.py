from src.link.linker import Linker
from src.data import utils as du
from src.data.datasets import Dataset
from src.data.probabilities import Probabilities
from src.data.clusters import Clusters


class DeterministicLinker(Linker):
    """
    A class to handle deterministic linking: either a match is achieved or
    it isn't. In the probabilistic framework, can only ever reach one point
    in the ROC curve.

    Parameters:
        * name: The name of the linker model you're making. Should be unique --
        link outputs are keyed to this name
        * dataset: An object of class Dataset
        * probabilities: An object of class Probabilities
        * clusters: An object of class Clusters
        * n: The current step in the pipeline process
        * overwrite: Whether the link() method should replace existing outputs
        of models with this linker model's name

    Methods:
        * get_data(): retrieves the left and right tables: clusters
        and dimensions
        * prepare(): cleans the data using a data processing dict,
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
    ):
        super().__init__(name, dataset, probabilities, clusters, n, overwrite)

        self.con = du.get_duckdb_connection(path=":memory:")
        self.linker = None
        self.predictions = None

    def _register_tables(self):
        self.con.register("cls", self.cluster_processed)
        self.con.register("dim", self.dim_processed)

    def prepare(
        self,
        cluster_pipeline: dict,
        dim_pipeline: dict,
        link_settings: dict,
        low_memory: bool = False,
    ):
        """
        Cleans the data using the supplied dictionaries of functions.

        Controls which fields should be exactly matched post-cleaning. Expects a
        dictionary where keys are the field in the cluster table, and values are
        the comparable field in the dimension table.

        When low_memory is true, raw data is purged after processing.

        Raises:
            KeyError: if the linker settings use fields not present in the cleaned
            datasets
        """
        self.linker = link_settings
        self._clean_data(cluster_pipeline, dim_pipeline, delete_raw=low_memory)

        cls_cols = set(self.linker.keys())
        dim_cols = set(self.linker.values())

        cls_proc_cols = set(self.cluster_processed.columns)
        dim_proc_cols = set(self.dim_processed.columns)

        if len(cls_cols.intersection(cls_proc_cols)) != len(cls_cols):
            missing = ", ".join(cls_proc_cols.difference(cls_cols))
            raise KeyError(
                f"""
                Specified columns {missing} not present in processed cluster
                data.
            """
            )

        if len(dim_cols.intersection(dim_proc_cols)) != len(dim_cols):
            missing = ", ".join(dim_proc_cols.difference(dim_cols))
            raise KeyError(
                f"""
                Specified columns {missing} not present in processed dimension
                data.
            """
            )

        self._register_tables()

    def link(self, log_output: bool = True, overwrite: bool = None):
        """
        Links the datasets deterministically by the supplied field(s).

        Arguments:
            log_output: whether to write outputs to the final table. Likely False as
            you refine your methodology, then True when you're happy
            overwrite: whether to overwrite existing outputs keyed to the specified
            model name. Defaults to option set at linker instantiation

        Returns:
            The linked dataframe as it was added to the probabilities table.
        """
        if overwrite is None:
            overwrite = self.overwrite

        join_clause = []
        for cls_field, dim_field in self.linker.items():
            join_clause.append(f"cls.{cls_field} = dim.{dim_field}")
        join_clause = " and ".join(join_clause)

        self.predictions = self.con.sql(
            f"""
            select
                cls.id as cluster,
                dim.id::text as id,
                {self.dataset.id} as source,
                case when
                    dim.id is null
                then
                    0
                else
                    1
                end as probability
            from
                cls cls
            left join
                dim dim on
                    {join_clause}
            where
                dim.id is not null
                and cls.id is not null
        """
        )

        out = self.predictions.df()

        super()._add_log_item(
            name="match_pct",
            item=out.id.nunique() / self.dim_processed.shape[0],
            item_type="metric",
        )

        if log_output:
            self.probabilities.add_probabilities(
                probabilities=out, model=self.name, overwrite=overwrite
            )

        return out
