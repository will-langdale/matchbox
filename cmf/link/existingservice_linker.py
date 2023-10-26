from cmf.link.deterministic_linker import DeterministicLinker
from cmf.data import utils as du
from cmf.data.datasets import Dataset
from cmf.data.probabilities import Probabilities
from cmf.data.clusters import Clusters


class ExistingCMSPlusLinker(DeterministicLinker):
    """
    A class to handle an extended version of the existing Company Matching
    Service's methodology.

    In the existing service, six items are matched deterministically to give a
    vector of binary values. The service has no opinion on how to combine
    these values to identify a positive match. The possible items are:

    * Companies House ID
    * Dun & Bradstreet ID
    * Company name
    * Contact email
    * CDMS ID
    * Company postcode

    This class attempts to replicate the first part of this methodology as
    close as possible, calculating a vector of binary values for whichever
    fields are cleaned and passed to it.

    The methodology is extended to take this vector, combine it with weights
    for each value, then combine and scale it to 0-1, giving an interpretable
    probability. In this way we hope to produce as close as possible the best
    possible usage of the existing service based on knowledge of the datasets
    it connects, even though this requires taking an opinion on matches that
    the service doesn't currently give.

    Inherits from DeterministicLinker as the methodology just extends the
    link_settings and replaces the link() method. prepare() is identical.

    Note prepare(link_settings) now requires an extra field: weight. An exact
    match between the cluster and dimension fields will be scaled by 1*weight.

    Attributes:
        * linker: The settings used to link
        * con: The connection object for the duckDB database
        * predictions: The dataset of predictions, once made

    Methods:
        * get_data(): retrieves the left and right tables: clusters
        and dimensions
        * prepare(): cleans the data using a data processing dict,
        establishes which fields to match on with link_settings, and the
        weight each field should be given
        * link(): performs linking and returns a match table appropriate for
        Probabilities
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
        """
        super().__init__(name, dataset, probabilities, clusters, n, overwrite)

        self.con = du.get_duckdb_connection(path=":memory:")
        self.linker = None
        self.predictions = None

    def _register_tables(self):
        self.con.register("cls", self.cluster_processed)
        self.con.register("dim", self.dim_processed)

    def link(
        self, threshold: float = None, log_output: bool = False, overwrite: bool = None
    ):
        """
        Links the datasets deterministically by the supplied field(s), scales
        the link vector by the match weights, then scales all vectors to
        a single probability.

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

        Raises:
            ValueError: if threshold is not a valid probability

        Returns:
            The linked dataframe as it was added to the probabilities table.
        """
        if overwrite is None:
            overwrite = self.overwrite

        if threshold is None:
            threshold = 0

        if threshold > 1 or threshold < 0:
            raise ValueError(
                f"""
                    threshold must be a valid probability between 0 and 1
                    Current value: {threshold}
                """
            )

        match_subquery = []
        weights = []

        for link in self.linker.values():
            cls = link["cluster"]
            dim = link["dimension"]
            wgt = link["weight"]

            match_subquery.append(
                f"""
                select
                    cls.id as cluster,
                    dim.id::text as id,
                    1 * {wgt} as probability
                from
                    cls cls
                left join
                    dim dim on
                        cls.{cls} = dim.{dim}
                where
                    dim.id is not null
                """
            )
            weights.append(wgt)

        match_subquery = " union all ".join(match_subquery)
        total_weight = sum(weights)

        self.predictions = self.con.sql(
            f"""
            select
                matches.cluster,
                matches.id,
                {self.dataset.id} as source,
                sum(matches.probability) / {total_weight} as probability
            from
                ({match_subquery}) matches
            group by
                matches.cluster,
                matches.id
            having
                sum(matches.probability) / {total_weight} > {threshold}
        """
        )

        out = self.predictions.df()

        super()._add_log_item(
            name="link_threshold", item=str(threshold), item_type="parameter"
        )

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
