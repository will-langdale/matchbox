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

    def prepare(self):
        pass

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

        to_insert = None

        du.data_workspace_write(
            df=to_insert, schema=self.schema, table=self.table, if_exists="append"
        )
