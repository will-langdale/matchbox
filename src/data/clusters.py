from src.data import utils as du


class Clusters(object):
    """
    A class to interact with the company matching framework's clusters table.
    Enforces things are written in the right shape, and facilates easy
    retrieval of data in various shapes.

    Attributes:
        * schema: the cluster table's schema name
        * table: the cluster table's table name
        * schema_table: the cluster table's full name

    Methods:
        * create(dim=None, overwrite): Drops all data and recreates the cluster
        table. If a dimension table is specified, adds each row as a new cluster
        * read(): Returns the cluster table
        * add_clusters(lookup): Using a lookup table, adds new entries
        to the cluster table
        * get_data(fields): returns the cluster table pivoted wide,
        with the requested fields attached
    """

    def __init__(self, schema: str, table: str):
        self.schema = schema
        self.table = table
        self.schema_table = f'"{self.schema}"."{self.table}"'

    def create(self, overwrite: bool, dim: int = None):
        """
        Creates a new cluster table. If a dimension table is specified, adds
        each row as a new cluster to the recreated table.

        Arguments:
            dim: [Optional] The STAR ID of a dimension table to populate the
            new cluster table with
            overwrite: Whether or not to overwrite an existing cluster table
        """

        if overwrite:
            drop = f"drop table if exists {self.schema_table};"
            exist_clause = ""
        else:
            drop = ""
            exist_clause = "if not exists"

        sql = f"""
            {drop}
            create table {exist_clause} {self.schema_table} (
                uuid uuid
                cluster uuid
                id text
                source int
                n int
            );
        """

        du.query_nonreturn(sql)

        if dim is not None:
            pass

    def get_data(self, fields: list):
        """
        Build the cluster table at point n in the 🐙blocktopus build process.

        Arguments:
            cluster_table: The location of the cluster table
            fields: The data to retrieve from the cluster's dimension tables

        Returns:
            A dataframe with one row per company entity, and one column per
            requested field
        """

        data = None

        return data
