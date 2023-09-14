from src.data import utils as du
from src.data.datasets import Dataset
import uuid
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("clusters")


class Clusters(object):
    """
    A class to interact with the company matching framework's clusters table.
    Enforces things are written in the right shape, and facilates easy
    retrieval of data in various shapes.

    Attributes:
        * schema: the cluster table's schema name
        * table: the cluster table's table name
        * schema_table: the cluster table's full name
        * star: an object of class Star that wraps the star table

    Methods:
        * create(dim=None, overwrite): Drops all data and recreates the cluster
        table. If a dimension table is specified, adds each row as a new cluster
        * read(): Returns the cluster table
        * add_clusters(probabilities): Using a probabilities table, adds new
        entries to the cluster table
        * get_data(fields): returns the cluster table pivoted wide,
        with the requested fields attached
    """

    def __init__(self, schema: str, table: str, star: object):
        self.schema = schema
        self.table = table
        self.schema_table = f'"{self.schema}"."{self.table}"'
        self.star = star

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
                uuid uuid primary key,
                cluster uuid not null,
                id text not null,
                source int not null,
                n int not null
            );
        """

        du.query_nonreturn(sql)

        if dim is not None:
            dim_table = self.star.get(star_id=dim, response="dim")

            sql = f"""
                select
                    id
                from
                    {dim_table}
            """

            to_insert = du.query(sql)
            to_insert["uuid"] = [uuid.uuid4() for _ in range(len(to_insert.index))]
            to_insert["cluster"] = [uuid.uuid4() for _ in range(len(to_insert.index))]
            to_insert["source"] = dim
            to_insert["n"] = 1

            du.data_workspace_write(
                df=to_insert, schema=self.schema, table=self.table, if_exists="append"
            )

    def read(self):
        return du.dataset(self.schema_table)

    def add_clusters(
        self, probabilities: str, validate: str, n: int, threshold: float = 0.7
    ):
        """
        The core probabilities > clusters algorithm, as proposed in the
        v0.2 output currently described in the README.

        1. Order probabilities from high to low probability
        2. Take the highest of each unique pair and add to cluster table
        3. Remove all members of matched pairs from either side of
        probabilities
        4. Repeat 2-3 until lookup is empty

        This algorithm should both work with one step in an additive
        pattern, or a big group of matches made concurrently.

        Arguments:
            probabilities: an object of class Probabilities
            validate: an object of class Validate

        """
        # TODO: implement once we've populated a probabilities table
        # DO NOT FORGET: use probabilities.source to get the dim and add
        # a new cluster for any rows that WEREN'T matched
        pass

    def get_data(self, select: dict, dim_only: bool = True, n: int = None):
        """
        Wrangle clusters and associated dimension fields into an output
        appropriate for the linker object to join a new dimension table onto.

        Returns a temporary dimension table that can use information from
        across the matched clusters so far to be the left "cluster" side of
        the next step n of the 🐙blocktopus architecture.

        Arguments:
            select: a dict where the key is a Postgres-quoted fact table, with
            values a list of fields you want from that fact table. We use fact
            table as all of these are on Data Workspace, and this will hopefully
            make a really interpretable dict to pass
            dim_only: force function to only return data from dimension tables.
            Used to build the left side of a join, where retrieving from the
            fact table would create duplication
            n: (optional) the step at which to retrive values

        Raises:
            KeyError:
                * If a requested field isn't in the dimension table and
                dim_only is True
                * If a requested field isn't in the fact or dimension table

            ValueError: if no data is returned

        Returns:
            A dataframe with one row per company entity, and one column per
            requested field
        """

        select_items = []
        join_clauses = ""
        if n is not None:
            n_clause = f"where cl.n < {n}"
        else:
            n_clause = ""

        for i, (table, fields) in enumerate(select.items()):
            data = Dataset(
                star_id=self.star.get(fact=table, response="id"),
                star=self.star,
            )
            if dim_only:
                cols = set(data.get_cols("dim"))

                if len(set(fields) - cols) > 0:
                    raise KeyError(
                        """
                        Requested field not in dimension table and dim_only
                        is True.
                    """
                    )

                for field in fields:
                    clean_name = f"{data.dim_table_clean}_{field}"
                    select_items.append(f"t{i}.{field} as {clean_name}")

                join_clauses += f"""
                    left join {data.dim_schema_table} t{i} on
                        cl.id = t{i}.id
                        and cl.source = {data.id}
                """
            else:
                cols = set(data.get_cols("fact"))

                if len(set(fields) - cols) > 0:
                    raise KeyError(
                        """
                        Requested field not in fact table.
                    """
                    )

                for field in fields:
                    clean_name = f"{data.fact_table_clean}_{field}"
                    select_items.append(f"t{i}.{field} as {clean_name}")

                join_clauses += f"""
                    left join {data.fact_schema_table} t{i} on
                        cl.id = t{i}.id
                        and cl.source = {data.id}
                """

        sql = f"""
            select
                cl.cluster,
                {', '.join(select_items)}
            from
                {self.schema_table} cl
            {join_clauses}
            {n_clause};
        """

        result = du.query(sql)

        if len(result.index) == 0:
            raise ValueError("Nothing returned. Something went wrong")

        return result
