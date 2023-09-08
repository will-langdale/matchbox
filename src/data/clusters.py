from src.data import utils as du
import uuid


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

    def add_clusters(self, probabilities):
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

        Instinctively, I think this should read the Postgres table but
        perform the algorithm in memory. We'll see.

        Arguments:
            dataset: an instantiated dataset object corresponding to the
            dimension side of the linker job that produced probabilities
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

        Returns:
            A dataframe with one row per company entity, and one column per
            requested field
        """
        # TODO: implement as part of Linker class update
        # build left join clauses
        # for table, fields in select.items()
        # get id from STAR and create dataset object
        # if dim_only is true
        # check everything in fields is in the dim
        # else
        # check everything in fields is in the dim or fact
        # use dataset obj to add to left join clause
        # left_clause = """
        #     left join companieshouse.companies t1 on
        #         cl.id = t1.id
        #         and cl.source = 1970
        # """
        # retrieve data from clusters, joining on
        # sql = """
        #     select
        #         cl.cluster,
        #         t1.company_name as companieshouse_companies_company_name,
        #         t1.company_number as companieshouse_companies_company_number
        #     from
        #         "_user_eaf4fd9a"."cm_clusters" cl
        #     left join
        #         companieshouse.companies t1 on
        #             cl.id = t1.id
        #             and cl.source = 1970
        #     where
        #         cl.n < 2
        #     limit 50;
        # """
        pass
