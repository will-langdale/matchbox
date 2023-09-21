from src.data import utils as du
from src.data.datasets import Dataset
from src.data.probabilities import Probabilities
from src.data.validation import Validation
from src.data.star import Star

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

    def __init__(self, schema: str, table: str, star: Star):
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
        self,
        probabilities: Probabilities,
        validation: Validation,
        n: int,
        threshold: float = 0.7,
        add_unmatched_dims: bool = True,
    ):
        """
        The core probabilities > clusters algorithm, as proposed in the
        v0.2 output currently described in the README.

        Reads from the supplied probabilities and validation objects. Writes
        to the current clusters table.

        1. Order probabilities from high to low probability
        2. Take the highest of each unique pair and add to cluster table
            a. On a tie, takes the lowest ID as an arbitrary tiebreaker
        3. Remove all members of matched pairs from either side of
        probabilities
        4. Repeat 2-3 until lookup is empty

        This algorithm should both work with one step in an additive
        pattern, or a big group of matches made concurrently.

        Arguments:
            probabilities: an object of class Probabilities
            validation: an object of class Validation
            n: the current step n of the link pipeline
            threshold: the probability threshold above which we consider a match
            valid for the current linking method
            add_unmatched_dims: if True, adds unmatched rows in the dimension table
            to the clusters table. Should always be True for a link pipeline -- False
            is useful for testing
        """
        prob = probabilities.schema_table
        val = validation.schema_table
        clus = self.schema_table
        clusters_temp = "clusters_temp"
        probabilities_temp = "probabilities_temp"
        to_insert_temp = "to_insert_temp"

        # Create a temporary clusters table to work with until the algorithm has
        # finished, for safety
        du.query_nonreturn(
            f"""
            drop table if exists {clusters_temp};
            create temporary table {clusters_temp} as
                select
                    uuid,
                    cluster,
                    id,
                    source,
                    n
                from
                    {clus}
                union
                select
                    gen_random_uuid() as uuid,
                    cluster,
                    id,
                    source,
                    {n} as n
                from
                    {val}
                where
                    source in (
                        select
                            source
                        from
                            {prob}
                    );
        """
        )

        # Create a temporary probabilities table so we can delete stuff safely
        du.query_nonreturn(
            f"""
            drop table if exists {probabilities_temp};
            create temporary table {probabilities_temp} as
                select
                    uuid,
                    link_type,
                    cluster,
                    id,
                    source,
                    probability
                from
                    {prob} prob
                where
                    prob.probability >= {threshold}
                order by
                    probability desc;
        """
        )

        # Find what we need to insert by comparing clusters_temp and probabilities_temp
        # Insert it into clusters_temp
        # Delete it from probabilities_temp
        # Keep going until there's nothing to find
        data_to_insert = True
        while data_to_insert:
            du.query_nonreturn(
                f"""
                drop table if exists {to_insert_temp};
                create temporary table {to_insert_temp} as
                    select
                        distinct on (id_rank.id, id_rank.source)
                        gen_random_uuid() as uuid,
                        id_rank.cluster,
                        id_rank.id,
                        id_rank.source,
                        {n} as n
                    from (
                        select
                            distinct on (clus_rank.cluster, clus_rank.source)
                            clus_rank.*,
                            rank() over (
                                partition by
                                    clus_rank.id,
                                    clus_rank.source
                                order by
                                    clus_rank.probability desc
                            ) as id_rank
                        from (
                            select
                                prob.*,
                                rank() over(
                                    partition by
                                        prob.cluster,
                                        prob.source
                                    order by
                                        prob.probability desc
                                ) as clus_rank
                            from
                                {probabilities_temp} prob
                        ) clus_rank
                        where
                            clus_rank.clus_rank = 1
                            and (
                                not exists (
                                    select
                                        id,
                                        source
                                    from
                                        {clusters_temp} clus
                                    where
                                        clus.id = clus_rank.id
                                        and clus.source = clus_rank.source
                                )
                                or not exists (
                                    select
                                        cluster,
                                        source
                                    from
                                        {clusters_temp} clus
                                    where
                                        clus.cluster = clus_rank.cluster
                                        and clus.source = clus_rank.source
                                )
                            )
                        order by
                            clus_rank.cluster,
                            clus_rank.source
                    ) id_rank
                    where
                        id_rank.id_rank = 1
                    order by
                        id_rank.id,
                        id_rank.source;
            """
            )

            if du.check_table_empty(f"{to_insert_temp}"):
                data_to_insert = False
                break

            du.query_nonreturn(
                f"""
                insert into {clusters_temp}
                select
                    uuid,
                    cluster,
                    id,
                    source,
                    n
                from
                    {to_insert_temp};
            """
            )

            du.query_nonreturn(
                f"""
                delete from {probabilities_temp} prob_temp
                where exists (
                    select
                        cl.cluster,
                        cl.id,
                        cl.source
                    from
                        {to_insert_temp} cl
                    where
                        (
                            cl.id = prob_temp.id
                            and cl.source = prob_temp.source
                        )
                        or (
                            cl.cluster = prob_temp.cluster
                            and cl.source = prob_temp.source
                        )
                );
            """
            )

        # Add new items to clusters from temp where the cluster match UUID is new

        du.query_nonreturn(
            f"""
            insert into {clus}
            select
                uuid,
                cluster,
                id,
                source,
                n
            from
                {clusters_temp} ct
            where not exists (
                select
                    uuid,
                    cluster,
                    id,
                    source,
                    n
                from
                    {clus} c
                where
                    c.uuid = ct.uuid
            );
        """
        )

        # Tidy up

        du.query_nonreturn(
            f"""
            drop table if exists {clusters_temp};
            drop table if exists {probabilities_temp};
            drop table if exists {to_insert_temp};
        """
        )

        if add_unmatched_dims:
            # Add a new cluster for any rows that weren't matched in the dim tables
            for table in probabilities.get_sources():
                dataset = Dataset(
                    star_id=table,
                    star=self.star,
                )
                du.query_nonreturn(
                    f"""
                    insert into {clus}
                    select
                        gen_random_uuid() as uuid,
                        gen_random_uuid() as cluster,
                        id,
                        {table} as source,
                        {n} as n
                    from
                        {dataset.dim_schema_table} dim
                    where not exists (
                        select
                            id,
                            source
                        from
                            {clus} c
                        where
                            c.id = dim.id
                            and c.source = dim.source
                    );
                """
                )

    def get_data(
        self,
        select: dict,
        cluster_uuid_to_id: bool = False,
        dim_only: bool = True,
        n: int = None,
        sample: float = None,
    ):
        """
        Wrangle clusters and associated dimension fields into an output
        appropriate for the linker object to join a new dimension table onto.

        Returns a temporary dimension table that can use information from
        across the matched clusters so far to be the left "cluster" side of
        the next step n of the 🔌hybrid additive architecture.

        Arguments:
            select: a dict where the key is a Postgres-quoted fact table, with
            values a list of fields you want from that fact table. We use fact
            table as all of these are on Data Workspace, and this will hopefully
            make a really interpretable dict to pass. Supposes "as" aliasing to
            allow control of name clashing between datasets
            cluster_uuid_to_id: alias the cluster UUID to name "id". Used to
            make comparable field names in link jobs
            dim_only: force function to only return data from dimension tables.
            Used to build the left side of a join, where retrieving from the
            fact table would create duplication
            n: [optional] the step at which to retrive values
            sample: [optional] the sample percentage to take, allowing quicker
            development of upstream objects that need to run this lots

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
            fields_unaliased = {field.split(" as ")[0] for field in fields}

            if dim_only:
                cols = set(data.get_cols("dim"))

                if len(fields_unaliased - cols) > 0:
                    raise KeyError(
                        """
                        Requested field not in dimension table and dim_only
                        is True.
                    """
                    )

                for field in fields:
                    select_items.append(f"t{i}.{field}")

                join_clauses += f"""
                    left join {data.dim_schema_table} t{i} on
                        cl.id = t{i}.id
                        and cl.source = {data.id}
                """
            else:
                cols = set(data.get_cols("fact"))

                if len(set(fields_unaliased) - cols) > 0:
                    raise KeyError(
                        """
                        Requested field not in fact table.
                    """
                    )

                for field in fields:
                    select_items.append(f"t{i}.{field}")

                join_clauses += f"""
                    left join {data.fact_schema_table} t{i} on
                        cl.id = t{i}.id
                        and cl.source = {data.id}
                """

        id_alias = " as id" if cluster_uuid_to_id else ""
        if sample is not None:
            sample_clause = f"tablesample system ({sample})"
        else:
            sample_clause = ""

        sql = f"""
            select
                cl.cluster{id_alias},
                {', '.join(select_items)}
            from
                {self.schema_table} cl {sample_clause}
            {join_clauses}
            {n_clause};
        """

        result = du.query(sql)

        if len(result.index) == 0:
            raise ValueError("Nothing returned. Something went wrong")

        return result
