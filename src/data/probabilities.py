from src.data import utils as du
import uuid


class Probabilities(object):
    """
    A class to interact with the company matching framework's probabilities
    table. Enforces things are written in the right shape, and facilates easy
    retrieval of data in various shapes.

    Attributes:
        * schema: the probabilities table's schema name
        * table: the probabilities table's table name
        * schema_table: the probabilities table's full name
        * star: an object of class Star that wraps the star table

    Methods:
        * create(overwrite): Drops all data and recreates the probabilities
        table
        * read(): Returns the probabilities table
        * add_probabilities(lookup): Add new entries to the probabilities
        table
    """

    def __init__(self, schema: str, table: str, star: object):
        self.schema = schema
        self.table = table
        self.schema_table = f'"{self.schema}"."{self.table}"'
        self.star = star

    def create(self, overwrite: bool):
        """
        Creates a new probabilities table.

        Arguments:
            overwrite: Whether or not to overwrite an existing probabilities
            table
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
                link_type text not null,
                source int not null,
                cluster uuid not null,
                id text not null,
                probability float not null
            );
        """

        du.query_nonreturn(sql)

    def read(self):
        return du.dataset(self.schema_table)

    def get_sources(self) -> list:
        """
        Returns a list of the sources currently present in the probabilities table.

        Raises:
            KeyError: if table currently contains no sources

        Returns:
            A list of source ints, as appear in the star table
        """
        sources = du.query(f"select distinct source from {self.schema_table}")

        if len(sources.index) == 0:
            raise KeyError("Probabilities table currently contains no sources")

        return sources["source"].tolist()

    def add_probabilities(self, probabilities):
        """
        Takes an output from Linker.predict() and adds it to the probabilities
        table.

        Arguments:
            probabilities: A data frame produced by Linker.predict(). Should
            contain columns cluster, table, id, source and probability.

        Raises:
            ValueError:
                * If probabilities doesn't contain columns cluster, table, id
                source and probability
                * If probabilities doesn't contain values between 0 and 1

        Returns:
            The dataframe of probabilities that were added to the table.
        """

        in_cols = set(probabilities.columns.tolist())
        check_cols = {"cluster", "table", "id", "probability", "source"}
        if len(in_cols - check_cols) != 0:
            raise ValueError(
                """
                Linker.predict() has not produced outputs in an appropriate
                format for the probabilities table.
            """
            )
        max_prob = max(probabilities.probability)
        min_prob = min(probabilities.probability)
        if max_prob > 1 or min_prob < 0:
            raise ValueError(
                f"""
                Probability column should contain valid probabilities.
                Max: {max_prob}
                Min: {min_prob}
            """
            )

        probabilities["uuid"] = [uuid.uuid4() for _ in range(len(probabilities.index))]
        probabilities["link_type"] = "link"

        du.data_workspace_write(
            df=probabilities, schema=self.schema, table=self.table, if_exists="append"
        )

        return probabilities
