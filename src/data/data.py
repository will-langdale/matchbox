from src.data import utils as du

# from src.config import link_pipeline

import logging
from dotenv import load_dotenv, find_dotenv
import os


class Data(object):
    """
    A class to interact with fact and dimension tables in the company
    matching framework.

    Attributes:
        * id: the key of the fact/dim row in the STAR table
        * dim_schema: the schema of the data's dimension table
        * dim_table: the name of the data's dimension table
        * dim_schema_table: the data's dimention table full name
        * fact_schema: the schema of the data's dimension table
        * fact_table: the name of the data's dimension table
        * fact_schema_table: the data's dimention table full name

    Methods:
        * create(dim=None, overwrite): Drops all data and recreates the
        dimension table using the unique fields specified
        * read(): Returns the cluster table
        * add_clusters(probabilities): Using a probabilities table, adds new
        entries to the cluster table
        * get_data(fields): returns the cluster table pivoted wide,
        with the requested fields attached
    """

    def __init__(self, star_id: int, star: object):
        self.id = star_id
        self.dim_schema = None
        self.dim_table = None
        self.dim_schema_table = None
        self.fact_schema = None
        self.fact_table = None
        self.fact_schema_table = None

    def create_dim(self, unique_fields: list, overwrite: bool):
        """
        Takes a fact table and a list of its unique fields, then writes the naive
        deduplicated version to the specified dimension table.

        Reuses the business identifier from the fact table.

        Arguments:
            unique_fields: A list of fields to derive unique rows from
            overwrite: Whether to overwrite the target if it exists

        Raises:
            IOError: Refuses to make the dimension table unless the schema matches
            the framework's, indicating the framework controls the dimension.
            ValueError: If the table exists and overwrite isn't set to True

        Returns:
            Nothing
        """

        dotenv_path = find_dotenv()
        load_dotenv(dotenv_path)

        if os.getenv("SCHEMA") != self.dim_schema:
            raise IOError(
                f"""
                Dimension schema is not {os.getenv("SCHEMA")}.
                This table is not controlled by the framework"
            """
            )

        unique_fields = ", ".join(unique_fields)

        if du.check_table_exists(self.dim_schema_table) and not overwrite:
            raise ValueError(
                "Table exists. Set overwrite to True if you want to proceed."
            )

        if overwrite:
            sql = f"drop table if exists {self.dim_schema_table};"
            du.query_nonreturn(sql)

        sql = f"""
            create table {self.dim_schema_table} as (
                select distinct on ({unique_fields})
                    id,
                    {unique_fields}
                from
                    {self.fact_schema_table}
                order by
                    {unique_fields}
            );
        """

        du.query_nonreturn(sql)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=du.LOG_FMT,
    )
    logger = logging.getLogger(__name__)
    logger.info("Creating dim tables")

    #     for table in link_pipeline:
    #         if link_pipeline[table]["fact"] != link_pipeline[table]["dim"]:
    #             dim_name = link_pipeline[table]["dim"]
    #             logger.info(f"Creating {dim_name}")

    #             dim_config = make_dim_table(
    #                 unique_fields=link_pipeline[table]["key_fields"],
    #                 fact_table=link_pipeline[table]["fact"],
    #                 dim_table=dim_name,
    #                 overwrite=True,
    #             )

    #             logger.info(f"Written {dim_name}")

    logger.info("Finished")
