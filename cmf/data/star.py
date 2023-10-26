from cmf.data import utils as du
from cmf.config import link_pipeline

import logging
from dotenv import load_dotenv, find_dotenv
import os
from hashlib import sha256
import pandas as pd


class Star(object):
    """
    A class to interact with the company matching framework's STAR table.
    Enforces things are written in the right shape, and facilates easy
    retrieval of data in various shapes.

    Attributes:
        * schema: the cluster table's schema name
        * table: the cluster table's table name
        * schema_table: the cluster table's full name

    Methods:
        * create(link_pipeline, overwrite): Drops all data and recreates the
        star table from the supplied link_pipeline
        * read(): Returns the STAR table
        * get(star_id, fact, dim, response): For strings of ID, fact, dimension,
        some, both or all, returns the requested datatype if a single match can
        be made
    """

    def __init__(self, schema: str, table: str):
        self.schema = schema
        self.table = table
        self.schema_table = f'"{self.schema}"."{self.table}"'

    def create(self, link_pipeline: dict, overwrite: bool):
        """
        Creates a STAR lookup table from a link_pipeline settings object.

        Arguments:
            link_pipeline: The source link_pipeline settings object
            overwrite: Whether to overwrite the target if it exists

        Raises:
            ValueError
                * If the table exists and overwrite is False
                * If the hash function fails to produce unique keys
        """

        if_exists = "replace" if overwrite else "fail"

        if du.check_table_exists(self.schema_table) and not overwrite:
            raise ValueError(
                """
                Table exists. Set overwrite to True if you want to proceed.
            """
            )

        out = {"id": [], "fact": [], "dim": []}
        for table in link_pipeline:
            fact = link_pipeline[table]["fact"]
            dim = link_pipeline[table]["dim"]
            hash_hex = sha256(
                f"{fact}{dim}".encode(encoding="UTF-8", errors="strict")
            ).hexdigest()
            # We need a hash that:
            # * Is an integer
            # * Is stable
            # * Is unique for the amount of dims we'll ever see
            # I therefore manipulate the hex to 0-65535 to fit in a 16-bit unsigned
            # int field
            hash_int = int(hash_hex, 16) % 65536

            out["id"].append(hash_int)
            out["fact"].append(fact)
            out["dim"].append(dim)

        star = pd.DataFrame.from_dict(out)

        if len(star) != len(star.id.unique()):
            raise ValueError(
                """
                Hash function has failed to produce unique keys. Change it.
            """
            )

        du.data_workspace_write(
            df=star, schema=self.schema, table=self.table, if_exists=if_exists
        )

    def read(self):
        return du.dataset(self.schema_table)

    def get(
        self, star_id: int = None, fact: str = None, dim: str = None, response="id"
    ):
        """
        For strings of ID, fact, dimension, some, both or all, returns the
        requested datatype if a single match can be made.

        Arguments:
            star_id: An integer in the ID column
            fact: A string found in the fact column
            dim: A string found in the dim column
            response: Returned object. One of 'id', 'fact' or 'dim'

        Raises:
            ValueError:
                * If response argument not one of 'id', 'fact' or 'dim'
                * If one of star_id, fact or dim isn't given
                * If more than one value is returned

        Returns:
            Integer of ID, or string of dimension or fact table
        """
        if response not in ["id", "fact", "dim"]:
            raise ValueError(
                """
                Response argument not one of 'id', 'fact' or 'dim'
            """
            )
        if all(i is None for i in [star_id, fact, dim]):
            raise ValueError(
                """
                Must supply at least one argument to star_id, fact or dim
            """
            )

        where_id = f"id = {star_id}" if star_id is not None else ""
        if fact is not None:
            fact_clean = fact.lower().replace('"', "")
            where_fact = f"replace(lower(fact), '\"', '') like '%{fact_clean}%'"
        else:
            where_fact = ""
        if dim is not None:
            dim_clean = dim.lower().replace('"', "")
            where_dim = f"replace(lower(dim), '\"', '') like '%{dim_clean}%'"
        else:
            where_dim = ""

        where_clause = " and ".join(
            filter(lambda i: (i != ""), [where_id, where_fact, where_dim])
        )

        sql = f"""
            select
                {response}
            from
                {self.schema_table}
            where
                {where_clause}
        """

        filtered_star = du.query(sql)

        if len(filtered_star) > 1:
            raise ValueError(
                """
                More than one value returned. Refine your request.
            """
            )

        results = filtered_star[response].values.tolist()

        if len(results) != 1:
            raise ValueError(
                """
               Nothing returned. Check the referenced table exists in STAR.
            """
            )

        return results[0]


if __name__ == "__main__":
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    logging.basicConfig(
        level=logging.INFO,
        format=du.LOG_FMT,
    )
    logger = logging.getLogger(__name__)
    logger.info(
        f'Creating STAR table "{os.getenv("SCHEMA")}"."{os.getenv("STAR_TABLE")}"'
    )

    star = Star(schema=os.getenv("SCHEMA"), table=os.getenv("STAR_TABLE"))

    star.create(
        link_pipeline=link_pipeline,
        overwrite=True,
    )

    logger.info("Written STAR table")
