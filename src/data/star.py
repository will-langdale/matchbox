from src.data import utils as du
from src.config import link_pipeline

import logging
from dotenv import load_dotenv, find_dotenv
import os
from hashlib import sha256
import pandas as pd


def make_star_table(
    link_pipeline: dict, star_schema: str, star_table: str, overwrite: bool
):
    """
    Creates a STAR lookup table from a link_pipeline settings object.

    Arguments:
        link_pipeline: The source link_pipeline settings object
        star_schema: The target STAR lookup schema
        star_table: The target STAR lookup table
        overwrite: Whether to overwrite the target if it exists

    Raises:
        ValueError if the table exists and overwrite is False
    """

    if_exists = "replace" if overwrite else "fail"
    schema_and_table = f'"{os.getenv("SCHEMA")}"."{os.getenv("STAR_TABLE")}"'

    if du.check_table_exists(schema_and_table) and not overwrite:
        raise ValueError("Table exists. Set overwrite to True if you want to proceed.")

    out = {"uuid": [], "fact": [], "dim": []}
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
        # I therefore manipulate the hex to 0-65535 to fit in a 16-bit signed
        # int field
        hash_int = int(hash_hex, 16) % 65535

        out["uuid"].append(hash_int)
        out["fact"].append(fact)
        out["dim"].append(dim)

    star = pd.DataFrame.from_dict(out)

    du.data_workspace_write(
        df=star, schema=star_schema, table=star_table, if_exists=if_exists
    )


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

    make_star_table(
        link_pipeline=link_pipeline,
        star_schema=os.getenv("SCHEMA"),
        star_table=os.getenv("STAR_TABLE"),
        overwrite=True,
    )

    logger.info("Written STAR table")
