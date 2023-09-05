from src.data import utils as du
from src.config import link_pipeline
import logging


def make_dim_table(
    fact_table: str, unique_fields: list, dim_table: str, overwrite: bool
):
    """
    Takes a fact table and a list of its unique fields, then writes the naive
    deduplicated version to the specified dimension table.

    Reuses the business identifier from the fact table.

    Arguments:
        fact_table: The source fact table
        unique_fields: A list of fields to derive unique rows from
        dim_table: The target dimension table
        overwrite: Whether to overwrite the target if it exists

    Returns:
        Nothing
    """

    unique_fields = ", ".join(unique_fields)

    if du.check_table_exists(dim_table) and not overwrite:
        raise ValueError("Table exists. Set overwrite to True if you want to proceed.")

    if overwrite:
        sql = f"drop table if exists {dim_table};"
        du.query_nonreturn(sql)

    sql = f"""
        create table {dim_table} as (
            select distinct on ({unique_fields})
                id,
                {unique_fields}
            from
                {fact_table}
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

    for table in link_pipeline:
        if link_pipeline[table]["fact"] != link_pipeline[table]["dim"]:
            dim_name = link_pipeline[table]["dim"]
            logger.info(f"Creating {dim_name}")

            dim_config = make_dim_table(
                unique_fields=link_pipeline[table]["key_fields"],
                fact_table=link_pipeline[table]["fact"],
                dim_table=dim_name,
                overwrite=True,
            )

            logger.info(f"Written {dim_name}")

    logger.info("Finished")
