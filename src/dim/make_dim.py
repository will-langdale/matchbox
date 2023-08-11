from src.data import utils as du
from src.config import tables
import logging


class MakeDim(object):
    def __init__(self, unique_fields: list, fact_table: str, dim_table: str):
        self.unique_fields = ", ".join(unique_fields)
        self.fact_table = fact_table
        self.dim_table = dim_table

    def make_dim_table(self, overwrite: bool) -> None:
        if du.check_table_exists(self.dim_table) and not overwrite:
            raise ValueError(
                "Table exists. Set overwrite to True if you want to proceed."
            )

        if overwrite:
            sql = f"drop table if exists {self.dim_table};"
            du.query_nonreturn(sql)

        sql = f"""
            create table {self.dim_table} as (
                select distinct on ({self.unique_fields})
                    id,
                    {self.unique_fields}
                from
                    {self.fact_table}
                order by
                    {self.unique_fields}
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

    for table in tables:
        if tables[table]["fact"] != tables[table]["dim"]:
            dim_name = tables[table]["dim"]
            logger.info(f"Creating {dim_name}")

            dim_config = MakeDim(
                unique_fields=tables[table]["key_fields"],
                fact_table=tables[table]["fact"],
                dim_table=dim_name,
            )

            dim_config.make_dim_table(overwrite=True)

            logger.info(f"Written {dim_name}")

    logger.info("Finished")
