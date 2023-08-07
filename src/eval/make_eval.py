from src.config import tables
from src.data import utils as du


class MakeEval(object):
    def __init__(self, pair: set, eval_table: str):
        self.table_l = tables[pair[0]]["dim"]
        self.table_r = tables[pair[1]]["dim"]
        self.eval_table = eval_table

    def make_eval_table(self, overwrite: bool) -> None:
        if du.check_table_exists(self.eval_table) and not overwrite:
            raise ValueError(
                "Table exists. Set overwrite to True if you want to proceed."
            )

        if overwrite:
            sql = f"drop table if exists {self.eval_table};"
            du.query_nonreturn(sql)

        sql = f"""
            create table {self.eval_table} as (
                --
            );
        """

        du.query_nonreturn(sql)
