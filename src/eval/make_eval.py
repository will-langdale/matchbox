from src.config import tables, pairs
from src.data import utils as du


class MakeEval(object):
    def __init__(self, pair: set, eval_table: str):
        self.table_l = tables[pair[0]]["dim"]
        self.table_l_match = tables[pair[0]]["match_v1"]
        self.table_r = tables[pair[1]]["dim"]
        self.table_r_match = tables[pair[1]]["match_v1"]
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
                select distinct on (cluster)
                    cluster,
                    l_id,
                    r_id,
                    score
                from (
                    select
                        l_lkp.id as l_id,
                        case
                            when l.id is not null
                            then true
                            else false
                        end as l_hit,
                        r_lkp.id as r_id,
                        case
                            when r.id is not null
                            then true
                            else false
                        end as r_hit,
                        l_lkp.match_id as cluster,
                        coalesce(
                            (
                                char_length(replace(l_lkp.similarity, '0', ''))
                                +
                                char_length(replace(r_lkp.similarity, '0', ''))
                            ),
                            0
                        ) as score
                    from
                        {self.table_l_match} l_lkp
                    full join
                        {self.table_r_match} r_lkp on
                        l_lkp.match_id = r_lkp.match_id
                    left join
                         {self.table_l} l on
                        l.id::text = l_lkp.id::text
                    left join
                         {self.table_r} r on
                        r.id::text = r_lkp.id::text
                ) raw_matches
                where
                    l_hit = true
                    and r_hit = true
                order by
                    cluster desc,
                    score desc,
                    l_hit desc,
                    r_hit desc
            );
        """

        du.query_nonreturn(sql)


if __name__ == "__main__":
    for pair in pairs.keys():
        evaluater = MakeEval(pair=pair, eval_table=pairs[pair]["eval"])
        evaluater.make_eval_table(overwrite=True)
