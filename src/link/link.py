from src.data import utils as du

# from src.models import utils as mu
from src.config import tables

from splink.duckdb.linker import DuckDBLinker


class LinkDatasets(object):
    def __init__(
        self,
        table_l: str,
        table_l_select: list,
        table_l_preproc: list,
        table_r: str,
        table_r_select: list,
        table_r_preproc: list,
        settings: dict,
        pipeline: list,
    ):
        self.settings = settings
        self.pipeline = pipeline

        self.table_l_alias = du.clean_table_name(table_l)
        self.table_l_select = ", ".join(table_l_select)
        self.table_l_dim = tables[table_l]["dim"]
        self.table_r_alias = du.clean_table_name(table_r)
        self.table_r_select = ", ".join(table_r_select)
        self.table_r_dim = tables[table_r]["dim"]

        self.table_l_raw = None
        self.table_r_raw = None

        self.table_l_proc_pipe = table_l_preproc
        self.table_r_proc_pipe = table_r_preproc

        self.table_l_proc = None
        self.table_r_proc = None

        self.linker = None

    def get_data(self):
        self.table_l_raw = du.query(
            f"""
                select
                    {self.table_l_select}
                from
                    {self.table_l_dim};
            """
        )
        self.table_r_raw = du.query(
            f"""
                select
                    {self.table_r_select}
                from
                    {self.table_r_dim};
            """
        )

    def preprocess_data(self):
        curr = self.table_l_raw
        for step in self.table_l_proc_pipe:
            curr = step(curr)
        self.table_l_proc = curr

        curr = self.table_r_raw
        for step in self.table_r_proc_pipe:
            curr = step(curr)
        self.table_r_proc = curr

    def create_linker(self):
        self.linker = DuckDBLinker(
            input_table_or_tables=[self.table_l_proc, self.table_r_proc],
            settings_dict=self.settings,
            input_table_aliases=[self.table_l_alias, self.table_r_alias],
        )

    def train_linker(self):
        for k in self.pipeline.keys():
            proc_func = getattr(self.linker, k)
            proc_func(**self.pipeline[k])
