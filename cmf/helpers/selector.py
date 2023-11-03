from cmf.data import utils as du
from cmf.data import Table

from typing import List, Dict


def selector(table: str, fields: List[str]) -> Dict[Table, List[str]]:
    db_schema, db_table = du.get_schema_table_names(full_name=table, validate=True)
    selected_table = Table(db_schema=db_schema, db_table=db_table)

    if selected_table.exists:
        all_cols = set(selected_table.db_fields)
        selected_cols = set(fields)
        if not len(all_cols.intersection(selected_cols)) == len(selected_cols):
            raise ValueError(
                f"{selected_cols.difference(all_cols)} not found in "
                f"{selected_table.db_schema_table}"
            )
    else:
        raise ValueError(f"{selected_table.db_schema_table} not found")

    return {selected_table: fields}


def selectors(*selector: Dict[Table, List[str]]) -> Dict[Table, List[str]]:
    return {k: v for d in (selector) for k, v in d.items()}


if __name__ == "__main__":
    pass
