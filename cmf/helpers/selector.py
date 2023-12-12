from typing import Dict, List, Optional

from pandas import DataFrame

from cmf.data import Table


def selector(table: str, fields: List[str]) -> Dict[str, List[str]]:
    selected_table = Table.from_schema_table(full_name=table)

    if selected_table.exists:
        all_cols = set(selected_table.db_fields)
        selected_cols = set(fields)
        if not selected_cols <= all_cols:
            raise ValueError(
                f"{selected_cols.difference(all_cols)} not found in "
                f"{selected_table.db_schema_table}"
            )
    else:
        raise ValueError(f"{selected_table.db_schema_table} not found")

    return {selected_table.db_schema_table: fields}


def selectors(*selector: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {k: v for d in (selector) for k, v in d.items()}


def query(
    select: Dict[str, List[str]], raw: bool = False, sample: Optional[float] = None
) -> DataFrame:
    if len(select) == 1:
        table, fields = tuple(select.items())[0]
        selected_table = Table.from_schema_table(full_name=table)
        return selected_table.read(select=fields, sample=sample)
    else:
        # selectors
        raise NotImplementedError("Not built this yet")
