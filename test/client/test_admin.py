from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import dedent

import pytest
from tomli_w import dumps

from matchbox.client.admin import load_datasets_from_config
from matchbox.common.sources import Source, SourceWarehouse


def warehouse_toml(warehouse: SourceWarehouse) -> str:
    return dedent(f"""
        [warehouses.{warehouse.alias}]
        db_type = "{warehouse.db_type}"
        user = "{warehouse.user}"
        password = "{warehouse.password.get_secret_value()}"
        host = "{warehouse.host}"
        port = {warehouse.port}
        database = "{warehouse.database}"
    """).strip()


def source_toml(source: Source, index: list[dict[str, str]]) -> str:
    index_str = dumps({"index": index}).replace("\n", "\n        ")
    alias = source.alias if "." not in source.alias else f'"{source.alias}"'
    return dedent(f"""
        [datasets.{alias}]
        database = "test_warehouse"
        db_schema = "{source.db_schema}"
        db_table = "{source.db_table}"
        db_pk = "{source.db_pk}"
        {index_str}
    """)


@pytest.mark.parametrize(
    "index",
    (
        [
            {"literal": "company_name"},
            {"literal": "crn"},
        ],
        [{"literal": "company_name", "alias": "name"}],
    ),
    ids=["vanilla", "alias"],
)
def test_load_datasets_from_config(
    index: list[dict[str, str]],
    warehouse: SourceWarehouse,
    warehouse_data: list[Source],
):
    """Tests loading datasets from a TOML file."""
    # Construct TOML from CRN data
    # Columns: "id", "company_name", "crn"
    crn = warehouse_data[0]
    raw_toml = dedent(f"""
        {warehouse_toml(warehouse)}
        {source_toml(crn, index)}      
    """).strip()

    with NamedTemporaryFile(suffix=".toml", delete=False) as temp_file:
        temp_file.write(raw_toml.encode())
        temp_file.flush()
        temp_file_path = Path(temp_file.name)

    # Ingest
    config = load_datasets_from_config(temp_file_path)

    # Helper variables
    source = config.get(crn.alias)
    named = [idx["literal"] for idx in index]
    col_names = [col.literal.name for col in source.db_columns]

    # Test 1: Core attributes match
    assert source.database == warehouse
    assert source.alias == crn.alias
    assert source.db_schema == crn.db_schema
    assert source.db_table == crn.db_table
    assert source.db_pk == crn.db_pk

    # Test 2: All non-pk columns present
    assert set(col_names) == {"company_name", "crn", "id"} - {source.db_pk}

    # Test 3: Aliases match
    for idx in index:
        col = next(c for c in source.db_columns if c.literal.name == idx["literal"])
        assert col.alias.name == idx.get("alias", idx["literal"])

    # Test 4: Column ordering
    for i, name in enumerate(named):
        assert col_names[i] == name

    # Test 5: column equalities
    assert source.db_columns[0] != source.db_columns[1]
    assert source.db_columns[0] == source.db_columns[0]
    assert source.db_columns[1].literal.hash == source.db_columns[1]
    assert source.db_columns[1].alias.hash == source.db_columns[1]
    assert source.db_columns[0].literal.hash != source.db_columns[1]
    assert source.db_columns[0].alias.hash != source.db_columns[1]
