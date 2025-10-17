import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from sqlglot.errors import ParseError

from matchbox.client.locations import RelationalDBLocation
from matchbox.common.dtos import (
    DataTypes,
    LocationType,
)
from matchbox.common.exceptions import MatchboxSourceExtractTransformError
from matchbox.common.factories.sources import (
    FeatureConfig,
    source_factory,
    source_from_tuple,
)


def test_relational_db_location_instantiation(sqlite_in_memory_warehouse: Engine):
    """Test that RelationalDBLocation can be instantiated with valid parameters."""
    location = RelationalDBLocation(name="dbname", client=sqlite_in_memory_warehouse)
    assert location.config.type == LocationType.RDBMS
    assert location.config.name == "dbname"


@pytest.mark.parametrize(
    ["sql", "is_valid"],
    [
        pytest.param("SELECT * FROM test_table", True, id="valid-select"),
        pytest.param(
            "SELECT id, name FROM test_table WHERE id > 1",
            True,
            id="valid-where",
        ),
        pytest.param("SLECT * FROM test_table", False, id="invalid-syntax"),
        pytest.param("", False, id="empty-string"),
        pytest.param("ALTER TABLE test_table", False, id="alter-sql"),
        pytest.param(
            "INSERT INTO users (name, age) VALUES ('John', '25')",
            False,
            id="insert-sql",
        ),
        pytest.param("DROP TABLE test_table", False, id="drop-sql"),
        pytest.param("SELECT * FROM users /* with a comment */", True, id="comment"),
        pytest.param(
            "WITH user_cte AS (SELECT * FROM users) SELECT * FROM user_cte",
            True,
            id="valid-with",
        ),
        pytest.param(
            (
                "WITH user_cte AS (SELECT * FROM users) "
                "INSERT INTO temp_users SELECT * FROM user_cte"
            ),
            False,
            id="invalid-with",
        ),
        pytest.param(
            "SELECT * FROM users; DROP TABLE users;", False, id="multiple-statements"
        ),
        pytest.param("SELECT * INTO new_table FROM users", False, id="select-into"),
        pytest.param(
            """
            WITH updated_rows AS (
                UPDATE employees
                SET salary = salary * 1.1
                WHERE department = 'Sales'
                RETURNING *
            )
            SELECT * FROM updated_rows;
            """,
            False,
            id="non-query-cte",
        ),
        pytest.param(
            """
            SELECT foo, bar FROM baz
            UNION
            SELECT foo, bar FROM qux;
            """,
            True,
            id="valid-union",
        ),
        # This test only works with postgres
        pytest.param(
            """
            SELECT 'ciao' ~ 'hello'
            """,
            True,
            id="valid-tilde",
        ),
    ],
)
def test_relational_db_extract_transform(
    sql: str, is_valid: bool, postgres_warehouse: Engine
):
    """Test SQL validation in validate_extract_transform."""
    location = RelationalDBLocation(name="dbname", client=postgres_warehouse)

    if is_valid:
        assert location.validate_extract_transform(sql)
    else:
        with pytest.raises((MatchboxSourceExtractTransformError, ParseError)):
            location.validate_extract_transform(sql)


def test_relational_db_infer_types(sqlite_warehouse: Engine):
    """Test that types are inferred correctly from the extract transform SQL."""
    source_testkit = source_from_tuple(
        data_tuple=(
            {"foo": "10", "bar": None},
            {"foo": "foo_val", "bar": None},
            {"foo": None, "bar": 10},
        ),
        data_keys=["a", "b", "c"],
        name="source",
        engine=sqlite_warehouse,
    ).write_to_location()
    location = RelationalDBLocation(name="dbname", client=sqlite_warehouse)

    query = f"""
        select key as renamed_key, foo, bar from
        (select key, foo, bar from {source_testkit.name});
    """

    inferred_types = location.infer_types(query)

    assert len(inferred_types) == 3
    assert inferred_types["renamed_key"] == DataTypes.STRING
    assert inferred_types["foo"] == DataTypes.STRING
    assert inferred_types["bar"] == DataTypes.INT64


def test_relational_db_execute(sqlite_warehouse: Engine):
    """Test executing a query and returning results using a real SQLite database."""
    features = [
        FeatureConfig(name="company", base_generator="company"),
        FeatureConfig(name="employees", base_generator="random_int"),
    ]

    source_testkit = source_factory(
        features=features, n_true_entities=10, engine=sqlite_warehouse
    ).write_to_location()
    location = RelationalDBLocation(name="dbname", client=sqlite_warehouse)

    sql = f"select key, upper(company) up_company, employees from {source_testkit.name}"

    batch_size = 2

    # Execute the query
    results = list(location.execute(sql, batch_size))

    # Right number of batches and total rows
    assert len(results[0]) == batch_size
    combined_df: pl.DataFrame = pl.concat(results)
    assert len(combined_df) == 10

    # Right fields, types and transformations
    assert set(combined_df.columns) == {"key", "up_company", "employees"}
    assert combined_df["employees"].dtype == pl.Int64
    sample_str = combined_df.select("up_company").row(0)[0]
    assert sample_str == sample_str.upper()

    # Try overriding schema
    overridden_results = list(
        location.execute(sql, batch_size, schema_overrides={"employees": pl.String})
    )
    assert overridden_results[0]["employees"].dtype == pl.String

    # Try query with filter
    keys_to_filter = source_testkit.data["key"][:2].to_pylist()
    filtered_results = pl.concat(
        location.execute(sql, batch_size, keys=("key", keys_to_filter))
    )
    assert len(filtered_results) == 2

    # Filtering by no keys has no effect
    unfiltered_results = pl.concat(location.execute(sql, batch_size, keys=("key", [])))
    assert_frame_equal(unfiltered_results, combined_df)


def test_relational_db_execute_invalid(sqlite_warehouse: Engine):
    """Test that invalid queries are handled correctly when executing."""
    location = RelationalDBLocation(name="dbname", client=sqlite_warehouse)

    # Invalid SQL query
    sql = "SELECT * FROM nonexistent_table"

    # Should raise an exception when executed
    with pytest.raises(OperationalError):
        list(location.execute(sql, batch_size=10))
