import copy
from typing import Any, Callable

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal
from pydantic import AnyUrl
from sqlalchemy import (
    Engine,
    Table,
    create_engine,
)
from sqlalchemy.exc import OperationalError
from sqlglot.errors import ParseError

from matchbox.client.helpers.selector import Match
from matchbox.common.db import fullname_to_prefix
from matchbox.common.exceptions import (
    MatchboxSourceColumnError,
    MatchboxSourceExtractTransformError,
)
from matchbox.common.factories.sources import source_factory, source_from_tuple
from matchbox.common.sources import (
    Location,
    RelationalDBLocation,
    SourceAddress,
    SourceColumn,
    SourceConfig,
)


def test_location_factory():
    """Test we can construct appropriate Location classes from raw data."""
    location = Location.create(
        {
            "type": "rdbms",
            "uri": "postgresql://host:1234/db2",
        }
    )
    assert isinstance(location, RelationalDBLocation)


def test_location_empty_credentials_error():
    """Test that operations requiring credentials fail when credentials are not set."""
    location = RelationalDBLocation(uri="postgresql://host:1234/db2")

    # Attempting to connect without credentials should raise an error
    with pytest.raises(AttributeError):
        location.connect()

    # Invalid location type
    with pytest.raises(ValueError, match="Unknown location type"):
        Location.create({"type": "unknown", "uri": "http://example.com"})

    # Missing required fields
    with pytest.raises(ValueError):
        Location.create({"type": "rdbms"})


def test_location_serialisation():
    """Test serialisation and deserialisation of Location objects."""
    original = RelationalDBLocation(uri="postgresql://host:1234/db2")

    # Convert to dict and back - credentials should be excluded
    location_dict = original.model_dump()
    assert "credentials" not in location_dict

    # Deserialize back to a Location
    reconstructed = RelationalDBLocation.model_validate(location_dict)
    assert reconstructed.uri == original.uri
    assert reconstructed.type == original.type
    assert reconstructed.credentials is None


def test_relational_db_location_instantiation():
    """Test that RelationalDBLocation can be instantiated with valid parameters."""
    location = RelationalDBLocation(uri="sqlite:///test.db")
    assert location.type == "rdbms"
    assert str(location.uri) == "sqlite:///test.db"
    assert location.credentials is None


@pytest.mark.parametrize(
    ["uri_str", "should_pass"],
    [
        pytest.param("sqlite:///test.db", True, id="valid-sqlite"),
        pytest.param("postgresql://localhost:5432/testdb", True, id="valid-postgres"),
        pytest.param(
            "postgresql://user:pass@localhost:5432/testdb",
            False,
            id="invalid-credentials-in-uri",
        ),
        pytest.param(
            "postgresql+psycopg://localhost:5432/testdb",
            False,
            id="driver-in-uri",
        ),
        pytest.param("sqlite:///test.db?mode=ro", False, id="invalid-query-params"),
        pytest.param("sqlite:///test.db#fragment", False, id="invalid-fragment"),
    ],
)
def test_relational_db_location_uri_validation(uri_str: str, should_pass: bool):
    """Test URI validation in RelationalDBLocation."""
    if should_pass:
        location = RelationalDBLocation(uri=uri_str)
        assert str(location.uri) == uri_str
    else:
        with pytest.raises(ValueError):
            RelationalDBLocation(uri=AnyUrl(uri_str))


def test_relational_db_add_credentials(sqlite_warehouse: Engine):
    """Test the public interface for adding credentials to a RelationalDBLocation.

    This test verifies:
        1. Credentials are properly added and validated for matching engines
        2. Validation fails for non-matching engines
        3. Connect succeeds only with valid credentials
    """
    # Create engines and URIs
    test_engines = {
        # PostgreSQL engines with different connection parameters
        "pg": create_engine("postgresql://user:pass@host:5432/db"),  # trufflehog:ignore
        "pg_diff_host": create_engine(
            "postgresql://user:pass@otherhost:5432/db"  # trufflehog:ignore
        ),
        "pg_diff_port": create_engine(
            "postgresql://user:pass@host:5433/db"  # trufflehog:ignore
        ),
        "pg_diff_db": create_engine(
            "postgresql://user:pass@host:5432/otherdb"  # trufflehog:ignore
        ),
        # These should match the same URI as "pg"
        # (different user/pass/dialect don't affect matching)
        "pg_diff_user": create_engine(
            "postgresql://user2:pass@host:5432/db"  # trufflehog:ignore
        ),
        "pg_diff_pass": create_engine(
            "postgresql://user:pass2@host:5432/db"  # trufflehog:ignore
        ),
        "pg_diff_dialect": create_engine(
            "postgresql+psycopg://user:pass@host:5432/db"  # trufflehog:ignore
        ),
    }

    uris = {
        "pg": "postgresql://host:5432/db",
        "pg_diff_host": "postgresql://otherhost:5432/db",
        "pg_diff_port": "postgresql://host:5433/db",
        "pg_diff_db": "postgresql://host:5432/otherdb",
        "sqlite": str(sqlite_warehouse.url),
    }

    # Test case 1: Adding matching credentials succeeds
    location1 = RelationalDBLocation(uri=uris["sqlite"])
    location1.add_credentials(sqlite_warehouse)
    assert location1.credentials == sqlite_warehouse
    assert location1.connect() is True

    # Test case 2: Adding mismatched credentials fails
    location2 = RelationalDBLocation(uri=uris["pg"])
    with pytest.raises(ValueError, match="does not match"):
        location2.add_credentials(sqlite_warehouse)

    # Test case 3: Check which parameters affect URI matching
    # Group by expected URI matching behavior
    should_match_pg = {"pg", "pg_diff_user", "pg_diff_pass", "pg_diff_dialect"}
    should_not_match_pg = {"pg_diff_host", "pg_diff_port", "pg_diff_db"}

    # Verify that add_credentials works for matching engines
    base_pg_location = RelationalDBLocation(uri=uris["pg"])
    base_pg_location.add_credentials(test_engines["pg"])
    assert base_pg_location.credentials == test_engines["pg"]

    # All engines in should_match_pg should work with this location
    for engine_key in should_match_pg:
        test_location = RelationalDBLocation(uri=uris["pg"])
        test_location.add_credentials(test_engines[engine_key])
        assert test_location.credentials == test_engines[engine_key]

    # All engines in should_not_match_pg should fail with this location
    for engine_key in should_not_match_pg:
        test_location = RelationalDBLocation(uri=uris["pg"])
        with pytest.raises(ValueError, match="does not match"):
            test_location.add_credentials(test_engines[engine_key])


@pytest.mark.parametrize(
    ["sql", "is_valid"],
    [
        pytest.param("SELECT * FROM test_table", True, id="valid-select"),
        pytest.param(
            "SELECT id, name FROM test_table WHERE id > 1", True, id="valid-where"
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
    ],
)
def test_relational_db_extract_transform(sql: str, is_valid: bool):
    """Test SQL validation in validate_extract_transform."""
    location = RelationalDBLocation(uri="postgresql://host:1234/db2")

    if is_valid:
        assert location.validate_extract_transform(sql)
    else:
        with pytest.raises((MatchboxSourceExtractTransformError, ParseError)):
            location.validate_extract_transform(sql)


def test_relational_db_execute(sqlite_warehouse: Engine):
    """Test executing a query and returning results using a real SQLite database."""
    source_testkit = source_factory(engine=sqlite_warehouse)
    source_testkit.to_warehouse(engine=sqlite_warehouse)
    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    sql = f"SELECT * FROM {source_testkit.source_config.address.full_name}"
    batch_size = 2

    # Execute the query
    results = list(location.execute(sql, batch_size))

    # Check that we got expected results
    assert len(results) > 0

    # Combine all batches to check total row count
    combined_df = pl.concat(results)
    assert len(combined_df) == 10


def test_relational_db_execute_invalid(sqlite_warehouse: Engine):
    """Test that invalid queries are handled correctly when executing."""
    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    # Invalid SQL query
    sql = "SELECT * FROM nonexistent_table"

    # Should raise an exception when executed
    with pytest.raises(OperationalError):
        list(location.execute(sql, batch_size=10))


def test_relational_db_from_engine(sqlite_warehouse: Engine):
    """Test creating a RelationalDBLocation from an engine."""
    # Create a location from the engine
    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    # Check that it was created correctly
    assert location.credentials == sqlite_warehouse
    assert location.type == "rdbms"

    # Verify it works by connecting
    assert location.connect() is True


def test_relational_db_retrieval_and_transformation(sqlite_warehouse: Engine):
    """Test a more complete workflow with data retrieval and transformation."""
    source_testkit = source_factory(engine=sqlite_warehouse)
    source_testkit.to_warehouse(engine=sqlite_warehouse)
    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    # Execute a query with transformation
    sql = f"""
    SELECT 
        company_name as name,
        UPPER(company_name) as company_name,
        crn as crn
    FROM {source_testkit.source_config.address.full_name};
    """

    results = list(location.execute(sql, batch_size=1))
    assert len(results) == 10  # 10 batches of 1 row

    df = pl.concat(results)

    # Verify the result structure
    assert set(df.columns) == {"name", "company_name", "crn"}

    # Verify the calculated columns
    sample_str: str = df.select("company_name").row(0)[0]
    assert sample_str == sample_str.upper()


def test_source_address_compose():
    """Correct addresses are generated from engines and table names."""
    pg = create_engine("postgresql://user:fakepass@host:1234/db")  # trufflehog:ignore
    pg_host = create_engine(
        "postgresql://user:fakepass@host2:1234/db"  # trufflehog:ignore
    )
    pg_port = create_engine(
        "postgresql://user:fakepass@host:4321/db"  # trufflehog:ignore
    )
    pg_db = create_engine(
        "postgresql://user:fakepass@host:1234/db2"  # trufflehog:ignore
    )
    pg_user = create_engine(
        "postgresql://user2:fakepass@host:1234/db"  # trufflehog:ignore
    )
    pg_password = create_engine(
        "postgresql://user:fakepass2@host:1234/db"  # trufflehog:ignore
    )
    pg_dialect = create_engine(
        "postgresql+psycopg://user:fakepass@host:1234/db"  # trufflehog:ignore
    )

    sqlite = create_engine("sqlite:///foo.db")
    sqlite_name = create_engine("sqlite:///bar.db")

    different_wh_hashes = set(
        [
            SourceAddress.compose(pg, "tablename").warehouse_hash,
            SourceAddress.compose(pg_host, "tablename").warehouse_hash,
            SourceAddress.compose(pg_port, "tablename").warehouse_hash,
            SourceAddress.compose(pg_db, "tablename").warehouse_hash,
            SourceAddress.compose(sqlite, "tablename").warehouse_hash,
            SourceAddress.compose(sqlite_name, "tablename").warehouse_hash,
        ]
    )
    different_wh_hashes_str = set([str(sa) for sa in different_wh_hashes])

    assert len(different_wh_hashes) == 6
    assert len(different_wh_hashes_str) == 6

    same_wh_hashes = set(
        [
            SourceAddress.compose(pg, "tablename").warehouse_hash,
            SourceAddress.compose(pg_user, "tablename").warehouse_hash,
            SourceAddress.compose(pg_password, "tablename").warehouse_hash,
            SourceAddress.compose(pg_dialect, "tablename").warehouse_hash,
        ]
    )
    same_wh_hashes_str = set([str(sa) for sa in same_wh_hashes])

    assert len(same_wh_hashes) == 1
    assert len(same_wh_hashes_str) == 1

    same_table_name = set(
        [
            SourceAddress.compose(pg, "tablename").full_name,
            SourceAddress.compose(sqlite, "tablename").full_name,
        ]
    )
    same_table_name_str = set([str(sa) for sa in same_table_name])

    assert len(same_table_name) == 1
    assert len(same_table_name_str) == 1


def test_source_address_format_columns():
    """Column names can get a standard prefix from a table name."""
    address1 = SourceAddress(full_name="foo", warehouse_hash=b"bar")
    address2 = SourceAddress(full_name="foo.bar", warehouse_hash=b"bar")

    assert address1.format_column("col") == "foo_col"
    assert address2.format_column("col") == "foo_bar_col"


def test_source_set_engine(sqlite_warehouse: Engine):
    """Engine can be set on SourceConfig."""
    source_testkit = source_factory(engine=sqlite_warehouse)

    # We can set engine with correct column specification
    source = source_testkit.source_config.set_engine(sqlite_warehouse)
    assert isinstance(source, SourceConfig)

    # Error is raised with wrong engine
    with pytest.raises(ValueError, match="engine does not match"):
        wrong_engine = create_engine("sqlite:///:memory:")
        source.set_engine(wrong_engine)


def test_source_check_columns(sqlite_warehouse: Engine):
    """SourceConfig columns are checked against the warehouse."""
    source_testkit = source_factory(
        features=[{"name": "b", "base_generator": "random_int", "sql_type": "BIGINT"}],
        engine=sqlite_warehouse,
    )
    source_testkit.to_warehouse(engine=sqlite_warehouse)

    # We can set engine with correct column specification
    source = source_testkit.source_config.set_engine(sqlite_warehouse)
    assert isinstance(source, SourceConfig)

    # Error is raised with custom columns
    with pytest.raises(MatchboxSourceColumnError, match="Columns {'c'} not in"):
        source.check_columns(columns=["c"])

    # Error is raised with missing primary key
    new_source = source_testkit.source_config.model_copy(
        update={"db_pk": "typo"}
    ).set_engine(sqlite_warehouse)
    with pytest.raises(
        MatchboxSourceColumnError, match="Primary key typo not available"
    ):
        new_source.check_columns()

    # Error is raised with missing column
    new_source = source_testkit.source_config.model_copy(
        update={"columns": (SourceColumn(name="c", type="TEXT"),)}
    ).set_engine(sqlite_warehouse)
    with pytest.raises(MatchboxSourceColumnError, match="Column c not available in"):
        new_source.check_columns()

    # Error is raised with wrong type
    new_source = source_testkit.source_config.model_copy(
        update={"columns": (SourceColumn(name="b", type="TEXT"),)}
    ).set_engine(sqlite_warehouse)
    with pytest.raises(MatchboxSourceColumnError, match="Type BIGINT != TEXT for b"):
        new_source.check_columns()


def test_source_hash_equality(sqlite_warehouse: Engine):
    """__eq__ and __hash__ behave as expected for a SourceConfig."""
    # This won't set the engine just yet
    source_testkit = source_factory(engine=sqlite_warehouse)
    source = source_testkit.source_config
    source_eq = source.model_copy(deep=True)

    source_testkit.to_warehouse(engine=sqlite_warehouse)
    source.set_engine(sqlite_warehouse)

    assert source.engine != source_eq.engine
    assert source == source_eq
    assert hash(source) == hash(source_eq)


def test_source_default_columns(sqlite_warehouse: Engine):
    """Default columns from the warehouse can be assigned to a SourceConfig."""
    source_testkit = source_factory(
        features=[
            {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
            {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
        ],
        engine=sqlite_warehouse,
    )

    source_testkit.to_warehouse(engine=sqlite_warehouse)

    expected_columns = (
        SourceColumn(name="a", type="BIGINT"),
        SourceColumn(name="b", type="TEXT"),
    )

    source = source_testkit.source_config.set_engine(sqlite_warehouse).default_columns()

    assert source.columns == expected_columns
    # We create a new source, but attributes and engine match
    assert source is not source_testkit.source_config
    assert source == source_testkit.source_config
    assert source.engine == sqlite_warehouse


def test_source_to_table(sqlite_warehouse: Engine):
    """Convert SourceConfig to SQLAlchemy Table."""
    source_testkit = source_factory(engine=sqlite_warehouse)
    source_testkit.to_warehouse(engine=sqlite_warehouse)

    source = source_testkit.source_config.set_engine(sqlite_warehouse)

    assert isinstance(source.to_table(), Table)


@pytest.mark.parametrize(
    ("converter", "to_pandas_fn"),
    [
        pytest.param(
            lambda src, **kwargs: src.to_pandas(**kwargs),
            lambda df: df,
            id="pandas",
        ),
        pytest.param(
            lambda src, **kwargs: src.to_arrow(**kwargs),
            lambda arrow: arrow.to_pandas(),
            id="arrow",
        ),
        pytest.param(
            lambda src, **kwargs: src.to_polars(**kwargs),
            lambda polars: polars.to_pandas(),
            id="polars",
        ),
    ],
)
def test_source_conversion_methods(
    sqlite_warehouse: Engine,
    converter: Callable[[Any], Any],
    to_pandas_fn: Callable[[Any], Any],
):
    """Check equivalence of SourceConfig to Arrow, Pandas or Polars, with options."""
    source_testkit = source_factory(
        features=[
            {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
            {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
        ],
        engine=sqlite_warehouse,
        n_true_entities=2,
    )
    source_testkit.to_warehouse(engine=sqlite_warehouse)
    source = source_testkit.source_config.set_engine(sqlite_warehouse).default_columns()
    prefix = fullname_to_prefix(source_testkit.source_config.address.full_name)
    expected_df_prefixed = (
        source_testkit.data.to_pandas().drop(columns=["id"]).add_prefix(prefix)
    )

    # Test basic conversion
    output = converter(source)
    result_df = to_pandas_fn(output)
    assert_frame_equal(
        expected_df_prefixed, result_df, check_like=True, check_dtype=False
    )

    # Test with limit parameter
    output_limited = converter(source, limit=1)
    result_df_limited = to_pandas_fn(output_limited)
    assert_frame_equal(
        expected_df_prefixed.iloc[:1],
        result_df_limited,
        check_like=True,
        check_dtype=False,
    )

    # Test with fields parameter
    output_fields = converter(source, fields=["a"])
    result_df_fields = to_pandas_fn(output_fields)
    assert_frame_equal(
        expected_df_prefixed[[f"{prefix}pk", f"{prefix}a"]],
        result_df_fields,
        check_like=True,
        check_dtype=False,
    )


def test_source_hash_data(sqlite_warehouse: Engine):
    """A SourceConfig can output hashed versions of its rows."""
    original = source_factory(
        full_name="original",
        features=[
            {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
            {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
        ],
        engine=sqlite_warehouse,
        n_true_entities=2,
        repetition=1,
    )

    reordered = copy.deepcopy(original)
    reordered.source_config = original.source_config.model_copy(
        update={
            "address": original.source_config.address.model_copy(
                update={"full_name": "reordered"}
            ),
            "columns": (
                original.source_config.columns[1],
                original.source_config.columns[0],
            ),
        }
    )

    renamed = copy.deepcopy(original)
    renamed.data = renamed.data.rename_columns({"a": "x"})
    renamed.source_config = original.source_config.model_copy(
        update={
            "address": original.source_config.address.model_copy(
                update={"full_name": "renamed"}
            ),
            "columns": (
                original.source_config.columns[0].model_copy(update={"name": "x"}),
                original.source_config.columns[1],
            ),
        }
    )

    original.to_warehouse(engine=sqlite_warehouse)
    reordered.to_warehouse(engine=sqlite_warehouse)
    renamed.to_warehouse(engine=sqlite_warehouse)

    original_source = original.source_config.set_engine(sqlite_warehouse)
    reordered_source = reordered.source_config.set_engine(sqlite_warehouse)
    renamed_source = renamed.source_config.set_engine(sqlite_warehouse)

    original_hash = original_source.hash_data(batch_size=3).to_pandas()
    reordered_hash = reordered_source.hash_data().to_pandas()
    renamed_hash = renamed_source.hash_data().to_pandas()

    # Hash have the right shape
    assert len(original_hash) == 2
    assert len(original_hash.source_pk.iloc[0]) == 2
    assert len(original_hash.source_pk.iloc[1]) == 2

    def sort_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by="hash").reset_index(drop=True)

    # Column order does not matter, column names do
    assert sort_df(original_hash).equals(sort_df(reordered_hash))
    assert not sort_df(original_hash).equals(sort_df(renamed_hash))


def test_source_hash_nulls(sqlite_warehouse: Engine):
    """A SourceConfig can output hashed versions of rows with nulls."""
    testkit = source_from_tuple(
        data_tuple=({"a": 1.0}, {"a": None}),
        data_pks=["a", "b"],
        full_name="null_test",
        engine=sqlite_warehouse,
    )
    source = testkit.source_config.set_engine(sqlite_warehouse)
    testkit.to_warehouse(engine=sqlite_warehouse)

    # Test hashing with nulls
    hashed_data = source.hash_data()

    # No nulls in the hash column
    assert pa.compute.count(hashed_data["hash"], mode="only_null").as_py() == 0

    # Test hashing with null PKs
    null_pk_testkit = source_from_tuple(
        data_tuple=({"a": 1}, {"a": 2}, {"a": 3}),
        data_pks=["a", None, None],
        full_name="null_pk_test",
        engine=sqlite_warehouse,
    )

    # Null PKs should error
    with pytest.raises(ValueError):
        source_with_null_pks = null_pk_testkit.source_config.set_engine(
            sqlite_warehouse
        )
        null_pk_testkit.to_warehouse(engine=sqlite_warehouse)
        source_with_null_pks.hash_data()


@pytest.mark.parametrize(
    ("method_name", "return_type"),
    [
        pytest.param("to_arrow", pa.Table, id="to_arrow"),
        pytest.param("to_pandas", pd.DataFrame, id="to_pandas"),
    ],
)
def test_source_data_batching(method_name, return_type, sqlite_warehouse: Engine):
    """Test SourceConfig data retrieval methods with batching parameters."""
    # Create a source with multiple rows of data
    source_testkit = source_factory(
        features=[
            {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
            {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
        ],
        engine=sqlite_warehouse,
        n_true_entities=9,
    )
    source_testkit.to_warehouse(engine=sqlite_warehouse)
    source = source_testkit.source_config.set_engine(sqlite_warehouse).default_columns()

    # Call the method with batching
    method = getattr(source, method_name)
    batch_iterator = method(return_batches=True, batch_size=3)
    batches = list(batch_iterator)

    # Verify we got the expected number of batches
    assert len(batches) == 3
    for batch in batches:
        assert isinstance(batch, return_type)
        assert len(batch) == 3


def test_match_validates():
    """Match objects are validated when they're instantiated."""
    Match(
        cluster=1,
        source=SourceAddress(full_name="test.source_config", warehouse_hash=b"bar"),
        source_id={"a"},
        target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
        target_id={"b"},
    )

    # Missing source_id with target_id
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source=SourceAddress(full_name="test.source_config", warehouse_hash=b"bar"),
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
            target_id={"b"},
        )

    # Missing cluster with target_id
    with pytest.raises(ValueError):
        Match(
            source=SourceAddress(full_name="test.source_config", warehouse_hash=b"bar"),
            source_id={"a"},
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
            target_id={"b"},
        )

    # Missing source_id with cluster
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source=SourceAddress(full_name="test.source_config", warehouse_hash=b"bar"),
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
        )
