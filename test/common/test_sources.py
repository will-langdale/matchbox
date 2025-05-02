import copy
from typing import Iterator

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from pydantic import AnyUrl
from sqlalchemy import (
    Engine,
    create_engine,
)
from sqlalchemy.exc import OperationalError
from sqlglot.errors import ParseError

from matchbox.client.helpers.selector import Match
from matchbox.common.dtos import DataTypes
from matchbox.common.exceptions import (
    MatchboxSourceCredentialsError,
    MatchboxSourceExtractTransformError,
)
from matchbox.common.factories.sources import source_factory
from matchbox.common.hash import HASH_FUNC
from matchbox.common.sources import (
    Location,
    RelationalDBLocation,
    Source,
    SourceAddress,
    SourceField,
)

# Locations


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
    with pytest.raises(MatchboxSourceCredentialsError):
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

    sql = f"SELECT * FROM {source_testkit.source.resolution_name}"
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
    FROM {source_testkit.source.resolution_name};
    """

    results = list(location.execute(sql, batch_size=1))
    assert len(results) == 10  # 10 batches of 1 row

    df = pl.concat(results)

    # Verify the result structure
    assert set(df.columns) == {"name", "company_name", "crn"}

    # Verify the calculated columns
    sample_str: str = df.select("company_name").row(0)[0]
    assert sample_str == sample_str.upper()


# Sources


def test_source_init():
    """Test that Source can be instantiated with valid parameters."""
    # Create a location
    location = RelationalDBLocation(uri="sqlite:///test.db")

    # Create basic fields
    fields = (
        SourceField(name="id", type=DataTypes.STRING, identifier=True),
        SourceField(name="name", type=DataTypes.STRING),
        SourceField(name="age", type=DataTypes.INT64),
    )

    # Create a Source
    source = Source(
        location=location,
        resolution_name="test_source",
        extract_transform="SELECT id, name, age FROM users",
        fields=fields,
    )

    et_hash = HASH_FUNC("SELECT id, name, age FROM users".encode("utf-8")).hexdigest()

    assert source.location == location
    assert source.resolution_name == "test_source"
    assert source.extract_transform == "SELECT id, name, age FROM users"
    assert source.fields == fields
    assert source.identifier.name == "id"
    assert source.column_qualifier == f"test_source_{et_hash[:6]}"


def test_source_field_validation():
    """Test validation of Source fields."""
    location = RelationalDBLocation(uri="sqlite:///test.db")

    # Test with missing identifier
    invalid_fields = (
        SourceField(name="name", type=DataTypes.STRING),
        SourceField(name="age", type=DataTypes.INT64),
    )

    with pytest.raises(ValueError):
        Source(
            location=location,
            resolution_name="test_source",
            extract_transform="SELECT name, age FROM users",
            fields=invalid_fields,
        )

    # Test with multiple identifiers
    invalid_fields_multiple_ids = (
        SourceField(name="id", type=DataTypes.STRING, identifier=True),
        SourceField(name="uuid", type=DataTypes.STRING, identifier=True),
        SourceField(name="name", type=DataTypes.STRING),
    )

    with pytest.raises(ValueError):
        Source(
            location=location,
            resolution_name="test_source",
            extract_transform="SELECT id, uuid, name FROM users",
            fields=invalid_fields_multiple_ids,
        )

    # Test with non-string identifier
    invalid_fields_non_string_id = (
        SourceField(name="id", type=DataTypes.INT64, identifier=True),
        SourceField(name="name", type=DataTypes.STRING),
    )

    with pytest.raises(ValueError):
        Source(
            location=location,
            resolution_name="test_source",
            extract_transform="SELECT id, name FROM users",
            fields=invalid_fields_non_string_id,
        )


def test_source_from_location(mocker):
    """Test creating a Source from a Location."""
    # Mock a location with execute method
    location = RelationalDBLocation(uri="sqlite:///test.db")

    # Create a mock DataFrame to return
    mock_df = pl.DataFrame(
        {
            "id": ["1", "2", "3"],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }
    )

    # Mock the execute method
    mocker.patch.object(Location, "execute", return_value=mock_df)

    # Test Source.from_location
    source = Source.from_location(
        location=location, extract_transform="SELECT id, name, age FROM users"
    )

    assert source.location == location
    assert source.extract_transform == "SELECT id, name, age FROM users"
    assert len(source.fields) == 3
    assert source.fields[0].name == "id"
    assert source.fields[0].identifier is True
    assert source.resolution_name.startswith(location.uri.host)


def test_source_set_credentials():
    """Test setting credentials on a Source."""
    # Create a location
    location = RelationalDBLocation(uri="sqlite:///test.db")

    # Create a Source
    source = Source(
        location=location,
        resolution_name="test_source",
        extract_transform="SELECT id, name FROM users",
        fields=(
            SourceField(name="id", type=DataTypes.STRING, identifier=True),
            SourceField(name="name", type=DataTypes.STRING),
        ),
    )

    # Mock credentials
    credentials = create_engine("sqlite:///test.db")

    # Mock the add_credentials method
    location_with_creds = copy.deepcopy(location)
    location_with_creds.credentials = credentials

    # Patch the add_credentials method
    source.location.add_credentials = lambda x: location_with_creds

    # Set credentials
    new_source = source.set_credentials(credentials)

    # Verify
    assert new_source.location.credentials == credentials
    assert source.location.credentials is None  # Original source unchanged


class MockEngine:
    """Mock engine for testing."""

    def __init__(self):
        self.url = None

    def execute(self, *args, **kwargs):
        """Mock execute method."""
        return []


@pytest.mark.parametrize(
    "return_batches, batch_size, return_type",
    [
        (False, 100, "polars"),
        (True, 50, "polars"),
        (False, 100, "arrow"),
        (True, 50, "pandas"),
    ],
)
def test_source_query(return_batches, batch_size, return_type, mocker):
    """Test Source query method with different parameters."""
    # Create a location
    location = RelationalDBLocation(uri="sqlite:///test.db")
    location.credentials = MockEngine()

    # Create a Source
    source = Source(
        location=location,
        resolution_name="test_source",
        extract_transform="SELECT id, name FROM users",
        fields=(
            SourceField(name="id", type=DataTypes.STRING, identifier=True),
            SourceField(name="name", type=DataTypes.STRING),
        ),
    )

    # Create mock return values
    if return_type == "polars":
        mock_return = pl.DataFrame({"id": ["1", "2"], "name": ["Alice", "Bob"]})
    elif return_type == "arrow":
        mock_return = pa.Table.from_pydict({"id": ["1", "2"], "name": ["Alice", "Bob"]})
    else:  # pandas
        mock_return = pd.DataFrame({"id": ["1", "2"], "name": ["Alice", "Bob"]})

    # Mock the location.execute method
    if return_batches:
        mocker.patch.object(location, "execute", return_value=iter([mock_return]))
    else:
        mocker.patch.object(location, "execute", return_value=mock_return)

    # Call query method
    result = source.query(
        return_batches=return_batches,
        batch_size=batch_size,
        return_type=return_type,
    )

    # Check results
    if return_batches:
        assert isinstance(result, Iterator)
        first_batch = next(result)
        if return_type == "polars":
            assert isinstance(first_batch, pl.DataFrame)
        elif return_type == "arrow":
            assert isinstance(first_batch, pa.Table)
        else:  # pandas
            assert isinstance(first_batch, pd.DataFrame)
    else:
        if return_type == "polars":
            assert isinstance(result, pl.DataFrame)
        elif return_type == "arrow":
            assert isinstance(result, pa.Table)
        else:  # pandas
            assert isinstance(result, pd.DataFrame)


def test_source_query_with_qualification(mocker):
    """Test Source query method with name qualification."""
    # Create a location
    location = RelationalDBLocation(uri="sqlite:///test.db")
    location.credentials = MockEngine()

    # Create a Source
    source = Source(
        location=location,
        resolution_name="test_source",
        extract_transform="SELECT id, name FROM users",
        fields=(
            SourceField(name="id", type=DataTypes.STRING, identifier=True),
            SourceField(name="name", type=DataTypes.STRING),
        ),
    )

    # Create mock return values and capture renamed columns
    renamed_columns = {}

    def mock_execute(
        extract_transform, rename, batch_size, return_batches, return_type
    ):
        """Mock execute that captures the rename function."""
        if rename:
            renamed_columns["id"] = rename("id")
            renamed_columns["name"] = rename("name")
        return pl.DataFrame({"id": ["1", "2"], "name": ["Alice", "Bob"]})

    mocker.patch.object(location, "execute", side_effect=mock_execute)

    # Call query method with qualify_names=True
    source.query(qualify_names=True)

    # Verify column renaming
    assert renamed_columns["id"].startswith("test_source_")
    assert renamed_columns["name"].startswith("test_source_")


def test_source_query_requires_credentials():
    """Test that query raises an error if credentials are not set."""
    # Create a location without credentials
    location = RelationalDBLocation(uri="sqlite:///test.db")

    # Create a Source
    source = Source(
        location=location,
        resolution_name="test_source",
        extract_transform="SELECT id, name FROM users",
        fields=(
            SourceField(name="id", type=DataTypes.STRING, identifier=True),
            SourceField(name="name", type=DataTypes.STRING),
        ),
    )

    # Call query method should raise error
    with pytest.raises(MatchboxSourceCredentialsError):
        source.query()


def test_source_hash_data(mocker):
    """Test Source hash_data method."""
    # Create a location
    location = RelationalDBLocation(uri="sqlite:///test.db")
    location.credentials = MockEngine()

    # Create a Source
    source = Source(
        location=location,
        resolution_name="test_source",
        extract_transform="SELECT id, name, age FROM users",
        fields=(
            SourceField(name="id", type=DataTypes.STRING, identifier=True),
            SourceField(name="name", type=DataTypes.STRING),
            SourceField(name="age", type=DataTypes.INTEGER),
        ),
    )

    # Create mock return values
    mock_df = pl.DataFrame(
        {"id": ["1", "2"], "name": ["Alice", "Bob"], "age": [25, 30]}
    )

    mocker.patch.object(location, "execute", return_value=mock_df)

    # Mock the hash_rows function
    mock_hash = pl.Series(["hash1", "hash2"])
    mocker.patch("matchbox.common.sources.hash_rows", return_value=mock_hash)

    # Call hash_data method
    result = source.hash_data(batch_size=100)

    # Verify result structure
    assert isinstance(result, pa.Table)
    assert result.column_names == ["hash", "source_pk"]

    # Verify result values
    assert len(result) == 2

    # Calling with batch_size should process in batches
    result_batched = source.hash_data(batch_size=1)
    assert isinstance(result_batched, pa.Table)


def test_source_hash_data_with_null_pk(mocker):
    """Test that hash_data raises an error if source_pk contains nulls."""
    # Create a location
    location = RelationalDBLocation(uri="sqlite:///test.db")
    location.credentials = MockEngine()

    # Create a Source
    source = Source(
        location=location,
        resolution_name="test_source",
        extract_transform="SELECT id, name FROM users",
        fields=(
            SourceField(name="id", type=DataTypes.STRING, identifier=True),
            SourceField(name="name", type=DataTypes.STRING),
        ),
    )

    # Create mock return values with null id
    mock_df = pl.DataFrame({"id": ["1", None], "name": ["Alice", "Bob"]})

    mocker.patch.object(location, "execute", return_value=mock_df)

    # Call hash_data method should raise error
    with pytest.raises(ValueError, match="source_pk column contains null values"):
        source.hash_data()


def test_source_equality_and_hash():
    """Test Source __eq__ and __hash__ methods."""
    # Create two identical sources
    location1 = RelationalDBLocation(uri="sqlite:///test.db")
    location2 = RelationalDBLocation(uri="sqlite:///test.db")

    fields1 = (
        SourceField(name="id", type=DataTypes.STRING, identifier=True),
        SourceField(name="name", type=DataTypes.STRING),
    )

    fields2 = (
        SourceField(name="id", type=DataTypes.STRING, identifier=True),
        SourceField(name="name", type=DataTypes.STRING),
    )

    source1 = Source(
        location=location1,
        resolution_name="test_source",
        extract_transform="SELECT id, name FROM users",
        fields=fields1,
    )

    source2 = Source(
        location=location2,
        resolution_name="test_source",
        extract_transform="SELECT id, name FROM users",
        fields=fields2,
    )

    # Add credentials to one source
    location1.credentials = MockEngine()

    # Verify equality and hash
    assert source1 == source2
    assert hash(source1) == hash(source2)

    # Modify one source
    source3 = Source(
        location=location1,
        resolution_name="different",
        extract_transform="SELECT id, name FROM users",
        fields=fields1,
    )

    assert source1 != source3
    assert hash(source1) != hash(source3)


def test_match_validation():
    """Test Match validation rules."""
    # Create source addresses
    source_addr = SourceAddress(full_name="source_table", warehouse_hash=b"hash1")
    target_addr = SourceAddress(full_name="target_table", warehouse_hash=b"hash2")

    # Valid match with cluster and source_id
    valid_match = Match(
        cluster=1,
        source=source_addr,
        source_id={"src1", "src2"},
        target=target_addr,
        target_id={"tgt1", "tgt2"},
    )

    assert valid_match.cluster == 1
    assert valid_match.source_id == {"src1", "src2"}
    assert valid_match.target_id == {"tgt1", "tgt2"}

    # Invalid: target_id but no cluster
    with pytest.raises(ValueError):
        Match(
            cluster=None,
            source=source_addr,
            source_id={"src1", "src2"},
            target=target_addr,
            target_id={"tgt1", "tgt2"},
        )

    # Invalid: cluster but no source_id
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source=source_addr,
            source_id=set(),
            target=target_addr,
            target_id={"tgt1", "tgt2"},
        )

    # Valid: No target_id with cluster and source_id
    valid_no_target = Match(
        cluster=1,
        source=source_addr,
        source_id={"src1", "src2"},
        target=target_addr,
        target_id=set(),
    )

    assert valid_no_target.cluster == 1
    assert valid_no_target.source_id == {"src1", "src2"}
    assert valid_no_target.target_id == set()


def test_source_address_methods():
    """Test SourceAddress methods."""
    # Create a source address
    addr = SourceAddress(full_name="schema.table", warehouse_hash=b"warehouse_hash")

    # Test string representation
    assert str(addr) == f"schema.table@{addr.warehouse_hash_b64}"

    # Test pretty representation
    assert addr.pretty == f"schema.table@{addr.warehouse_hash_b64[:5]}..."

    # Test format_column method
    assert addr.format_column("id") == "schema.table_id"

    # Test compose method (mocked)
    mock_engine = MockEngine()
    mock_engine.url = AnyUrl("sqlite:///test.db")

    # Mock the get_dialect method
    mock_engine.url.get_dialect = lambda: type("obj", (object,), {"name": "sqlite"})
    mock_engine.url.database = "test"
    mock_engine.url.host = "localhost"
    mock_engine.url.port = 1234
    mock_engine.url.schema = "main"
    mock_engine.url.query = {}

    # Test compose
    composed_addr = SourceAddress.compose(mock_engine, "schema.table")
    assert composed_addr.full_name == "schema.table"
    assert isinstance(composed_addr.warehouse_hash, bytes)


# Match


def test_match_validates():
    """Match objects are validated when they're instantiated."""
    Match(
        cluster=1,
        source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
        source_id={"a"},
        target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
        target_id={"b"},
    )

    # Missing source_id with target_id
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
            target_id={"b"},
        )

    # Missing cluster with target_id
    with pytest.raises(ValueError):
        Match(
            source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
            source_id={"a"},
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
            target_id={"b"},
        )

    # Missing source_id with cluster
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
        )
