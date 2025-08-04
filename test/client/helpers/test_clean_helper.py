import duckdb

from matchbox.client.helpers.clean import compose, vectorise


def test_compose():
    """Test compose function with multiple transformations."""

    def to_upper(col):
        return f"upper({col})"

    def add_prefix(col):
        return f"'prefix_' || {col}"

    # Single function
    assert compose(to_upper)("name") == "upper(name)"

    # Multiple functions (left-to-right)
    assert compose(to_upper, add_prefix)("name") == "'prefix_' || upper(name)"

    # No functions (identity)
    assert compose()("name") == "name"


def test_vectorise():
    """Test vectorise function with list_transform."""

    def to_upper(col):
        return f"upper({col})"

    def complex_func(col):
        return f"trim(lower({col}))"

    assert vectorise(to_upper)("arr") == "list_transform(arr, x -> upper(x))"
    assert vectorise(complex_func)("arr") == "list_transform(arr, x -> trim(lower(x)))"


def test_compose_and_vectorise():
    """Test combining compose and vectorise."""

    def to_upper(col):
        return f"upper({col})"

    def trim_spaces(col):
        return f"trim({col})"

    # Vectorise composed functions
    composed = compose(to_upper, trim_spaces)
    result = vectorise(composed)("arr")
    assert result == "list_transform(arr, x -> trim(upper(x)))"


def test_duckdb_integration():
    """Test that generated SQL actually works in DuckDB."""

    # Test compose with scalar data
    def to_upper(col):
        return f"upper({col})"

    def add_prefix(col):
        return f"'hello_' || {col}"

    composed = compose(to_upper, add_prefix)

    with duckdb.connect(":memory:") as conn:
        sql = f"""
            select 
                {composed("name")} as result
            from 
                (values ('world')) t(name)
        """
        result = conn.execute(sql).fetchone()[0]
        assert result == "hello_WORLD"

    # Test vectorise with array data
    def trim_lower(col):
        return f"trim(lower({col}))"

    vectorised = vectorise(trim_lower)

    with duckdb.connect(":memory:") as conn:
        sql = f"""
            select 
                {vectorised("names")} as result
            from 
                (values ([' ALICE ', ' BOB '])) t(names)
        """
        result = conn.execute(sql).fetchone()[0]
        assert result == ["alice", "bob"]

    # Test combined compose + vectorise
    composed_trim = compose(to_upper, lambda col: f"trim({col})")
    vectorised_composed = vectorise(composed_trim)

    with duckdb.connect(":memory:") as conn:
        sql = f"""
            select 
                {vectorised_composed("names")} as result
            from 
                (values ([' alice ', ' bob '])) t(names)
        """
        result = conn.execute(sql).fetchone()[0]
        assert result == ["ALICE", "BOB"]
