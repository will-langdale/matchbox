# def test_source_reader_validates():
#     """Fields passed to a SourceReader are checked against the source table."""


# def test_source_set_engine(sqlite_warehouse: Engine):
#     """Engine can be set on Source."""
#     source_testkit = source_factory(
#         features=[{"name": "b", "base_generator": "random_int", "sql_type": "BIGINT"}]
#         engine=sqlite_warehouse,
#     )
#     source_testkit.to_warehouse(engine=sqlite_warehouse)

#     # We can set engine with correct column specification
#     source = source_testkit.source.set_engine(sqlite_warehouse)
#     assert isinstance(source, Source)

#     # Error is raised with missing column
#     with pytest.raises(MatchboxSourceColumnError, match="Column c not available in"):
#         new_source = source_testkit.source.model_copy(
#             update={"columns": (SourceColumn(name="c", type="TEXT"),)}
#         )
#         new_source.set_engine(sqlite_warehouse)

#     # Error is raised with wrong type
#     with pytest.raises(MatchboxSourceColumnError, match="Type BIGINT != TEXT for b"):
#         new_source = source_testkit.source.model_copy(
#             update={"columns": (SourceColumn(name="b", type="TEXT"),)}
#         )
#         new_source.set_engine(sqlite_warehouse)


# Engine validates correctly

# hash data overlaps, and doesn't


# def test_source_format_columns():
#     """Column names can get a standard prefix from a table name."""
#     source1 = Source(
#         address=SourceAddress(full_name="foo", warehouse_hash=b"bar"), db_pk="i"
#     )

#     source2 = Source(
#         address=SourceAddress(full_name="foo.bar", warehouse_hash=b"bar"), db_pk="i"
#     )

#     assert source1.format_column("col") == "foo_col"
#     assert source2.format_column("col") == "foo_bar_col"


# def test_source_default_columns(sqlite_warehouse: Engine):
#     """Default columns from the warehouse can be assigned to a Source."""
#     source_testkit = source_factory(
#         features=[
#             {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
#             {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
#         ],
#         engine=sqlite_warehouse,
#     )

#     source_testkit.to_warehouse(engine=sqlite_warehouse)

#     expected_columns = (
#         SourceColumn(name="a", type="BIGINT"),
#         SourceColumn(name="b", type="TEXT"),
#     )

#     source = source_testkit.source.set_engine(sqlite_warehouse).default_columns()

#     assert source.columns == expected_columns
#     # We create a new source, but attributes and engine match
#     assert source is not source_testkit.source
#     assert source == source_testkit.source
#     assert source.engine == sqlite_warehouse


# def test_source_to_table(sqlite_warehouse: Engine):
#     """Convert Source to SQLAlchemy Table."""
#     source_testkit = source_factory(engine=sqlite_warehouse)
#     source_testkit.to_warehouse(engine=sqlite_warehouse)

#     source = source_testkit.source.set_engine(sqlite_warehouse)

#     assert isinstance(source.to_table(), Table)


# def test_source_to_arrow_to_pandas(sqlite_warehouse: Engine):
#     """Convert Source to Arrow table or Pandas dataframe with options."""
#     source_testkit = source_factory(
#         features=[
#             {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
#             {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
#         ],
#         engine=sqlite_warehouse,
#         n_true_entities=2,
#     )
#     source_testkit.to_warehouse(engine=sqlite_warehouse)
#     source = source_testkit.source.set_engine(sqlite_warehouse).default_columns()
#     prefix = fullname_to_prefix(source_testkit.source.address.full_name)
#     expected_df_prefixed = (
#         source_testkit.data.to_pandas().drop(columns=["id"]).add_prefix(prefix)
#     )

#     # Test basic conversion
#     assert_frame_equal(
#         expected_df_prefixed, source.to_pandas(), check_like=True, check_dtype=False
#     )
#     assert_frame_equal(
#         expected_df_prefixed,
#         source.to_arrow().to_pandas(),
#         check_like=True,
#         check_dtype=False,
#     )

#     # Test with limit parameter
#     assert_frame_equal(
#         expected_df_prefixed.iloc[:1],
#         source.to_pandas(limit=1),
#         check_like=True,
#         check_dtype=False,
#     )
#     assert_frame_equal(
#         expected_df_prefixed.iloc[:1],
#         source.to_arrow(limit=1).to_pandas(),
#         check_like=True,
#         check_dtype=False,
#     )

#     # Test with fields parameter
#     assert_frame_equal(
#         expected_df_prefixed[[f"{prefix}pk", f"{prefix}a"]],
#         source.to_pandas(fields=["a"]),
#         check_like=True,
#         check_dtype=False,
#     )
#     assert_frame_equal(
#         expected_df_prefixed[[f"{prefix}pk", f"{prefix}a"]],
#         source.to_arrow(fields=["a"]).to_pandas(),
#         check_like=True,
#         check_dtype=False,
#     )


# def test_source_hash_data(sqlite_warehouse: Engine):
#     """A Source can output hashed versions of its rows."""
#     source_testkit = source_factory(
#         features=[
#             {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
#             {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
#         ],
#         engine=sqlite_warehouse,
#         n_true_entities=2,
#         repetition=1,
#     )

#     source_testkit.to_warehouse(engine=sqlite_warehouse)
#     source = source_testkit.source.set_engine(sqlite_warehouse).default_columns()

#     res = source.hash_data().to_pandas()
#     assert len(res) == 2
#     assert len(res.source_pk.iloc[0]) == 2
#     assert len(res.source_pk.iloc[1]) == 2

#     result = source.hash_data(iter_batches=True, batch_size=3)
#     assert isinstance(result, pa.Table)


# @pytest.mark.parametrize(
#     ("method_name", "return_type"),
#     [
#         pytest.param("to_arrow", pa.Table, id="to_arrow"),
#         pytest.param("to_pandas", pd.DataFrame, id="to_pandas"),
#     ],
# )
# def test_source_data_batching(method_name, return_type, sqlite_warehouse: Engine):
#     """Test Source data retrieval methods with batching parameters."""
#     # Create a source with multiple rows of data
#     source_testkit = source_factory(
#         features=[
#             {"name": "a", "base_generator": "random_int", "sql_type": "BIGINT"},
#             {"name": "b", "base_generator": "word", "sql_type": "TEXT"},
#         ],
#         engine=sqlite_warehouse,
#         n_true_entities=9,
#     )
#     source_testkit.to_warehouse(engine=sqlite_warehouse)
#     source = source_testkit.source.set_engine(sqlite_warehouse).default_columns()

#     # Call the method with batching
#     method = getattr(source, method_name)
#     batch_iterator = method(iter_batches=True, batch_size=3)
#     batches = list(batch_iterator)

#     # Verify we got the expected number of batches
#     assert len(batches) == 3
#     for batch in batches:
#         assert isinstance(batch, return_type)
#         assert len(batch) == 3
