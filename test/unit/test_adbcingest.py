import unittest
from unittest.mock import patch, MagicMock
import pyarrow as pa
import pandas as pd
from sqlalchemy.engine import Engine, Connection


from matchbox.server.postgresql.utils.db import adbc_ingest_data, _adbc_insert_data, _save_to_postgresql, _run_query, _run_queries


class TestAdbcIngestData(unittest.TestCase):

    @patch('my_module._create_adbc_table_constraints')
    @patch('my_module._adbc_insert_data')
    @patch('my_module.datetime')
    def test_adbc_ingest_data(self, mock_datetime, mock_adbc_insert_data, mock_create_adbc_table_constraints):
        # Mock datetime
        mock_datetime.now.return_value.strftime.return_value = "20250101123045"

        # Create mock arguments
        clusters = pa.Table.from_pandas(pd.DataFrame({"column1": [1, 2], "column2": [3, 4]}))
        contains = pa.Table.from_pandas(pd.DataFrame({"column1": [1, 2], "column2": [3, 4]}))
        probabilities = pa.Table.from_pandas(pd.DataFrame({"column1": [1, 2], "column2": [3, 4]}))
        engine = MagicMock(spec=Engine)
        resolution_id = 1

        # Mock the engine connection context manager
        mock_connection = engine.connect.return_value.__enter__.return_value

        # Test when _adbc_insert_data returns True
        mock_adbc_insert_data.return_value = True
        mock_create_adbc_table_constraints.return_value = True
        result = adbc_ingest_data(clusters, contains, probabilities, engine, resolution_id)
        self.assertTrue(result)

        # Test when _adbc_insert_data returns False
        mock_adbc_insert_data.return_value = False
        result = adbc_ingest_data(clusters, contains, probabilities, engine, resolution_id)
        self.assertFalse(result)


@patch('my_module._save_to_postgresql')
@patch('my_module._run_query')
@patch('my_module.adbc_driver_postgresql.dbapi.connect')
def test_adbc_insert_data(self, mock_connect, mock_run_query, mock_save_to_postgresql):
    # Mock the connect method
    mock_conn = mock_connect.return_value.__enter__.return_value

    # Create mock arguments
    clusters = pa.Table.from_pandas(pd.DataFrame({"column1": [1, 2], "column2": [3, 4]}))
    contains = pa.Table.from_pandas(pd.DataFrame({"column1": [1, 2], "column2": [3, 4]}))
    probabilities = pa.Table.from_pandas(pd.DataFrame({"column1": [1, 2], "column2": [3, 4]}))
    suffix = "20250101123045"
    alchemy_conn = MagicMock()
    resolution_id = 1

    # Test when all queries and saves succeed
    mock_run_query.side_effect = [None, None, None]
    mock_save_to_postgresql.side_effect = [None, None, None]
    result = _adbc_insert_data(clusters, contains, probabilities, suffix, alchemy_conn, resolution_id)
    self.assertTrue(result)

    # Test when a query fails
    mock_run_query.side_effect = Exception("Query failed")
    result = _adbc_insert_data(clusters, contains, probabilities, suffix, alchemy_conn, resolution_id)
    self.assertFalse(result)

    # Test when save_to_postgresql fails
    mock_run_query.side_effect = [None, None, None]
    mock_save_to_postgresql.side_effect = [None, Exception("Save failed"), None]
    result = _adbc_insert_data(clusters, contains, probabilities, suffix, alchemy_conn, resolution_id)
    self.assertFalse(result)

    @patch('my_module.pa.RecordBatchReader.from_batches')
    def test_save_to_postgresql(self, mock_from_batches):
        # Mock the from_batches method
        mock_batch_reader = MagicMock()
        mock_from_batches.return_value = mock_batch_reader

        # Create mock arguments
        table = pa.Table.from_pandas(pd.DataFrame({"column1": [1, 2], "column2": [3, 4]}))
        conn = MagicMock()
        schema = "test_schema"
        table_name = "test_table"

        # Mock the cursor context manager
        mock_cursor = conn.cursor.return_value.__enter__.return_value

        # Call the function
        _save_to_postgresql(table, conn, schema, table_name)

        # Verify the cursor method was called correctly
        mock_cursor.adbc_ingest.assert_called_once_with(
            table_name=table_name,
            data=mock_batch_reader,
            mode="append",
            db_schema_name=schema,
        )

    @patch('my_module.text')
    def test_run_query(self, mock_text):
        # Create mock arguments
        query = "SELECT * FROM test_table"
        conn = MagicMock(spec=Connection)

        # Call the function
        _run_query(query, conn)

        # Verify the execute method was called correctly
        conn.execute.assert_called_once_with(mock_text(query))
        conn.commit.assert_called_once()

    @patch('my_module.text')
    def test_run_queries(self, mock_text):
        # Create mock arguments
        queries = ["SELECT * FROM test_table", "DELETE FROM test_table"]
        conn = MagicMock(spec=Connection)

        # Call the function
        _run_queries(queries, conn)

        # Verify the execute method was called correctly for each query
        conn.begin.assert_called_once()
        self.assertEqual(conn.execute.call_count, len(queries))
        for query in queries:
            conn.execute.assert_any_call(mock_text(query))


if __name__ == '__main__':
    unittest.main()
