from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import UploadFile
from pandas import DataFrame
from sqlalchemy import create_engine

from matchbox.common.exceptions import (
    MatchboxServerFileError,
)
from matchbox.common.sources import Source, SourceAddress, SourceColumn
from matchbox.server.api.cache import MetadataSchema, MetadataStore
from matchbox.server.api.routes import s3_to_recordbatch, table_to_s3

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


@pytest.mark.asyncio
async def test_file_to_s3(s3: S3Client, all_companies: DataFrame):
    """Test that a file can be uploaded to S3."""
    # Create a mock bucket
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Test 1: Upload a parquet file
    # Create a mock UploadFile
    all_companies["id"] = all_companies["id"].astype(str)
    table = pa.Table.from_pandas(all_companies)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    file_content = sink.getvalue().to_pybytes()

    parquet_file = UploadFile(filename="test.parquet", file=BytesIO(file_content))

    # Call the function
    key = "foo.parquet"
    upload_id = await table_to_s3(
        client=s3,
        bucket="test-bucket",
        key=key,
        file=parquet_file,
        expected_schema=table.schema,
    )
    # Validate response
    assert key == upload_id

    response_table = pa.Table.from_batches(
        [
            batch
            async for batch in s3_to_recordbatch(
                client=s3, bucket="test-bucket", key=upload_id
            )
        ]
    )

    assert response_table.equals(table)

    # Test 2: Upload a non-parquet file
    text_file = UploadFile(filename="test.txt", file=BytesIO(b"test"))

    with pytest.raises(MatchboxServerFileError):
        await table_to_s3(
            client=s3,
            bucket="test-bucket",
            key=key,
            file=text_file,
            expected_schema=table.schema,
        )

    # Test 3: Upload a parquet file with a different schema
    corrupted_schema = table.schema.remove(0)
    with pytest.raises(MatchboxServerFileError):
        await table_to_s3(
            client=s3,
            bucket="test-bucket",
            key=key,
            file=parquet_file,
            expected_schema=corrupted_schema,
        )


def test_basic_cache_and_retrieve():
    """Test basic caching and retrieval functionality."""
    store = MetadataStore()
    source = Source(
        address=SourceAddress.compose(
            full_name="test.source", engine=create_engine("sqlite:///:memory:")
        ),
        db_pk="pk",
        columns=[SourceColumn(name="col1")],
    )

    # Cache the source
    cache_id = store.cache_source(source)
    assert isinstance(cache_id, str)

    # Retrieve and verify
    entry = store.get(cache_id)
    assert entry is not None
    assert entry.metadata == source
    assert entry.upload_schema == MetadataSchema.index
    assert isinstance(entry.timestamp, datetime)


@patch("matchbox.server.api.cache.datetime")
def test_expiration(mock_datetime: Mock):
    """Test that entries expire correctly."""
    store = MetadataStore(expiry_minutes=30)
    source = Source(
        address=SourceAddress.compose(
            full_name="test.source", engine=create_engine("sqlite:///:memory:")
        ),
        db_pk="pk",
        columns=[SourceColumn(name="col1")],
    )

    # Cache the source
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    cache_id = store.cache_source(source)

    # Should still be valid after 29 minutes
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 29)
    assert store.get(cache_id) is not None

    # Should be expired after 31 minutes
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 31)
    assert store.get(cache_id) is None


@patch("matchbox.server.api.cache.datetime")
def test_cleanup(mock_datetime: Mock):
    """Test that cleanup removes expired entries."""
    store = MetadataStore(expiry_minutes=30)
    source = Source(
        address=SourceAddress.compose(
            full_name="test.source", engine=create_engine("sqlite:///:memory:")
        ),
        db_pk="pk",
        columns=[SourceColumn(name="col1")],
    )

    # Create two entries
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    id1 = store.cache_source(source)
    id2 = store.cache_source(source)

    # Move time forward past expiration
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 31)

    # Adding a new entry should trigger cleanup
    id3 = store.cache_source(source)

    # First two should be gone, new one should exist
    assert store.get(id1) is None
    assert store.get(id2) is None
    assert store.get(id3) is not None


def test_remove():
    """Test manual removal of cache entries."""
    store = MetadataStore()
    source = Source(
        address=SourceAddress.compose(
            full_name="test.source", engine=create_engine("sqlite:///:memory:")
        ),
        db_pk="pk",
        columns=[SourceColumn(name="col1")],
    )

    # Cache and remove
    cache_id = store.cache_source(source)
    assert store.remove(cache_id) is True
    assert store.get(cache_id) is None

    # Try removing non-existent entry
    assert store.remove("nonexistent") is False
