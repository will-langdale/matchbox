import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import UploadFile

from matchbox.common.dtos import BackendUploadType
from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.factories.sources import source_factory
from matchbox.server.api.arrow import s3_to_recordbatch, table_to_s3
from matchbox.server.api.uploads import UploadTracker, heartbeat

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


def test_file_to_s3(s3: S3Client):
    """Test that a file can be uploaded to S3."""
    # Create a mock bucket
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )
    source_testkit = source_factory(
        features=[
            {"name": "company_name", "base_generator": "company"},
            {"name": "address", "base_generator": "address"},
            {
                "name": "crn",
                "base_generator": "bothify",
                "parameters": (("text", "???-###-???-###"),),
            },
            {
                "name": "duns",
                "base_generator": "bothify",
                "parameters": (("text", "??######"),),
            },
        ],
    )
    all_companies = source_testkit.query.to_pandas()

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
    upload_id = table_to_s3(
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
            for batch in s3_to_recordbatch(
                client=s3, bucket="test-bucket", key=upload_id
            )
        ]
    )

    assert response_table.equals(table)

    # Test 2: Upload a non-parquet file
    text_file = UploadFile(filename="test.txt", file=BytesIO(b"test"))

    with pytest.raises(MatchboxServerFileError):
        table_to_s3(
            client=s3,
            bucket="test-bucket",
            key=key,
            file=text_file,
            expected_schema=table.schema,
        )

    # Test 3: Upload a parquet file with a different schema
    corrupted_schema = table.schema.remove(0)
    with pytest.raises(MatchboxServerFileError):
        table_to_s3(
            client=s3,
            bucket="test-bucket",
            key=key,
            file=parquet_file,
            expected_schema=corrupted_schema,
        )


def test_basic_upload_tracking():
    """Test adding upload to tracker and retrieving."""
    tracker = UploadTracker()
    source = source_factory().source_config

    # Add the source
    upload_id = tracker.add_source(source)
    assert isinstance(upload_id, str)

    # Retrieve and verify
    entry = tracker.get(upload_id)
    assert entry is not None
    assert entry.metadata == source
    assert entry.upload_type.schema == BackendUploadType.INDEX.schema
    assert isinstance(entry.update_timestamp, datetime)

    # Verify initial status
    assert entry.status.status == "awaiting_upload"
    assert entry.status.entity == BackendUploadType.INDEX
    assert entry.status.id == upload_id


@patch("matchbox.server.api.uploads.datetime")
def test_expiration(mock_datetime: Mock):
    """Test that entries expire correctly after period of inactivity."""
    tracker = UploadTracker(expiry_minutes=30)
    source = source_factory().source_config

    # Add the source
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    upload_id = tracker.add_source(source)

    # Should still be valid after 29 minutes
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 29)
    assert tracker.get(upload_id) is not None

    # Status update should refresh timestamp
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 30)
    tracker.update_status(upload_id, "processing")

    # Should still be valid 29 minutes after status update
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 59)
    assert tracker.get(upload_id) is not None

    # Should expire 31 minutes after status update
    mock_datetime.now.return_value = datetime(2024, 1, 1, 13, 30)
    assert tracker.get(upload_id) is None


@patch("matchbox.server.api.uploads.datetime")
def test_cleanup(mock_datetime: Mock):
    """Test that cleanup removes expired entries."""
    tracker = UploadTracker(expiry_minutes=30)
    source = source_factory().source_config

    # Create two entries
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    id1 = tracker.add_source(source)
    id2 = tracker.add_source(source)

    # Update status of one entry
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 15)
    tracker.update_status(id2, "processing")

    # Move time forward past expiration for first entry but not second
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 31)

    # Adding a new entry should trigger cleanup
    id3 = tracker.add_source(source)

    # First should be gone, second should exist due to status update
    assert tracker.get(id1) is None
    assert tracker.get(id2) is not None
    assert tracker.get(id3) is not None


def test_remove():
    """Test manual removal of cache entries."""
    tracker = UploadTracker()
    source = source_factory().source_config

    # Cache and remove
    upload_id = tracker.add_source(source)
    assert tracker.remove(upload_id) is True
    assert tracker.get(upload_id) is None

    # Try removing non-existent entry
    assert tracker.remove("nonexistent") is False


def test_status_management():
    """Test status update functionality."""
    tracker = UploadTracker()
    source = source_factory().source_config

    # Create entry and verify initial status
    upload_id = tracker.add_source(source)
    entry = tracker.get(upload_id)
    assert entry.status.status == "awaiting_upload"

    # Update status
    assert tracker.update_status(upload_id, "processing") is True
    entry = tracker.get(upload_id)
    assert entry.status.status == "processing"

    # Update with details
    assert tracker.update_status(upload_id, "failed", "Error details") is True
    entry = tracker.get(upload_id)
    assert entry.status.status == "failed"
    assert entry.status.details == "Error details"

    # Try updating non-existent entry
    with pytest.raises(KeyError):
        tracker.update_status("nonexistent", "processing")


@patch("matchbox.server.api.uploads.datetime")
def test_timestamp_updates(mock_datetime: Mock):
    """Test that timestamps update correctly on different operations."""
    tracker = UploadTracker()
    source = source_factory().source_config

    # Initial creation
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    upload_id = tracker.add_source(source)
    entry = tracker.get(upload_id)
    assert entry.update_timestamp == datetime(2024, 1, 1, 12, 0)

    # Status update
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 15)
    tracker.update_status(upload_id, "processing")
    entry = tracker.get(upload_id)
    assert entry.update_timestamp == datetime(2024, 1, 1, 12, 15)

    # Get operation
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 30)
    entry = tracker.get(upload_id)
    assert entry.update_timestamp == datetime(2024, 1, 1, 12, 30)


def wait_for_heartbeat(
    tracker: UploadTracker,
    upload_id: str,
    timeout: float = 1.0,
    poll_interval: float = 0.05,
):
    """Wait until heartbeat updates status or timeout is reached."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        entry = tracker.get(upload_id)
        if (
            entry
            and entry.status.details
            and "Still processing... Last heartbeat:" in entry.status.details
        ):
            return entry
        time.sleep(poll_interval)
    # Return latest entry even if heartbeat not found
    return tracker.get(upload_id)


@patch("matchbox.server.api.uploads.datetime")
def test_heartbeat_updates_status(mock_datetime: Mock):
    """Test that heartbeat updates status periodically."""
    tracker = UploadTracker()
    source = source_factory().source_config

    # Create initial entry
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    upload_id = tracker.add_source(source)

    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 5)
    with heartbeat(tracker, upload_id, interval_seconds=0.1):
        wait_for_heartbeat(tracker, upload_id)

    # Verify the status was updated with heartbeat details
    entry = tracker.get(upload_id)
    assert entry.status.status == "processing"
    assert "Still processing... Last heartbeat:" in entry.status.details
    assert entry.update_timestamp == datetime(2024, 1, 1, 12, 5)


def test_heartbeat_no_overwrite():
    """Heartbeat stops before we step outside context manager"""
    tracker = UploadTracker()
    source = source_factory().source_config

    upload_id = tracker.add_source(source)

    # Wait for one heartbeat to succeed
    with heartbeat(tracker, upload_id, interval_seconds=0.1):
        wait_for_heartbeat(tracker, upload_id)

    tracker.update_status(upload_id=upload_id, status="complete")
    assert tracker.get(upload_id=upload_id).status.status == "complete"


@patch("matchbox.server.api.uploads.datetime")
def test_heartbeat_with_expiry(mock_datetime: Mock):
    """Test heartbeat behavior with cache expiry."""
    tracker = UploadTracker(expiry_minutes=30)
    source = source_factory().source_config

    # Create entry
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    upload_id = tracker.add_source(source)

    # Start heartbeat and let it update
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 25)
    with heartbeat(tracker, upload_id, interval_seconds=0.1):
        wait_for_heartbeat(tracker, upload_id)

        # Entry should still exist and have updated timestamp
        entry = tracker.get(upload_id)
        assert entry is not None
        assert entry.update_timestamp == datetime(2024, 1, 1, 12, 25)

    # Move past expiry time
    mock_datetime.now.return_value = datetime(2024, 1, 1, 13, 0)
    assert tracker.get(upload_id) is None


def test_multiple_heartbeats():
    """Test multiple concurrent heartbeats on different entries."""
    tracker = UploadTracker()
    source = source_factory().source_config

    # Create two entries
    id1 = tracker.add_source(source)
    id2 = tracker.add_source(source)

    def run_heartbeat(upload_id):
        with heartbeat(tracker, upload_id, interval_seconds=0.1):
            wait_for_heartbeat(tracker, upload_id)

    # Run heartbeats concurrently
    with ThreadPoolExecutor() as executor:
        executor.submit(run_heartbeat, id1)
        executor.submit(run_heartbeat, id2)

    # Verify both entries were updated
    entry1 = tracker.get(id1)
    entry2 = tracker.get(id2)
    assert entry1.status.status == "processing"
    assert entry2.status.status == "processing"
    assert "Last heartbeat:" in entry1.status.details
    assert "Last heartbeat:" in entry2.status.details
