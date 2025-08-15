from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from matchbox.common.arrow import table_to_buffer
from matchbox.common.exceptions import (
    MatchboxServerFileError,
)
from matchbox.common.factories.sources import source_factory
from matchbox.server.tasks.uploads import process_upload
from matchbox.server.uploads import InMemoryUploadTracker

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


def test_process_upload_deletes_file_on_failure(s3: S3Client):
    """Test that files are deleted from S3 even when processing fails.

    Other behaviours of this task are captured in the API integration tests for adding a
    source or a model.
    """
    # Setup
    tracker = InMemoryUploadTracker()
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.index = Mock(side_effect=ValueError("Simulated processing failure"))

    bucket = "test-bucket"
    test_key = "test-upload-id.parquet"

    s3.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Add parquet to S3 and verify
    source_testkit = source_factory()
    buffer = table_to_buffer(source_testkit.data_hashes)
    s3.put_object(Bucket=bucket, Key=test_key, Body=buffer)

    assert s3.head_object(Bucket=bucket, Key=test_key)

    # Setup metadata store with test data
    upload_id = tracker.add_source(source_testkit.source_config)
    tracker.update(upload_id, "awaiting_upload")

    # Run the process, expecting it to fail
    with pytest.raises(MatchboxServerFileError) as excinfo:
        process_upload(
            backend=mock_backend,
            tracker=tracker,
            s3_client=s3,
            upload_id=upload_id,
            bucket=bucket,
            filename=test_key,
            upload_type="type",
            resolution_name="name",
        )

    assert "Simulated processing failure" in str(excinfo.value)

    # Check that the status was updated to failed
    status = tracker.get(upload_id).status
    assert status.stage == "failed", f"Expected status 'failed', got '{status.stage}'"

    # Verify file was deleted despite the failure
    with pytest.raises(ClientError) as excinfo:
        s3.head_object(Bucket=bucket, Key=test_key)
    assert "404" in str(excinfo.value) or "NoSuchKey" in str(excinfo.value), (
        f"File was not deleted: {str(excinfo.value)}"
    )
