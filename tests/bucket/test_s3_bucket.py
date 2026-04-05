import pathlib

import boto3
import pytest
from moto import mock_aws

from aflux.utils.bucket import S3Bucket


@pytest.fixture
def s3_client():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield client


class TestS3Bucket:
    def test_put_and_get(self, s3_client) -> None:
        bucket = S3Bucket("test-bucket", s3_client=s3_client)
        remote_path = "a/b/c/file.txt"
        data = b"hello s3"

        bucket.put_bytes(data, remote_path)
        assert bucket.check_file_exists(remote_path)
        assert bucket.get_bytes(remote_path) == data

    def test_missing_file_exists(self, s3_client) -> None:
        bucket = S3Bucket("test-bucket", s3_client=s3_client)
        assert not bucket.check_file_exists("missing.txt")

    def test_refresh_logic(self, s3_client) -> None:
        bucket = S3Bucket("test-bucket", s3_client=s3_client)
        remote_path = "data.txt"

        bucket.put_bytes(b"v1", remote_path)
        assert bucket.get_bytes(remote_path) == b"v1"

        s3_client.put_object(Bucket="test-bucket", Key="data.txt", Body=b"v2")

        assert bucket.get_bytes(remote_path, refresh=False) == b"v1"
        assert bucket.get_bytes(remote_path, refresh=True) == b"v2"

    def test_context_manager_cleanup(self, s3_client, tmp_path: pathlib.Path) -> None:
        with S3Bucket("test-bucket", temp_dir=tmp_path, s3_client=s3_client) as bucket:
            remote_path = "temp.txt"
            bucket.put_bytes(b"data", remote_path)
            local_file = bucket.get_file(remote_path)
            assert local_file.exists()

        assert tmp_path.exists()
        assert not any(tmp_path.iterdir())
