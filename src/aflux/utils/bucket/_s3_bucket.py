import io
import pathlib
import shutil
import tempfile
import weakref
from types import TracebackType
from typing import TYPE_CHECKING, Self, override

import boto3
import botocore.exceptions

from aflux.protocols.bucket import Bucket
from aflux.types.bucket import BucketFileMeta

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client
else:
    S3Client = object


class S3Bucket(Bucket):
    def __init__(
        self,
        bucket_name: str,
        bucket_prefix: str = "",
        *,
        temp_dir: str | pathlib.Path | None = None,
        s3_client: S3Client | None = None,
    ):
        self._bucket_name = bucket_name
        self._bucket_prefix = bucket_prefix

        if temp_dir is not None:
            self._temp_dir = pathlib.Path(temp_dir).resolve()
            self._temp_dir_finalizer = None
        else:
            self._temp_dir = pathlib.Path(tempfile.mkdtemp()).resolve()
            self._temp_dir_finalizer = weakref.finalize(self, shutil.rmtree, self._temp_dir, ignore_errors=True)

        if s3_client is None:
            s3_client = boto3.client("s3")
        self._s3_client = s3_client

    def _get_remote_file(self, remote_path: str) -> pathlib.Path:
        remote_file = (self._temp_dir / remote_path).resolve()
        if not remote_file.is_relative_to(self._temp_dir):
            msg = f"Remote path escapes temp directory: {remote_path!r}"
            raise ValueError(msg)
        return remote_file

    def _get_bucket_key(self, remote_path: str) -> str:
        return f"{self._bucket_prefix}{remote_path}"

    @override
    def check_file_exists(self, remote_path: str) -> bool:
        bucket_key = self._get_bucket_key(remote_path)
        try:
            self._s3_client.head_object(Bucket=self._bucket_name, Key=bucket_key)
            return True
        except botocore.exceptions.ClientError as e:
            if int(e.response["Error"]["Code"]) != 404:
                raise
            return False

    @override
    def get_file_meta(self, remote_path: str) -> BucketFileMeta:
        bucket_key = self._get_bucket_key(remote_path)
        resp = self._s3_client.head_object(Bucket=self._bucket_name, Key=bucket_key)
        last_modified = resp["LastModified"]
        size = resp["ContentLength"]
        return BucketFileMeta(path=remote_path, size=size, last_modified=last_modified)

    @override
    def get_file(self, remote_path: str, *, refresh: bool = False) -> pathlib.Path:
        remote_file = self._get_remote_file(remote_path)
        if not refresh and remote_file.exists():
            return remote_file
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        bucket_key = self._get_bucket_key(remote_path)
        self._s3_client.download_file(self._bucket_name, bucket_key, str(remote_file))
        return remote_file

    @override
    def get_bytes(self, remote_path: str, *, refresh: bool = False) -> bytes:
        if refresh:
            bucket_key = self._get_bucket_key(remote_path)
            buffer = io.BytesIO()
            self._s3_client.download_fileobj(self._bucket_name, bucket_key, buffer)
            return buffer.getvalue()
        local_file = self.get_file(remote_path, refresh=refresh)
        return local_file.read_bytes()

    @override
    def put_file(self, local_file: str | pathlib.Path, remote_path: str) -> None:
        bucket_key = self._get_bucket_key(remote_path)
        self._s3_client.upload_file(str(local_file), self._bucket_name, bucket_key)

    @override
    def put_bytes(self, local_bytes: bytes, remote_path: str) -> None:
        bucket_key = self._get_bucket_key(remote_path)
        self._s3_client.upload_fileobj(io.BytesIO(local_bytes), self._bucket_name, bucket_key)

    @override
    def delete_file(self, remote_path: str) -> None:
        bucket_key = self._get_bucket_key(remote_path)
        self._s3_client.delete_object(Bucket=self._bucket_name, Key=bucket_key)

    def clear_temp_dir(self) -> None:
        if not self._temp_dir.exists():
            return
        for path in self._temp_dir.iterdir():
            if path.is_file():
                path.unlink()
                continue
            shutil.rmtree(path, ignore_errors=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.clear_temp_dir()
