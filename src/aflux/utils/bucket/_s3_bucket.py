import io
import threading
from collections.abc import Iterator
from concurrent.futures import Future
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self, override

import boto3
import botocore.exceptions

from aflux.protocols.bucket import Bucket
from aflux.types.bucket import BucketFileMeta

from ._file_allocator import FileAllocator

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
        temp_dir: str | Path | FileAllocator | None = None,
        s3_client: S3Client | None = None,
    ):
        self._bucket_name = bucket_name
        self._bucket_prefix = bucket_prefix
        self._allocator = temp_dir if isinstance(temp_dir, FileAllocator) else FileAllocator(temp_dir)

        if s3_client is None:
            s3_client = boto3.client("s3")
        self._s3_client = s3_client

        self._registry_lock = threading.Lock()
        self._active_download_map: dict[str, Future[Path]] = {}

    def _get_bucket_path(self, remote_path: str) -> str:
        return f"{self._bucket_prefix}{remote_path}"

    @override
    def check_file_exists(self, remote_path: str) -> bool:
        bucket_key = self._get_bucket_path(remote_path)
        try:
            self._s3_client.head_object(Bucket=self._bucket_name, Key=bucket_key)
            return True
        except botocore.exceptions.ClientError as e:
            if int(e.response["Error"]["Code"]) != 404:
                raise
            return False

    @override
    def get_file_meta(self, remote_path: str) -> BucketFileMeta:
        bucket_key = self._get_bucket_path(remote_path)
        resp = self._s3_client.head_object(Bucket=self._bucket_name, Key=bucket_key)
        last_modified = resp["LastModified"]
        size = resp["ContentLength"]
        return BucketFileMeta(path=remote_path, size=size, last_modified=last_modified)

    @override
    def get_file_metas(self, remote_prefix: str = "") -> Iterator[BucketFileMeta]:
        bucket_prefix = self._get_bucket_path(remote_prefix)
        paginator = self._s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket_name, Prefix=bucket_prefix):
            for obj in page.get("Contents", []):
                file_meta = BucketFileMeta(
                    path=obj["Key"].removeprefix(self._bucket_prefix),
                    size=obj["Size"],
                    last_modified=obj["LastModified"],
                )
                yield file_meta

    @override
    def get_file(self, remote_path: str) -> Path:
        temp_file = self._allocator.allocate(remote_path)
        with self._registry_lock:
            future = Future()
            self._active_download_map[temp_file.name] = future

        try:
            bucket_key = self._get_bucket_path(remote_path)
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            self._s3_client.download_file(self._bucket_name, bucket_key, str(temp_file))

            future.set_result(temp_file)
            return temp_file
        except BaseException as e:
            future.set_exception(e)
            raise
        finally:
            with self._registry_lock:
                self._active_download_map.pop(temp_file.name, None)

    @override
    def get_bytes(self, remote_path: str) -> bytes:
        bucket_key = self._get_bucket_path(remote_path)
        buffer = io.BytesIO()
        self._s3_client.download_fileobj(self._bucket_name, bucket_key, buffer)
        return buffer.getvalue()

    @override
    def put_file(self, local_file: str | Path, remote_path: str) -> None:
        bucket_key = self._get_bucket_path(remote_path)
        self._s3_client.upload_file(str(local_file), self._bucket_name, bucket_key)

    @override
    def put_bytes(self, local_bytes: bytes, remote_path: str) -> None:
        bucket_key = self._get_bucket_path(remote_path)
        self._s3_client.upload_fileobj(io.BytesIO(local_bytes), self._bucket_name, bucket_key)

    @override
    def delete_file(self, remote_path: str) -> None:
        bucket_key = self._get_bucket_path(remote_path)
        self._s3_client.delete_object(Bucket=self._bucket_name, Key=bucket_key)

    @override
    def with_prefix(self, remote_prefix: str) -> "S3Bucket":
        return S3Bucket(
            self._bucket_name,
            self._get_bucket_path(remote_prefix),
            temp_dir=self._allocator.make_child(),
            s3_client=self._s3_client,
        )

    def clear_temp_dir(self) -> None:
        self._allocator.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._allocator.clear()
