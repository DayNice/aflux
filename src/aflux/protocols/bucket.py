import pathlib
from abc import abstractmethod
from typing import Protocol, runtime_checkable

from aflux.types.bucket import BucketFileMeta


@runtime_checkable
class Bucket(Protocol):
    @abstractmethod
    def check_file_exists(self, remote_path: str) -> bool: ...

    @abstractmethod
    def get_file_meta(self, remote_path: str) -> BucketFileMeta: ...

    @abstractmethod
    def get_file(self, remote_path: str, *, refresh: bool = False) -> pathlib.Path: ...

    @abstractmethod
    def get_bytes(self, remote_path: str, *, refresh: bool = False) -> bytes: ...

    @abstractmethod
    def put_file(self, local_file: str | pathlib.Path, remote_path: str) -> None: ...

    @abstractmethod
    def put_bytes(self, local_bytes: bytes, remote_path: str) -> None: ...

    @abstractmethod
    def delete_file(self, remote_path: str) -> None: ...
