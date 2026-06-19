import datetime
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Self, override

from aflux.protocols.bucket import Bucket
from aflux.types.bucket import BucketFileMeta

from ._file_allocator import FileAllocator


class DirBucket(Bucket):
    def __init__(
        self,
        root_dir: str | Path,
        *,
        temp_dir: str | Path | FileAllocator | None = None,
    ):
        self._root_dir = Path(root_dir).resolve()
        self._allocator = temp_dir if isinstance(temp_dir, FileAllocator) else FileAllocator(temp_dir)

    def _get_remote_file(self, remote_path: str) -> Path:
        remote_file = (self._root_dir / remote_path).resolve()
        if not remote_file.is_relative_to(self._root_dir):
            msg = f"Remote path escapes root directory: {remote_path!r}"
            raise ValueError(msg)
        return remote_file

    def _validate_remote_file(self, remote_path: str) -> Path:
        remote_file = self._get_remote_file(remote_path)
        if not remote_file.is_file():
            msg = f"Remote file does not exist: {remote_path!r}"
            raise ValueError(msg)
        return remote_file

    @override
    def check_file_exists(self, remote_path: str) -> bool:
        remote_file = self._get_remote_file(remote_path)
        return remote_file.is_file()

    @override
    def get_file_meta(self, remote_path: str) -> BucketFileMeta:
        remote_file = self._validate_remote_file(remote_path)
        file_stat = remote_file.stat()
        size = file_stat.st_size
        last_modified = datetime.datetime.fromtimestamp(file_stat.st_mtime, datetime.UTC)
        return BucketFileMeta(path=remote_path, size=size, last_modified=last_modified)

    @override
    def get_file_metas(self, remote_prefix: str = "") -> Iterator[BucketFileMeta]:
        search_dir = self._get_remote_file(remote_prefix)
        if search_dir != self._root_dir and not search_dir.is_dir():
            search_dir = search_dir.parent
        if not search_dir.exists():
            return

        file_metas: list[BucketFileMeta] = []
        for remote_file in search_dir.rglob("*"):
            remote_path = remote_file.relative_to(self._root_dir).as_posix()
            if not remote_file.is_file() or not remote_path.startswith(remote_prefix):
                continue

            file_stat = remote_file.stat()
            size = file_stat.st_size
            last_modified = datetime.datetime.fromtimestamp(file_stat.st_mtime, datetime.UTC)

            file_meta = BucketFileMeta(path=remote_path, size=size, last_modified=last_modified)
            file_metas.append(file_meta)
        file_metas.sort(key=lambda el: el.path)

        yield from file_metas

    @override
    def get_file(self, remote_path: str, *, refresh: bool = False) -> Path:
        remote_file = self._validate_remote_file(remote_path)
        temp_file = self._allocator.allocate(remote_path)
        shutil.copy(remote_file, temp_file)
        return temp_file

    @override
    def get_bytes(self, remote_path: str, *, refresh: bool = False) -> bytes:
        return self._validate_remote_file(remote_path).read_bytes()

    @override
    def put_file(self, local_file: str | Path, remote_path: str) -> None:
        remote_file = self._get_remote_file(remote_path)
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_file, remote_file)

    @override
    def put_bytes(self, local_bytes: bytes, remote_path: str) -> None:
        remote_file = self._get_remote_file(remote_path)
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        remote_file.write_bytes(local_bytes)

    @override
    def delete_file(self, remote_path: str) -> None:
        remote_file = self._get_remote_file(remote_path)
        if not remote_file.exists():
            return
        remote_file.unlink()

        # remove empty directories between file and root directory
        parent_dir = remote_file.parent
        assert parent_dir.is_relative_to(self._root_dir), "Remote file should be within root directory."
        while parent_dir != self._root_dir:
            if any(parent_dir.iterdir()):
                break
            parent_dir.rmdir()
            parent_dir = parent_dir.parent

    @override
    def with_prefix(self, remote_prefix: str) -> "DirBucket":
        return DirBucket(self._root_dir / remote_prefix, temp_dir=self._allocator.make_child())

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
