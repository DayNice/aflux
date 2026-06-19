import shutil
import tempfile
import weakref
from pathlib import Path
from types import TracebackType
from typing import Self

from aflux import utils


class FileAllocator:
    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            self._path = Path(tempfile.mkdtemp()).resolve()
            self._path_finalizer = weakref.finalize(self, shutil.rmtree, self._path, ignore_errors=True)
        else:
            self._path = Path(path).resolve()
            self._path_finalizer = None

    @property
    def path(self) -> Path:
        return self._path

    def allocate(self, suffix_like: str | Path = "") -> Path:
        suffix = "".join(Path(suffix_like).suffixes)
        name = f"{utils.get_uuid_v7().hex}{suffix}"
        return (self._path / name).resolve()

    def clear(self) -> None:
        if not self._path.exists():
            return
        for item in self._path.iterdir():
            if item.is_file():
                item.unlink()
                continue
            shutil.rmtree(item, ignore_errors=True)

    def make_child(self) -> "FileAllocator":
        return FileAllocator(tempfile.mkdtemp(dir=self._path))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.clear()
