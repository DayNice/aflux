import pathlib

import pytest

from aflux.utils.bucket import DirBucket


class TestDirBucket:
    def test_path_traversal_prevention(self, tmp_path: pathlib.Path) -> None:
        bucket = DirBucket(tmp_path)
        with pytest.raises(ValueError, match="escapes"):
            bucket.get_file("../outside.txt")
        with pytest.raises(ValueError, match="escapes"):
            bucket.put_bytes(b"data", "../outside.txt")

    def test_put_and_get(self, tmp_path: pathlib.Path) -> None:
        bucket = DirBucket(tmp_path)
        remote_path = "a/b/c/file.txt"
        data = b"hello world"

        bucket.put_bytes(data, remote_path)
        assert bucket.check_file_exists(remote_path)
        assert bucket.get_bytes(remote_path) == data

        local_file = bucket.get_file(remote_path)
        assert local_file.read_bytes() == data

    def test_delete_and_cleanup(self, tmp_path: pathlib.Path) -> None:
        bucket = DirBucket(tmp_path)
        path1 = "nested/dir/file1.txt"
        path2 = "nested/file2.txt"

        bucket.put_bytes(b"data", path1)
        bucket.put_bytes(b"data", path2)

        bucket.delete_file(path1)

        assert not bucket.check_file_exists(path1)
        assert not (tmp_path / "nested" / "dir").exists()
        assert (tmp_path / "nested").exists()
        assert bucket.check_file_exists(path2)
