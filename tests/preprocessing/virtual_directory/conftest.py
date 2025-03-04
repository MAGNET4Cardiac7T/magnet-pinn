import pytest


@pytest.fixture(scope="module")
def tmp_dirs(tmp_path_factory):
    dir1 = tmp_path_factory.mktemp("dir1")
    dir2 = tmp_path_factory.mktemp("dir2")
    (dir1 / "file1.txt").write_text("content1")
    (dir2 / "file2.txt").write_text("content2")
    return [dir1, dir2]
