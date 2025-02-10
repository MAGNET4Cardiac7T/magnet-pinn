from pathlib import Path
from magnet_pinn.preprocessing.virtual_directory import VirtualDirectory


def test_virtual_directory_exists(tmp_dirs):
    vdir = VirtualDirectory(tmp_dirs)
    assert vdir.exists() is True

def test_virtual_directory_is_dir(tmp_dirs):
    vdir = VirtualDirectory(tmp_dirs)
    assert vdir.is_dir() is True

def test_virtual_directory_is_file(tmp_dirs):
    vdir = VirtualDirectory(tmp_dirs)
    assert vdir.is_file() is False

def test_virtual_directory_iterdir(tmp_dirs):
    vdir = VirtualDirectory(tmp_dirs)
    files = list(vdir.iterdir())
    assert len(files) == 2
    assert files[0].name == "file1.txt"
    assert files[1].name == "file2.txt"

def test_virtual_directory_str(tmp_dirs):
    vdir = VirtualDirectory(tmp_dirs)
    assert str(vdir) == f"VirtualDirectory({tmp_dirs[0]}, {tmp_dirs[1]})"

def test_virtual_directory_truediv(tmp_dirs):
    vdir = VirtualDirectory(tmp_dirs)
    assert (vdir / "file1.txt") == tmp_dirs[0] / "file1.txt"
    assert (vdir / "file2.txt") == tmp_dirs[1] / "file2.txt"
    