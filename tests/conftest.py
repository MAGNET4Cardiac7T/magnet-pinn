from shutil import rmtree

import pytest


PROCESSED_DIR_PATH = "processed"


@pytest.fixture(scope='module')
def data_dir_path(tmp_path_factory):
    data_path = tmp_path_factory.mktemp('data')
    yield data_path
    if data_path.exists():
        """rmtree(data_path)"""


@pytest.fixture(scope='module')
def processed_dir_path(data_dir_path):
    processed_dir = data_dir_path / PROCESSED_DIR_PATH
    processed_dir.mkdir()
    yield processed_dir
    if processed_dir.exists():
        """rmtree(processed_dir)"""
