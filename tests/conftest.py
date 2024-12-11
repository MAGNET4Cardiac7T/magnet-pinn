from shutil import rmtree

import pytest

@pytest.fixture(scope='session')
def data_dir_path(tmp_path_factory):
    data_path = tmp_path_factory.mktemp('data')
    yield data_path
    if data_path.exists():
        rmtree(data_path)
