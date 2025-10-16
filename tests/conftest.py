import random
from shutil import rmtree
from pathlib import Path

import pytest
import numpy as np


PROCESSED_DIR_PATH = "processed"


@pytest.fixture(autouse=True)
def deterministicity():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)


@pytest.fixture(scope='session', autouse=True)
def cleanup_basetemp(request):
    """Clean up the basetemp directory after all tests complete."""
    yield
    # Clean up basetemp if it was explicitly set via --basetemp flag
    basetemp = request.config.option.basetemp
    if basetemp:
        basetemp_path = Path(basetemp)
        if basetemp_path.exists():
            rmtree(basetemp_path, ignore_errors=True)


@pytest.fixture(scope='module')
def data_dir_path(tmp_path_factory):
    data_path = tmp_path_factory.mktemp('data')
    yield data_path
    if data_path.exists():
        rmtree(data_path)


@pytest.fixture(scope='module')
def processed_dir_path(data_dir_path):
    processed_dir = data_dir_path / PROCESSED_DIR_PATH
    processed_dir.mkdir()
    yield processed_dir
    if processed_dir.exists():
        rmtree(processed_dir)
