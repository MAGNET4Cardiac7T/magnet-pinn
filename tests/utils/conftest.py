import pytest
import torch

@pytest.fixture
def random_iterator():
    def _iterator(seed=42, num_batches=10, batch_size=5, num_features=3):
        torch.manual_seed(seed)
        for _ in range(num_batches):
            yield {"input": torch.randn(batch_size, num_features)}
    return _iterator

@pytest.fixture
def random_batch(random_iterator):
    iterator = random_iterator(seed=42, num_batches=1, batch_size=10, num_features=3)
    return next(iterator)["input"]
