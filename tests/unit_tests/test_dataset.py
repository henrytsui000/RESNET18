import pytest
import sys

sys.path.append("./")
from dataset import load_dataset

@pytest.mark.parametrize("dataset_name", ["cifar10", "cifar100"])
def test_load_dataset(dataset_name):
    data, nc = load_dataset(dataset_name)
    assert data is not None
    assert isinstance(nc, int)
