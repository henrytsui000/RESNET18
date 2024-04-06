import pytest
import sys

sys.path.append("./")
from model import load_model
from dataset import load_dataset

@pytest.mark.parametrize("model_name, args", [("normal", ()), ("modified", ()), ("invalid", ())])
def test_load_model(model_name, args):
    if model_name == "invalid":
        with pytest.raises(ValueError):
            load_model(model_name, *args)
    else:
        assert load_model(model_name, *args) is not None


@pytest.mark.parametrize("dataset_name", ["cifar10", "cifar100"])
def test_load_dataset(dataset_name):
    data, nc = load_dataset(dataset_name)
    assert data is not None
    assert isinstance(nc, int)
