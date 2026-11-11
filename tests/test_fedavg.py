import pytest
import torch
from fdit.federated import fedavg


def test_weighted_fedavg():
    result = fedavg([{"a": torch.tensor([1., 3.])}, {"a": torch.tensor([5., 7.])}], [1, 3])
    torch.testing.assert_close(result["a"], torch.tensor([4., 6.]))
    with pytest.raises(ValueError):
        fedavg([{"a": torch.ones(2)}, {"a": torch.ones(3)}], [1, 1])
    with pytest.raises(ValueError):
        fedavg([{"a": torch.ones(2)}, {"b": torch.ones(2)}], [1, 1])
