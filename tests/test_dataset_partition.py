import numpy as np
import pytest

from fdit.dataset import stratified_split
from fdit.partition import partition, statistics
from fdit.config import fingerprint


def test_splits_capacity_and_skew():
    labels = np.repeat(np.arange(10), 270)
    splits = stratified_split(labels)
    assert list(map(len, splits.values())) == [2160, 270, 270]
    assert len(set(sum(splits.values(), []))) == len(labels)
    iid = partition(splits["train"], labels, "iid")
    skew = partition(splits["train"], labels, "non_iid")
    assert [len(c) for c in iid] == [720] * 3
    assert [len(c) for c in skew] == [720] * 3
    assert sorted(sum(skew, [])) == splits["train"]
    assert statistics(skew, labels)["mean_js"] > statistics(iid, labels)["mean_js"] + .05
    assert partition(splits["train"], labels, "non_iid") == skew
    assert stratified_split(labels) == splits
    assert fingerprint(splits) != fingerprint(stratified_split(labels, 1))


def test_small_remainder_and_bad_mode():
    labels = np.arange(101) % 10
    clients = partition(list(range(101)), labels, "non_iid")
    assert sorted(map(len, clients)) == [33, 34, 34]
    with pytest.raises(ValueError):
        partition(list(range(101)), labels, "invalid")
