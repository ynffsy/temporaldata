import pytest
import torch
from kirby.data.data import IrregularTimeSeries


def test_sortedness():
    a = IrregularTimeSeries(torch.Tensor([0, 1, 2]))
    assert a.sorted

    a.timestamps = torch.Tensor([0, 2, 1])
    assert not a.sorted
    a = a.slice(0, 1)
    assert torch.all(a.timestamps == torch.Tensor([0, 1]))