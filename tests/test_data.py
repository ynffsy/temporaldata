import pytest
import torch
from kirby.data.data import IrregularTimeSeries


def test_sortedness():
    a = IrregularTimeSeries(torch.Tensor([0, 1, 2]))
    assert a.sorted

    a.timestamps = torch.Tensor([0, 2, 1])
    with pytest.raises(AssertionError):
        a.slice(0, 1)

    with pytest.raises(AssertionError):
        a = IrregularTimeSeries(torch.Tensor([0, 1, 0]))
    
    a = IrregularTimeSeries(torch.Tensor([0, 1, 2]))
    #a.timestamps[2] = 0

    a.slice(0, 10)  # Does not raise any error, known bug