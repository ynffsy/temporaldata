import pytest
import numpy as np
from kirby.data.data import IrregularTimeSeries


def test_sortedness():
    a = IrregularTimeSeries(np.array([0, 1, 2]))
    assert a.sorted

    a.timestamps = np.array([0, 2, 1])
    assert not a.sorted
    a = a.slice(0, 1)
    assert np.allclose(a.timestamps, np.array([0, 1]))
