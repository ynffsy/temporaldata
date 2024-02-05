import pytest
import numpy as np
from kirby.data import Interval


def test_indexing():
    # same code but with numpy arrays
    interval = Interval(start=np.array([0, 1, 2]), end=np.array([1, 2, 3]))

    # Test single index
    result = interval[0]
    expected = Interval(np.array([0]), np.array([1]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    # Test slice indexing
    result = interval[0:2]
    expected = Interval(np.array([0, 1]), np.array([1, 2]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    # Test list indexing
    result = interval[[0, 2]]
    expected = Interval(np.array([0, 2]), np.array([1, 3]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )

    # Test boolean indexing
    result = interval[[True, False, True]]
    expected = Interval(np.array([0, 2]), np.array([1, 3]))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )


def test_linspace():
    result = Interval.linspace(0, 1, 10)
    expected = Interval(np.arange(0, 1.0, 0.1), np.arange(0.1, 1.1, 0.1))
    assert np.allclose(result.start, expected.start) and np.allclose(
        result.end, expected.end
    )


def test_split():
    interval = Interval.linspace(0, 1, 10)

    # split into 3 sets using an int list
    result = interval.split([6, 2, 2])
    expected = [
        Interval.linspace(0, 0.6, 6),
        Interval.linspace(0.6, 0.8, 2),
        Interval.linspace(0.8, 1, 2),
    ]
    assert len(result) == len(expected)
    for i in range(len(result)):
        assert np.allclose(result[i].start, expected[i].start) and np.allclose(
            result[i].end, expected[i].end
        )

    # split into 2 sets using a float list
    result = interval.split([0.8, 0.2])
    expected = [Interval.linspace(0, 0.8, 8), Interval.linspace(0.8, 1, 2)]
    assert len(result) == len(expected)
    for i in range(len(result)):
        assert np.allclose(result[i].start, expected[i].start) and np.allclose(
            result[i].end, expected[i].end
        )

    # shuffle
    result = interval.split([0.5, 0.5], shuffle=True, random_seed=42)
    print(result[0].start, result[1].start)
    print(result[0].end, result[1].end)
    expected = [
        Interval(
            start=np.array([0.5000, 0.6000, 0.0000, 0.7000, 0.3000]),
            end=np.array([0.6000, 0.7000, 0.1000, 0.8000, 0.4000]),
        ),
        Interval(
            start=np.array([0.2000, 0.4000, 0.9000, 0.1000, 0.8000]),
            end=np.array([0.3000, 0.5000, 1.0000, 0.2000, 0.9000]),
        ),
    ]
    assert len(result) == len(expected)
    for i in range(len(result)):
        assert np.allclose(result[i].start, expected[i].start) and np.allclose(
            result[i].end, expected[i].end
        ), (
            f"result: {result[i].start} {result[i].end} "
            f"expected: {expected[i].start} {expected[i].end}"
        )
