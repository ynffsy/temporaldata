import pytest
import os
import h5py
import numpy as np
import tempfile
from kirby.data.data import IrregularTimeSeries, Interval, Data


@pytest.fixture
def test_filepath(request):
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    filepath = tmpfile.name

    def finalizer():
        tmpfile.close()
        # clean up the temporary file after the test
        if os.path.exists(filepath):
            os.remove(filepath)

    request.addfinalizer(finalizer)
    return filepath


def test_save_to_hdf5(test_filepath):
    a = IrregularTimeSeries(np.array([0, 1, 2]), x=np.array([1, 2, 3]))

    with h5py.File(test_filepath, "w") as file:
        a.to_hdf5(file)

    b = Interval(start=np.array([0, 1, 2]), end=np.array([1, 2, 3]))

    with h5py.File(test_filepath, "w") as file:
        b.to_hdf5(file)

    c = Data(
        start=0,
        end=3,
        a_timeseries=a,
        b_intervals=b,
        x=np.array([0, 1, 2]),
        y=np.array([1, 2, 3]),
        z=np.array([2, 3, 4]),
    )

    with h5py.File(test_filepath, "w") as file:
        c.to_hdf5(file)


def test_load_from_h5(test_filepath):
    # create a file and save it
    a = IrregularTimeSeries(np.array([0, 1, 2]), x=np.array([1, 2, 3]))
    with h5py.File(test_filepath, "w") as file:
        a.to_hdf5(file)

    del a

    # load it again
    with h5py.File(test_filepath, "r") as file:
        a = IrregularTimeSeries.from_hdf5(file)

        assert np.all(a.timestamps[:] == np.array([0, 1, 2]))
        assert np.all(a.x[:] == np.array([1, 2, 3]))

    b = Interval(start=np.array([0, 1, 2]), end=np.array([1, 2, 3]))

    with h5py.File(test_filepath, "w") as file:
        b.to_hdf5(file)

    del b

    with h5py.File(test_filepath, "r") as file:
        b = Interval.from_hdf5(file)

        assert np.all(b.start[:] == np.array([0, 1, 2]))
        assert np.all(b.end[:] == np.array([1, 2, 3]))

    a = IrregularTimeSeries(np.array([0, 1, 2]), x=np.array([1, 2, 3]))
    b = Interval(start=np.array([0, 1, 2]), end=np.array([1, 2, 3]))
    c = Data(
        start=0,
        end=3,
        a_timeseries=a,
        b_intervals=b,
        x=np.array([0, 1, 2]),
        y=np.array([1, 2, 3]),
        z=np.array([2, 3, 4]),
    )

    with h5py.File(test_filepath, "w") as file:
        c.to_hdf5(file)

    del c

    with h5py.File(test_filepath, "r") as file:
        c = Data.from_hdf5(file)

        assert np.all(c.a_timeseries.timestamps[:] == np.array([0, 1, 2]))
        assert np.all(c.a_timeseries.x[:] == np.array([1, 2, 3]))
        assert np.all(c.b_intervals.start[:] == np.array([0, 1, 2]))
        assert np.all(c.b_intervals.end[:] == np.array([1, 2, 3]))
        assert np.all(c.x[:] == np.array([0, 1, 2]))
        assert np.all(c.y[:] == np.array([1, 2, 3]))
        assert np.all(c.z[:] == np.array([2, 3, 4]))


def test_lazy_slicing(test_filepath):
    # create a file and save it
    a = IrregularTimeSeries(
        np.array([0.1, 0.3, 1.0, 1.2, 2.4]), x=np.array([1, 2, 3, 4, 5])
    )
    with h5py.File(test_filepath, "w") as file:
        a.to_hdf5(file)

    del a

    # load it again
    with h5py.File(test_filepath, "r") as file:
        a = IrregularTimeSeries.from_hdf5(file)

        assert a._lazy, "object should be lazy"
        assert isinstance(a.timestamps, h5py.Dataset), "timestamps should be a dataset"

        b = a.slice(0.2, 1.15)
        assert not b._lazy, "object should not be lazy"
        assert isinstance(
            b.timestamps, np.ndarray
        ), "timestamps should be a numpy array"
        assert np.all(b.timestamps == np.array([0.3, 1.0]) - 0.2)
        assert np.all(b.x == np.array([2, 3]))

        c = a.slice(0.0, 0.8)
        assert np.allclose(c.timestamps, np.array([0.1, 0.3]))
        assert np.all(c.x == np.array([1, 2]))

        d = a.slice(0.0, 3.0)
        assert np.allclose(d.timestamps, np.array([0.1, 0.3, 1.0, 1.2, 2.4]))
        assert np.all(d.x == np.array([1, 2, 3, 4, 5]))
