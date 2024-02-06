import pytest
import numpy as np
import pandas as pd
from kirby.data.data import IrregularTimeSeries, ArrayDict


def test_sortedness():
    a = IrregularTimeSeries(np.array([0, 1, 2]))
    assert a.sorted

    a.timestamps = np.array([0, 2, 1])
    assert not a.sorted
    a = a.slice(0, 1)
    assert np.allclose(a.timestamps, np.array([0, 1]))

def test_from_dataframe():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'col1': np.array([1, 2, 3]), # ndarray
        'col2': [np.array(4), np.array(5), np.array(6)], # list of ndarrays
        'col3': ['a', 'b', 'c'] # list of strings
    })

    # Call the from_dataframe method
    # Test the `allow_string_ndarray=True`
    a = ArrayDict.from_dataframe(df)

    # Assert the correctness of the conversion
    assert np.array_equal(a.col1, np.array([1, 2, 3]))
    assert np.array_equal(a.col2, np.array([4, 5, 6]))
    assert np.array_equal(a.col3, np.array(['a', 'b', 'c']))

    b = ArrayDict.from_dataframe(df, allow_string_ndarray=False)
    # strings are converted to ndarrays when `allow_string_ndarray=True`

    assert not hasattr(b, 'col3') # col3 is not present in the ArrayDict
    # Test unsigned_to_long parameter
    df_unsigned = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    }, dtype=np.uint32)

    a_unsigned = ArrayDict.from_dataframe(df_unsigned, unsigned_to_long=False)

    assert np.array_equal(a_unsigned.col1, np.array([1, 2, 3], dtype=np.int32))
    assert np.array_equal(a_unsigned.col2, np.array([4, 5, 6], dtype=np.int32))

    df_non_ascii = pd.DataFrame({
        'col1': ['Ä', 'é', 'é'], # not ASCII, should catch thsi and not convert to ndarray
        'col2': ['d', 'e', 'f'] # should be converted to fixed length ASCII "S" type ndarray
    })

    a_with_non_ascii_col = ArrayDict.from_dataframe(df_non_ascii)

    assert hasattr(a_with_non_ascii_col, 'col2')
    assert not hasattr(a_with_non_ascii_col, 'col1')
