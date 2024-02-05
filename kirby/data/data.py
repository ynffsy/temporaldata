"""Inspired from PyG's data class."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    NamedTuple,
    Optional,
    Union,
)
import logging

import h5py
import numpy as np
import pandas as pd
import torch


from kirby.taxonomy import Dictable, RecordingTech, StringIntEnum


class ArrayDict(object):
    r"""A dictionary of arrays that share the same first dimension.

    .. note::
        Private attributes (starting with an underscore) do not need to be arrays, or
        have the same first dimension as the other attributes.
    """

    @property
    def keys(self) -> List[str]:
        r"""Returns a list of all attribute names."""
        return [x for x in self.__dict__.keys() if not x.startswith('_')]

    def _maybe_first_dim(self):
        r"""If `ArrayDict` has at least one attribute, returns the first dimension of
        the first attribute. Otherwise, returns :obj:`None`."""
        if len(self.keys) == 0:
            return None
        else:
            return self.__dict__[self.keys[0]].shape[0]

    def __setattr__(self, name, value):
        if not name.startswith("_"):
            # only ndarrays are accepted
            assert isinstance(value, np.ndarray) or isinstance(
                value, h5py.Dataset
            ), f"{name} must be a numpy array, got object of type {type(value)}"

            first_dim = self._maybe_first_dim()
            if first_dim is not None and value.shape[0] != first_dim:
                raise ValueError(
                    f"All elements of {self.__class__.__name__} must have the same "
                    f"first dimension. The first dimension of {name} is "
                    f"{value.shape[0]} but must be {first_dim}."
                )
        super(ArrayDict, self).__setattr__(name, value)

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the data."""
        return key in self.keys

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        return out

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [size_repr(k, getattr(self, k), indent=2) for k in self.keys]
        info = ",\n".join(info)
        return f"{cls}(\n{info}\n)"


class IrregularTimeSeries(ArrayDict):
    r"""An irregular time series is defined by a set of timestamps and a set of
    attributes that must share the same first dimension as the timestamps.
    The timestamps are not necessarily regularly sampled.

    Args:
        timestamps: an array of timestamps of shape (N,).
        timekeys: a list of strings that specify which attributes are time-based
            attributes.
        **kwargs: Arbitrary keyword arguments where the values are arbitrary
            multi-dimensional (2d, 3d, ..., nd) arrays with shape (N, *).
    """
    _lazy = False
    _sorted = None

    def __init__(
        self,
        timestamps: Union[np.ndarray, h5py.Dataset],
        *,
        timekeys: List[str] = ["timestamps"],
        **kwargs: Dict[str, Union[np.ndarray, h5py.Dataset]],
    ):
        super().__init__()

        self.timestamps = timestamps

        for key, value in kwargs.items():
            setattr(self, key, value)

        if "timestamps" not in timekeys:
            timekeys.append("timestamps")

        for key in timekeys:
            assert key in self.keys, f"Time attribute {key} does not exist."

        self._timekeys = timekeys

    def __setattr__(self, name, value):
        super(IrregularTimeSeries, self).__setattr__(name, value)
        if name == "timestamps":
            # timestamps has been updated, we no longer know whether it is sorted or not
            self._sorted = None
            self._start = None
            self._end = None

    @property
    def sorted(self):
        # check if we already know that the sequence is sorted
        # if lazy loading, we'll have to skip this check
        if self._sorted is None and not self._lazy:
            self._sorted = np.all(self.timestamps[1:] >= self.timestamps[:-1])
        return self._sorted

    def __len__(self) -> int:
        r"""Returns the number of time points."""
        return self.timestamps.shape[0]

    @property
    def start(self) -> float:
        if self._lazy and self._start is None:
            raise ValueError("Cannot compute start time of lazy time series.")

        if self._start is None:
            if self.sorted:
                self._start = self.timestamps[0]
            else:
                self._start = np.min(self.timestamps)

        return self._start

    @property
    def end(self) -> float:
        if self._lazy and self._end is None:
            raise ValueError("Cannot compute end time of lazy time series.")

        if self._end is None:
            if self.sorted:
                self._end = self.timestamps[-1]
            else:
                self._end = np.max(self.timestamps)

        return self._end

    def sort(self):
        r"""Sorts the timestamps."""
        if self._lazy:
            raise ValueError("Cannot sort lazy time series.")

        if not self.sorted:
            sorted_indices = np.argsort(self.timestamps)
            for key in self.keys:
                self.__dict__[key] = self.__dict__[key][sorted_indices]
        self._sorted = True

    def slice(self, start: float, end: float, request_keys: Optional[List[str]] = None):
        if request_keys is None:
            request_keys = self.keys

        assert "timestamps" in request_keys

        if not self._lazy:
            # the data is in memory
            if not self.sorted:
                self.sort()

            idx_l = np.searchsorted(self.timestamps, start)
            idx_r = np.searchsorted(self.timestamps, end, side="right")

            out = self.__class__.__new__(self.__class__)
            out._timekeys = self._timekeys

            for key in self.keys:
                if key in request_keys:
                    out.__dict__[key] = self.__dict__[key][idx_l:idx_r].copy()

            for key in self._timekeys:
                if key in request_keys:
                    out.__dict__[key] = out.__dict__[key] - start
            return out
        else:
            # lazy loading, we will use the precomputed grid to extract the chunk of
            # data that we need
            start_closest_sec_idx = np.clip(
                np.floor(start - self.start).astype(int),
                0,
                len(self._timestamp_indices_1s) - 1,
            )
            end_closest_sec_idx = np.clip(
                np.ceil(end - self.start).astype(int),
                0,
                len(self._timestamp_indices_1s) - 1,
            )

            idx_l = self._timestamp_indices_1s[start_closest_sec_idx]
            idx_r = self._timestamp_indices_1s[end_closest_sec_idx]

            out = self.__class__.__new__(self.__class__)
            out._timekeys = self._timekeys
            out._sorted = True

            for key in self.keys:
                if key in request_keys:
                    out.__dict__[key] = self.__dict__[key][idx_l:idx_r]

            # the slice we get is only precise to the 1sec, so we re-slice
            return out.slice(start, end)

    def to_hdf5(self, file):
        if not self.sorted:
            logging.warn("time series is not sorted, sorting before saving to h5")
            self.sort()

        for key in self.keys:
            value = getattr(self, key)
            file.create_dataset(key, data=value)

        # make sure to save the start and end times
        file.attrs["start"] = self.start
        file.attrs["end"] = self.end

        file.attrs["timekeys"] = np.array(self._timekeys, dtype="S")

        # timestamps is special
        grid_timestamps = np.arange(
            self.start, self.end + 1.0, 1.0
        )  # 1 second resolution
        file.create_dataset(
            "timestamp_indices_1s",
            data=np.searchsorted(self.timestamps, grid_timestamps),
        )

        file.attrs["object"] = self.__class__.__name__

    @classmethod
    def from_hdf5(cls, file):
        assert file.attrs["object"] == cls.__name__, "object type mismatch"
        data = {}
        for key, value in file.items():
            if key == "timestamp_indices_1s":
                data["_timestamp_indices_1s"] = value[:]
            else:
                data[key] = value

        obj = cls(**data)
        obj._lazy = True

        # recover start and end times
        obj._start = file.attrs["start"]
        obj._end = file.attrs["end"]
        obj._sorted = True
        obj._timekeys = file.attrs["timekeys"].astype(str).tolist()

        return obj


class RegularTimeSeries(IrregularTimeSeries):
    """A regular time series is the same as a regular time series, but it has a
    regular sampling rate.
    """

    @property
    def sampling_rate(self):
        return 1 / (self.timestamps[1] - self.timestamps[0])


class Interval(ArrayDict):
    r"""An interval object is a set of time intervals each defined by a start time and
    an end time."""
    _sorted = None
    _lazy = False

    def __init__(
        self, start: np.ndarray, end: np.ndarray, *, timekeys=["start", "end"], **kwargs
    ):
        self.start = start
        self.end = end

        for key, value in kwargs.items():
            setattr(self, key, value)

        if "start" not in self.keys:
            timekeys.append("start")
        if "end" not in self.keys:
            timekeys.append("end")
        for key in timekeys:
            assert key in self.keys, f"Time attribute {key} not found in data."

        self._timekeys = timekeys

    def __setattr__(self, name, value):
        super(Interval, self).__setattr__(name, value)
        if name == "start" or name == "end":
            # start or end have been updated, we no longer know whether it is sorted
            # or not
            self._sorted = None

    @property
    def disjoint(self):
        # check if we already know that the sequence is sorted
        # if lazy loading, we'll have to skip this check
        if self._lazy:
            raise ValueError(
                "Cannot check if intervals are disjoint for lazy time series."
            )
        if not self.sorted:
            return copy.deepcopy(self).sort().disjoint
        return np.all(self.end[:-1] <= self.start[1:])

    @property
    def sorted(self):
        # check if we already know that the sequence is sorted
        # if lazy loading, we'll have to skip this check
        if self._sorted is None and not self._lazy:
            self._sorted = np.all(self.start[1:] >= self.start[:-1]) and np.all(
                self.end[1:] >= self.end[:-1]
            )
        return self._sorted

    def sort(self):
        r"""Sorts the timestamps."""
        if self._lazy:
            raise ValueError("Cannot sort lazy time series.")
        if not self.sorted:
            sorted_indices = np.argsort(self.start)
            for key in self.keys:
                self.__dict__[key] = self.__dict__[key][sorted_indices]
        self._sorted = True

        assert self.disjoint, "Intervals must be disjoint after sorting."

    def __len__(self) -> int:
        return self.start.shape[0]

    def __getitem__(self, item: Union[str, int, slice, list, np.ndarray]):
        r"""Allows indexing of the :obj:`Interval` object.

        It can be indexed with:
            - a string, which will return the corresponding attribute.
            - an integer, which will return a new :obj:`Interval` object with a single
            element and its corresponding attributes.
            - a slice, which will return a new :obj:`Interval` object with the selected
            elements and their corresponding attributes.
            - a list of integers, which will return a new :obj:`Interval` object with
            the selected elements and their corresponding attributes.
            - a binary mask, which will return a new :obj:`Interval` object with the
            selected elements and their corresponding attributes.

        Example:
            >> import numpy as np
            >> from kirby.data import Interval
            >> interval = Interval(np.array([0, 1, 2]), np.array([1, 2, 3]))
            >> interval[0]
            Interval(start=np.array([0]), end=np.array([1]))
            >> interval[0:2]
            Interval(start=np.array([0, 1]), end=np.array([1, 2]))
            >> interval[[0, 2]]
            Interval(start=np.array([0, 2]), end=np.array([1, 3]))
            >> interval[[True, False, True]]
            Interval(start=np.array([0, 2]), end=np.array([1, 3]))
        """
        if isinstance(item, str):
            # return the corresponding attribute
            return getattr(self, item)
        else:
            out = self.__class__.__new__(self.__class__)
            for key, value in self.__dict__.items():
                if key == "_sorted":
                    out.__dict__[
                        "_sorted"
                    ] = None  # We don't know if these intervals are sorted
                elif key == "_timekeys":
                    out.__dict__["_timekeys"] = value
                else:
                    out.__dict__[key] = value[item]

            return out

    def slice(self, start: float, end: float, request_keys: Optional[List[str]] = None):
        if request_keys is None:
            request_keys = self.keys

        if self._lazy:
            # load the request keys only and return a new object
            out = self.__class__.__new__(self.__class__)
            out._timekeys = self._timekeys

            for key in self.keys:
                if key in request_keys:
                    out.__dict__[key] = self.__dict__[key][:]  # load into memory

            return out.slice(start, end, request_keys=request_keys)

        assert "start" in request_keys and "end" in request_keys

        if not self.sorted:
            self.sort()

        idx_l = np.searchsorted(
            self.end, start
        )  # anything that starts before the end of the slicing window
        idx_r = np.searchsorted(
            self.start, end, side="right"
        )  # anything that will end after the start of the slicing window

        out = self.__class__.__new__(self.__class__)
        out._timekeys = self._timekeys

        for key in self.keys:
            if key in request_keys:
                out.__dict__[key] = self.__dict__[key][idx_l:idx_r].copy()

        for key in self._timekeys:
            if key in request_keys:
                out.__dict__[key] = out.__dict__[key] - start
        return out

    def split(
        self,
        sizes: Union[List[int], List[float]],
        *,
        shuffle=False,
        random_seed=None,
    ):
        r"""Splits the set of intervals into multiple subsets. This will
        return a number of new :obj:`Interval` objects equal to the number of elements
        in `sizes`. If `shuffle` is set to :obj:`True`, the intervals will be shuffled
        before splitting.

        Args:
            sizes: A list of integers or floats. If integers, the list must sum to the
            number of intervals. If floats, the list must sum to 1.0.
            shuffle: If :obj:`True`, the intervals will be shuffled before splitting.
            random_seed: The random seed to use for shuffling.

        .. note::
            This method will not guarantee that the resulting sets will be disjoint, if
            the intervals are not already disjoint.
        """

        assert len(sizes) > 1, "must split into at least two sets"
        assert len(sizes) < len(self), f"cannot split {len(self)} intervals into "
        " {len(sizes)} sets"

        # if sizes are floats, convert them to integers
        if all(isinstance(x, float) for x in sizes):
            assert sum(sizes) == 1.0, "sizes must sum to 1.0"
            sizes = [round(x * len(self)) for x in sizes]
            # there might be rounding errors
            # make sure that the sum of sizes is still equal to the number of intervals
            largest = np.argmax(sizes)
            sizes[largest] = len(self) - (sum(sizes) - sizes[largest])
        elif all(isinstance(x, int) for x in sizes):
            assert sum(sizes) == len(self), "sizes must sum to the number of intervals"
        else:
            raise ValueError("sizes must be either all floats or all integers")

        # shuffle
        if shuffle:
            rng = np.random.default_rng(random_seed)  # Create a new generator instance
            idx = rng.permutation(len(self))  # Use the generator for permutation
        else:
            idx = np.arange(len(self))  # Create a sequential index array

        # split
        splits = []
        start = 0
        for size in sizes:
            splits.append(self[idx[start : start + size]])
            start += size

        return splits

    @classmethod
    def linspace(cls, start: float, end: float, steps: int):
        """Create a regular interval with a given number of samples."""
        timestamps = np.linspace(start, end, steps + 1)
        return cls(
            start=timestamps[:-1],
            end=timestamps[1:],
        )

    @classmethod
    def from_dataframe(cls, df):
        r"""Create an :obj:`Interval` object from a Pandas dataframe. The dataframe
        must have a start time and end time columns. The names of these columns need
        to be "start" and "end" (use `pd.Dataframe.rename` if needed).

        Columns that are numeric will be converted to ndarrays. Columns that are
        ndarrays will be stacked if they share the same size. Any other column type will
        be skipped.

        # todo: add support for string ndarrays

        Raises:
            AssertionError: if the start or end column is not found in the dataframe.
            AssertionError: if a column is specified in `kwargs` but is not found in
            the dataframe.
            Warning: if a column is not numeric or an ndarray.
        """
        data = {}

        assert "start" in df.columns, f"Column 'start' not found in dataframe."
        assert "end" in df.columns, f"Column 'end' not found in dataframe."

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                # Directly convert numeric columns to numpy arrays
                data[column] = df[column].to_numpy()
            elif df[column].apply(lambda x: isinstance(x, np.ndarray)).all():
                # Check if all ndarrays in the column have the same shape
                ndarrays = df[column]
                first_shape = ndarrays.iloc[0].shape
                if all(
                    arr.shape == first_shape
                    for arr in ndarrays
                    if isinstance(arr, np.ndarray)
                ):
                    # If all elements in the column are ndarrays with the same shape,
                    # stack them
                    data[column] = np.stack(df[column].values)
                else:
                    logging.warn(
                        f"The ndarrays in column '{column}' do not all have the "
                        "same shape."
                    )
            else:
                logging.warn(
                    f"Unable to convert column '{column}' to a array. Skipping."
                )
        return cls(**data)

    def to_hdf5(self, file):
        for key in self.keys:
            value = getattr(self, key)
            file.create_dataset(key, data=value)

        file.attrs["timekeys"] = np.array(self._timekeys, dtype="S")
        file.attrs["object"] = self.__class__.__name__

    @classmethod
    def from_hdf5(cls, file):
        assert file.attrs["object"] == cls.__name__, "object type mismatch"
        data = {}
        for key, value in file.items():
            data[key] = value

        obj = cls(**data)
        obj._lazy = True

        obj._timekeys = file.attrs["timekeys"].astype(str).tolist()

        return obj


class Hemisphere(StringIntEnum):
    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class Channel(Dictable):
    """Channels are the physical channels used to record the data. Channels are grouped
    into probes."""

    id: str
    local_index: int

    # Position relative to the reference location of the probe, in microns.
    relative_x_um: float
    relative_y_um: float
    relative_z_um: float

    area: StringIntEnum
    hemisphere: Hemisphere = Hemisphere.UNKNOWN


@dataclass
class Probe(Dictable):
    """Probes are the physical probes used to record the data."""

    id: str
    type: RecordingTech
    lfp_sampling_rate: float
    wideband_sampling_rate: float
    waveform_sampling_rate: float
    waveform_samples: int
    channels: list[Channel]
    ecog_sampling_rate: float = 0.0


AttrTensor = Union[np.ndarray, ArrayDict]


class Data(object):
    def __init__(
        self,
        *,
        start: Optional[float] = None,
        end: Optional[float] = None,
        spikes: Optional[IrregularTimeSeries] = None,
        **kwargs,
    ):
        # if any time-based attribute is present, start and end must be specified
        if spikes is not None or any(
            isinstance(value, (IrregularTimeSeries, RegularTimeSeries, Interval))
            for value in kwargs.values()
        ):
            assert (
                start is not None and end is not None
            ), "If any time-based attribute is present, start and end must be specified."

        self.start = start
        self.end = end

        # these variables will hold the original start and end times
        # and won't be modified when slicing
        self.original_start = start
        self.original_end = end

        if spikes is not None:
            self.spikes = spikes

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, name, value):
        if isinstance(value, (IrregularTimeSeries, RegularTimeSeries, Interval)):
            assert self.start is not None and self.end is not None, (
                "Attempted to set an time-based attribute, but start and end times were"
                " not specified. Please set start and end times first."
            )
        super(Data, self).__setattr__(name, value)

    def slice(self, start: float, end: float, request_keys: Optional[List[str]] = None):
        if self.start is None:
            # this data object does not have any time-based attributes
            return copy.deepcopy(self)

        out = self.__class__.__new__(self.__class__)

        if request_keys is None:
            request_keys = self.keys

        request_tree = parse_request_keys(request_keys)

        for key, value in self.__dict__.items():
            if key in request_tree:
                assert isinstance(
                    value, (Data, IrregularTimeSeries, RegularTimeSeries, Interval)
                ), f"Cannot slice {key} of type {type(value)}."
                out.__dict__[key] = value.slice(
                    start, end, request_keys=request_tree[key]
                )
            elif key in request_tree["_root"]:
                if isinstance(
                    value, (IrregularTimeSeries, RegularTimeSeries, Interval)
                ):
                    out.__dict__[key] = value.slice(start, end, request_keys=None)
                else:
                    out.__dict__[key] = copy.copy(value)

        # keep track of the original start and end times
        out.original_start = self.original_start + start - self.start
        out.original_end = self.original_end + end - self.end

        # update the start and end times relative to the new slice
        out.start = 0.0
        out.end = end - start
        return out

    def __repr__(self) -> str:
        cls = self.__class__.__name__

        info = ""
        for key, value in self.__dict__.items():
            if isinstance(value, ArrayDict):
                info = info + key + "=" + repr(value) + ",\n"
            elif value is not None:
                info = info + size_repr(key, value) + ",\n"
        info = info.rstrip()
        return f"{cls}(\n{info}\n)"

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        return copy.deepcopy(self.__dict__)

    def to_hdf5(self, file):
        for key, value in self.__dict__.items():
            if isinstance(
                value, (Data, IrregularTimeSeries, RegularTimeSeries, Interval)
            ):
                grp = file.create_group(key)
                value.to_hdf5(grp)
            elif isinstance(value, np.ndarray):
                file.create_dataset(key, data=value)
            elif value is not None:
                # each attribute should be small (generally < 64k)
                # there is no partial I/O; the entire attribute must be read
                file.attrs[key] = value
        file.attrs["object"] = self.__class__.__name__

    @classmethod
    def from_hdf5(cls, file):
        data = {}
        for key, value in file.items():
            if isinstance(value, h5py.Group):
                group_cls = globals()[value.attrs["object"]]
                data[key] = group_cls.from_hdf5(value)
            else:
                data[key] = value[:]

        for key, value in file.attrs.items():
            # Things like "start", "end", "original_start" etc are file attributes
            data[key] = value

        obj = cls(**data)

        return obj

    @property
    def keys(self) -> List[str]:
        r"""Returns a list of all attribute names."""
        return list(self.__dict__.keys())

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def get_nested_attribute(self, path: str) -> Any:
        # Split key by dots, resolve using getattr
        components = path.split(".")
        out = self
        for c in components:
            try:
                out = getattr(out, c)
            except AttributeError:
                raise AttributeError(
                    f"Could not resolve {path} in data (specifically, at level {c}))"
                )
        return out


def parse_request_keys(request_keys: List[str]) -> Dict[str, List[str]]:
    request_tree = defaultdict(list)
    for key in request_keys:
        if "." in key:
            key, subkey = key.split(".", 1)
            request_tree[key].append(subkey)
        else:
            request_tree["_root"].append(key)
    return request_tree

    def get_nested_attribute(self, path: str) -> Any:
        # Split key by dots, resolve using getattr
        components = path.split(".")
        out = self
        for c in components:
            try:
                out = getattr(out, c)
            except AttributeError:
                raise AttributeError(
                    f"Could not resolve {path} in data (specifically, at level {c}))"
                )
        return out


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, torch.Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, torch.Tensor):
        out = str(list(value.size()))
    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = "{}"
    elif (
        isinstance(value, Mapping)
        and len(value) == 1
        and not isinstance(list(value.values())[0], Mapping)
    ):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = "{ " + ", ".join(lines) + " }"
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + pad + "}"
    else:
        out = str(value)
    key = str(key).replace("'", "")
    return f"{pad}{key}={out}"


def recursive_apply(data: Any, func: Callable) -> Any:
    if isinstance(data, torch.Tensor):
        return func(data)
    elif isinstance(data, torch.nn.utils.rnn.PackedSequence):
        return func(data)
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(*(recursive_apply(d, func) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [recursive_apply(d, func) for d in data]
    elif isinstance(data, Mapping):
        return {key: recursive_apply(data[key], func) for key in data}
    else:
        try:
            return func(data)
        except:  # noqa
            return data
