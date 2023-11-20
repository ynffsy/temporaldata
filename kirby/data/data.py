"""Inspired from PyG's data class."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping, Sequence
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from kirby.taxonomy import Dictable, RecordingTech, StringIntEnum


class DatumBase(object):
    @property
    def keys(self) -> List[str]:
        r"""Returns a list of all attribute names."""
        return list(self.__dict__.keys())

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the data."""
        return key in self.keys

    def __len__(self) -> int:
        raise NotImplementedError

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
        info = [size_repr(k, v, indent=2) for k, v in self.__dict__.items()]
        info = ",\n".join(info)
        return f"{cls}(\n{info}\n)"

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        raise NotImplementedError

    def to_namedtuple(self) -> NamedTuple:
        r"""Returns a :obj:`NamedTuple` of stored key/value pairs."""
        raise NotImplementedError

    @property
    def attrs(self) -> List[str]:
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError


class IrregularTimeSeries(DatumBase):
    def __init__(self, timestamps: Tensor, **kwargs):
        super().__init__()

        self.timestamps = timestamps

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, name, value):
        super(IrregularTimeSeries, self).__setattr__(name, value)
        if name == "timestamps":
            # timestamps has been updated, we no longer know whether it is sorted or not
            self._sorted = None
            # timestamps has been updated, any precomputed index dict are no longer valid
            for key in self.attrs:
                if key.endswith("_index_dict"):
                    delattr(self, key)

    @property
    def sorted(self):
        # check if we already know that the sequence is sorted
        if self._sorted is None:
            self._sorted = (
                torch.all(self.timestamps[1:] >= self.timestamps[:-1])
                .detach()
                .cpu()
                .item()
            )
        return self._sorted

    def __len__(self) -> int:
        r"""Returns the number of graph attributes."""
        return self.timestamps.size(0)

    @property
    def attrs(self) -> List[str]:
        r"""Returns all tensor attribute names."""
        return list(set(self.keys).difference({"timestamps"}))

    @property
    def start(self) -> float:
        return torch.min(self.timestamps).item()

    @property
    def end(self) -> float:
        return torch.max(self.timestamps).item()

    def sort(self):
        r"""Sorts the timestamps."""
        if not self.sorted:
            sorted_indices = torch.argsort(self.timestamps)
            for key, value in self.__dict__.items():
                if isinstance(value, Tensor):
                    self.__dict__[key] = value[sorted_indices]
        self._sorted = True

    def slice(self, start, end):
        # assert self.sorted, "Timestamps must be sorted"
        # todo: maybe can still speed up with dict-lookup indexing

        if not self.sorted:
            self.sort()

        # torch.searchsorted uses binary search
        idx_l = torch.searchsorted(self.timestamps, start)
        idx_r = torch.searchsorted(self.timestamps, end, right=True)

        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if isinstance(value, Tensor):
                out.__dict__[key] = value[idx_l:idx_r].clone()
            elif isinstance(value, np.ndarray):
                # E.g. array of strings.
                out.__dict__[key] = value[idx_l:idx_r].copy()
            elif isinstance(value, list):
                # e.g. lists of names.
                out.__dict__[key] = value[idx_l:idx_r]

        out.timestamps = out.timestamps - start
        return out

    def clip(self, start=None, end=None):
        r"""While :meth:`slice` resets the timestamps, this method does not."""
        if not self.sorted:
            self.sort()
        assert (
            start is not None or end is not None
        ), "start or/and end must be specified"

        idx_l = idx_r = None

        if start is not None:
            idx_l = torch.searchsorted(self.timestamps, start)

        if end is not None:
            idx_r = torch.searchsorted(self.timestamps, end, right=True)

        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value[idx_l:idx_r].clone()
        return out


class Interval(DatumBase):
    r"""An interval object is a set of time intervals each defined by a start time and
    an end time."""

    def __init__(self, start: torch.Tensor, end: torch.Tensor, **kwargs):
        self.start = start
        self.end = end

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return self.start.size(0)

    def __getitem__(self, item: Union[str, int, slice, list, Tensor, np.ndarray]):
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
            >> import torch
            >> from kirby.data import Interval
            >> interval = Interval(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
            >> interval[0]
            Interval(start=tensor([0]), end=tensor([1]))
            >> interval[0:2]
            Interval(start=tensor([0, 1]), end=tensor([1, 2]))
            >> interval[[0, 2]]
            Interval(start=tensor([0, 2]), end=tensor([1, 3]))
            >> interval[[True, False, True]]
            Interval(start=tensor([0, 2]), end=tensor([1, 3]))
        """
        if isinstance(item, str):
            # return the corresponding attribute
            return getattr(self, item)
        else:
            out = self.__class__.__new__(self.__class__)
            for key, value in self.__dict__.items():
                out.__dict__[key] = value[item]

            return out

    def slice(self, start, end):
        # torch.searchsorted uses binary search
        idx_l = torch.searchsorted(
            self.end, start
        )  # anything that starts before the end of the slicing window
        idx_r = torch.searchsorted(
            self.start, end, right=True
        )  # anything that will end after the start of the slicing window

        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if isinstance(value, Tensor):
                out.__dict__[key] = value[idx_l:idx_r].clone()
            elif isinstance(value, np.ndarray):
                # E.g. array of strings.
                out.__dict__[key] = value[idx_l:idx_r].copy()
            elif isinstance(value, list):
                # e.g. lists of names.
                out.__dict__[key] = value[idx_l:idx_r]

        out.start = out.start - start
        out.end = out.end - start
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
            generator = torch.Generator()
            generator.manual_seed(random_seed)
            idx = torch.randperm(len(self), generator=generator)
        else:
            idx = torch.arange(len(self))

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
        timestamps = torch.linspace(start, end, steps + 1)
        return cls(
            start=timestamps[:-1],
            end=timestamps[1:],
        )

    @classmethod
    def from_dataframe(cls, df):
        r"""Create an :obj:`Interval` object from a Pandas dataframe. The dataframe
        must have a start time and end time columns. The names of these columns need
        to be "start" and "end" (use `pd.Dataframe.rename` if needed).

        Columns that are numeric will be converted to tensors. Columns that are
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
                data[column] = torch.tensor(df[column].to_numpy())
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
                    data[column] = torch.tensor(np.stack(df[column].values))
                else:
                    logging.warn(
                        f"The ndarrays in column '{column}' do not all have the "
                        "same shape."
                    )
            else:
                logging.warn(
                    f"Unable to convert column '{column}' to a tensor. Skipping."
                )
        return cls(**data)


class RegularTimeSeries(IrregularTimeSeries):
    """A regular time series is the same as a regular time series, but it has a
    regular sampling rate. This allows for faster indexing and meaningful Fourier
    operations.

    For now, we simply do a pass-through, but later we will implement faster
    algorithms.
    """

    @property
    def sampling_rate(self):
        return 1 / (self.timestamps[1] - self.timestamps[0])


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


AttrTensor = Union[Tensor, DatumBase]


class Data(object):
    def __init__(
        self,
        *,
        start: Optional[float] = None,
        end: Optional[float] = None,
        spikes: Optional[IrregularTimeSeries] = None,
        **kwargs,
    ):
        # unit_attr: OptTensor = None,
        # x_timestamps: OptTensor = None, x_attr: OptTensor = None,
        # y_start: OptTensor = None, y_end: OptTensor = None, y_attr: OptTensor = None,

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

    def slice(self, start, end):
        out = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            if isinstance(value, (IrregularTimeSeries, RegularTimeSeries, Interval)):
                out.__dict__[key] = value.slice(start, end)
            else:
                out.__dict__[key] = copy.copy(value)

        # keep track of the original start and end times
        out.original_start = self.original_start + start - self.start
        out.original_end = self.original_end + end - self.end

        # update the start and end times relative to the new slice
        out.start = torch.tensor(0.0)
        out.end = end - start
        return out

    def slice_along_fixed(self, key, start, length, offset=0):
        start_vec = self.__dict__[key].__dict__[start]
        for id, start in enumerate(start_vec):
            start = start + offset
            end = start + length
            out = self.__class__.__new__(self.__class__)
            for key_, value in self.__dict__.items():
                if isinstance(value, IrregularTimeSeries):
                    out.__dict__[key_] = value.slice(start, end)
                elif key_ == key:
                    out.__dict__[key_] = value[id]
                else:
                    out.__dict__[key_] = copy.copy(value)
            yield out

    def slice_along(self, key, start, end):
        start_vec = self.__dict__[key].__dict__[start]
        end_vec = self.__dict__[key].__dict__[end]
        for id, (start, end) in enumerate(zip(start_vec, end_vec)):
            out = self.__class__.__new__(self.__class__)
            for key_, value in self.__dict__.items():
                if isinstance(value, IrregularTimeSeries):
                    out.__dict__[key_] = value.slice(start, end)
                elif key_ == key:
                    out.__dict__[key_] = value[id]
                else:
                    out.__dict__[key_] = copy.copy(value)
            yield out

    def __repr__(self) -> str:
        cls = self.__class__.__name__

        info = ""
        for key, value in self.__dict__.items():
            if isinstance(value, DatumBase):
                info = info + key + "=" + repr(value) + ",\n"
            elif value is not None:
                info = info + size_repr(key, value) + ",\n"
        info = info.rstrip()
        return f"{cls}(\n{info}\n)"

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        return copy.deepcopy(self.__dict__)

    def save_to(self, f) -> None:
        r"""Saves self to a file with pickle"""
        with open(f, "wb") as output:
            pickle.dump(self.__dict__, output)

    @staticmethod
    def load_from(f) -> Data:
        r"""Load and return Data object from filename"""
        data = Data()
        with open(f, "rb") as fp:
            data.__dict__ = pickle.load(fp)
        return data

    def bucketize(self, bucket_size, step, jitter):
        r"""Bucketize the data into buckets of size bucket_size (in seconds))."""
        assert (
            self.start is not None and self.end is not None
        ), "start and end must be specified"

        # get start and end of buckets
        bucket_start = self.start + np.arange(0, self.end - self.start, step)
        bucket_end = bucket_start + bucket_size

        # add padding to buckets to enable jittering
        bucket_start = np.maximum(self.start, bucket_start - jitter)
        bucket_end = np.minimum(bucket_end + jitter, self.end)

        for id, (start, end) in enumerate(zip(bucket_start, bucket_end)):
            out = self.__class__.__new__(self.__class__)
            for key_, value in self.__dict__.items():
                if isinstance(value, IrregularTimeSeries):
                    out.__dict__[key_] = value.slice(start, end)
                else:
                    out.__dict__[key_] = copy.copy(value)
            out.start, out.end = start, end
            yield out

    ###########################################################################

    @property
    def keys(self) -> List[str]:
        r"""Returns a list of all attribute names."""
        return list(self.__dict__.keys())

    def __len__(self) -> int:
        r"""Returns the number of graph attributes."""
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def apply(self, func: Callable):
        r"""Applies the function :obj:`func` to all attributes."""
        for key, value in self.__dict__.items():
            setattr(self, key, recursive_apply(value, func))
        return self

    def clone(self):
        r"""Performs cloning of tensors for all attributes."""
        return copy.copy(self).apply(lambda x: x.clone())

    def contiguous(self):
        r"""Ensures a contiguous memory layout for all attributes."""
        return self.apply(lambda x: x.contiguous())

    def to(self, device: Union[int, str], non_blocking: bool = False):
        r"""Performs tensor device conversion for all attributes."""
        return self.apply(lambda x: x.to(device=device, non_blocking=non_blocking))

    def cpu(self):
        r"""Copies attributes to CPU memory, either for all attributes or only
        the ones given in :obj:`*args`."""
        return self.apply(lambda x: x.cpu())

    def cuda(
        self, device: Optional[Union[int, str]] = None, non_blocking: bool = False
    ):
        r"""Copies attributes to CUDA memory, either for all attributes ."""
        # Some PyTorch tensor like objects require a default value for `cuda`:
        device = "cuda" if device is None else device
        return self.apply(
            lambda x: x.cuda(device, non_blocking=non_blocking),
        )

    def pin_memory(self):
        r"""Copies attributes to pinned memory for all attributes."""
        return self.apply(lambda x: x.pin_memory())

    def share_memory_(self):
        r"""Moves attributes to shared memory for all attributes."""
        return self.apply(lambda x: x.share_memory_())

    def detach_(self, *args: List[str]):
        r"""Detaches attributes from the computation graph."""
        return self.apply(lambda x: x.detach_())

    def detach(self):
        r"""Detaches attributes from the computation graph by creating a new tensor."""
        return self.apply(lambda x: x.detach())

    def requires_grad_(self, requires_grad: bool = True):
        r"""Tracks gradient computation for all attributes."""
        return self.apply(lambda x: x.requires_grad_(requires_grad=requires_grad))

    @property
    def is_cuda(self) -> bool:
        r"""Returns :obj:`True` if any :class:`torch.Tensor` attribute is
        stored on the GPU, :obj:`False` otherwise."""
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.is_cuda:
                return True
        return False


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, Tensor):
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
    if isinstance(data, Tensor):
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
