"""Inspired from PyG's data class."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping, Sequence
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

    def validate(self):
        pass

    def is_monotonic(self):
        raise NotImplementedError

    def slice(self, start, end):
        # todo: maybe can still speed up with dict-lookup indexing

        # torch.searchsorted uses binary search
        idx_l = torch.searchsorted(self.timestamps, start)
        idx_r = torch.searchsorted(self.timestamps, end)

        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if isinstance(value, Tensor):
                out.__dict__[key] = value[idx_l:idx_r].clone()
            elif isinstance(value, np.ndarray):
                # E.g. array of strings.
                out.__dict__[key] = value[idx_l:idx_r].copy()

        out.timestamps = out.timestamps - start
        return out

    def clip(self, start=None, end=None):
        assert (
            start is not None or end is not None
        ), "start or/and end must be specified"

        idx_l = idx_r = None

        if start is not None:
            idx_l = torch.searchsorted(self.timestamps, start)

        if end is not None:
            idx_r = torch.searchsorted(self.timestamps, end)

        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value[idx_l:idx_r].clone()
        return out


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


class Interval(DatumBase):
    def __init__(self, start, end, **kwargs):
        self.start = start
        self.end = end
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return self.start.size(0)

    def __getitem__(self, item):
        out = {}
        for key, value in self.__dict__.items():
            out[key] = value[item]
        return out


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

        if spikes is not None:
            self.spikes = spikes

        for key, value in kwargs.items():
            setattr(self, key, value)

    def slice(self, start, end):
        out = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            if isinstance(value, IrregularTimeSeries):
                out.__dict__[key] = value.slice(start, end)
            else:
                out.__dict__[key] = copy.copy(value)

        out.start = start
        out.end = end
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
