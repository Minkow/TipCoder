"""Utilities for serializing reward model payloads over HTTP.

The remote reward service returns tensors and other rich Python objects that
need to be transported as JSON. These helpers encode and decode values in a
lossless way so the training pipeline receives exactly the same data structure
as the local reward implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Union

import torch

_TENSOR_MARKER = "__tensor__"


@dataclass
class TensorPayload:
    """Structured representation of a tensor for JSON transport."""

    dtype: str
    shape: List[int]
    values: List[Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            _TENSOR_MARKER: True,
            "dtype": self.dtype,
            "shape": self.shape,
            "values": self.values,
        }


def _dtype_to_name(dtype: torch.dtype) -> str:
    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        return name
    return str(dtype).replace("torch.", "")


def _name_to_dtype(name: str) -> torch.dtype:
    # torch has dtype attributes like torch.float32; try to resolve
    attr_name = name if name.startswith("torch.") else f"torch.{name}"
    try:
        return getattr(torch, attr_name.split(".")[-1])
    except AttributeError:
        return torch.float32


def _tensor_to_payload(tensor: torch.Tensor) -> TensorPayload:
    tensor_cpu = tensor.detach().cpu()
    dtype_name = _dtype_to_name(tensor_cpu.dtype)
    shape = list(tensor_cpu.shape)
    values = tensor_cpu.tolist()
    return TensorPayload(dtype=dtype_name, shape=shape, values=values)


def encode_for_transport(obj: Any) -> Any:
    """Recursively encode Python objects into JSON-serializable values."""

    if isinstance(obj, torch.Tensor):
        return _tensor_to_payload(obj).to_dict()
    if isinstance(obj, dict):
        return {key: encode_for_transport(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_for_transport(item) for item in obj]
    if isinstance(obj, set):
        return [encode_for_transport(item) for item in obj]
    return obj


def _payload_to_tensor(data: Dict[str, Any]) -> torch.Tensor:
    dtype = _name_to_dtype(str(data.get("dtype", "float32")))
    values = data.get("values", [])
    tensor = torch.tensor(values, dtype=dtype)
    return tensor


def decode_from_transport(obj: Any) -> Any:
    """Restore Python objects from transport-safe representations."""

    if isinstance(obj, dict):
        if obj.get(_TENSOR_MARKER):
            return _payload_to_tensor(obj)
        return {key: decode_from_transport(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [decode_from_transport(item) for item in obj]
    return obj
