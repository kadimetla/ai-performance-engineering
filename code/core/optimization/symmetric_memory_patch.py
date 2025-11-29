"""Compatibility layer for torch.distributed.nn.SymmetricMemory.

PyTorch 2.10 exposes symmetric memory primitives through the private
``torch.distributed._symmetric_memory`` namespace.  Older builds (and
minimal installs) do not register ``torch.distributed.nn.SymmetricMemory``
even though the underlying primitives exist.  Several chapter-4 examples
expect ``dist.nn.SymmetricMemory`` to be available, so we provide a thin
wrapper that:

1. Uses the official ``torch.distributed._symmetric_memory`` rendezvous
   API when it is present and the tensor lives on a CUDA device.
2. Installs the wrapper as ``torch.distributed.nn.SymmetricMemory`` exactly
   once, without stomping on future upstream implementations.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, Optional, Union, TYPE_CHECKING, cast

try:
    import torch
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d
except Exception:  # pragma: no cover - torch not available in some tooling environments
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    c10d = None  # type: ignore[assignment]

_symm_mem: Any
if torch is not None:
    try:  # pragma: no cover - import guarded for CPU-only builds
        from torch.distributed import _symmetric_memory as _symm_mem
    except ImportError:  # pragma: no cover
        _symm_mem = None
else:  # pragma: no cover
    _symm_mem = None

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup as _ProcessGroup
else:
    _ProcessGroup = object

ProcessGroup = _ProcessGroup


def _resolve_process_group(
    group: Union[str, ProcessGroup, None]
) -> ProcessGroup:
    """Normalize ``group`` to a ``ProcessGroup`` instance."""
    if dist is None or c10d is None:
        raise RuntimeError("torch.distributed is not available")
    if group is None:
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before using SymmetricMemory"
            )
        if dist.group.WORLD is None:
            raise RuntimeError("Default process group is not initialized")
        return dist.group.WORLD
    if isinstance(group, str):
        return c10d._resolve_process_group(group)
    if isinstance(group, c10d.ProcessGroup):
        return group
    raise TypeError(f"Unsupported process group type: {type(group)}")


class SymmetricMemory:
    """High level wrapper that mirrors ``torch.distributed.nn.SymmetricMemory``."""

    def __init__(
        self,
        tensor: torch.Tensor,
        group: Union[str, ProcessGroup, None] = None,
    ) -> None:
        if torch is None or dist is None:
            raise RuntimeError("PyTorch is not available")
        if _symm_mem is None:
            raise RuntimeError(
                "torch.distributed._symmetric_memory is not available. "
                "Install PyTorch 2.10+ with symmetric memory support."
            )
        if not tensor.is_cuda:
            raise ValueError("SymmetricMemory tensors must live on CUDA devices")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SymmetricMemory")

        self._group = _resolve_process_group(group)
        self._rank = dist.get_rank(group=self._group)
        self._world_size = dist.get_world_size(group=self._group)
        self._handle: Any
        self.buffer: torch.Tensor = tensor

        self._shape = tuple(tensor.shape)
        self._dtype = tensor.dtype

        symm_tensor = _symm_mem.empty(
            self._shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if tensor.numel() > 0:
            symm_tensor.copy_(tensor)
        handle = _symm_mem.rendezvous(symm_tensor, self._group)
        self.buffer = cast(
            torch.Tensor,
            cast(Any, handle).get_buffer(self._rank, self._shape, self._dtype),
        )
        self._handle = handle
        backend = _symm_mem.get_backend(tensor.device)
        self.backend = backend if backend is not None else "symmetric_memory"

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def get_buffer(self, rank: int) -> torch.Tensor:
        """Return a view of the symmetric buffer for ``rank``."""
        return cast(
            torch.Tensor,
            cast(Any, self._handle).get_buffer(rank, self._shape, self._dtype),
        )


def ensure_symmetric_memory_api() -> None:
    """Install ``dist.nn.SymmetricMemory`` if the upstream build did not register it."""
    if torch is None or dist is None:
        return
    
    module_name = "torch.distributed.nn"
    try:
        nn_module = importlib.import_module(module_name)
        setattr(dist, "nn", nn_module)
    except ImportError:
        existing = getattr(dist, "nn", None)
        if isinstance(existing, types.ModuleType):
            nn_module = existing
        else:
            nn_module = types.ModuleType(module_name)
            setattr(dist, "nn", nn_module)
        sys.modules.setdefault(module_name, nn_module)

    if not hasattr(nn_module, "SymmetricMemory"):
        setattr(nn_module, "SymmetricMemory", SymmetricMemory)
