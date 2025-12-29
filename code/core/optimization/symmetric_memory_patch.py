"""Helpers for torch.distributed._symmetric_memory without shims."""

from __future__ import annotations

from typing import Any, Optional, Union, TYPE_CHECKING, cast

try:
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover - torch not available in some tooling environments
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]

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


def symmetric_memory_available() -> bool:
    """Return True when NVSHMEM-backed symmetric memory is available."""
    if torch is None or dist is None or _symm_mem is None:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return bool(_symm_mem.is_nvshmem_available())
    except Exception:
        return False


class SymmetricMemoryHandle:
    """Direct symmetric memory handle backed by torch.distributed._symmetric_memory."""

    __slots__ = (
        "buffer",
        "_symm_tensor",
        "_handle",
        "_shape",
        "_dtype",
        "_group",
        "_rank",
        "_world_size",
        "backend",
    )

    def __init__(self, tensor: torch.Tensor, group: Union[str, ProcessGroup, None] = None) -> None:
        if torch is None or dist is None or _symm_mem is None:
            raise RuntimeError("torch.distributed._symmetric_memory is not available")
        if not tensor.is_cuda:
            raise ValueError("SymmetricMemory tensors must live on CUDA devices")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SymmetricMemory")
        if group is None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized before using SymmetricMemory"
                )
            group = dist.group.WORLD

        if not symmetric_memory_available():
            raise RuntimeError("NVSHMEM symmetric memory is not available on this system")

        backend = None
        try:
            backend = _symm_mem.get_backend(tensor.device)
        except Exception:
            backend = None
        if backend != "NVSHMEM":
            try:
                _symm_mem.set_backend("NVSHMEM")
                backend = _symm_mem.get_backend(tensor.device)
            except Exception as exc:
                raise RuntimeError("NVSHMEM backend is not available for symmetric memory") from exc

        self._group = group
        self._rank = dist.get_rank(group=group)
        self._world_size = dist.get_world_size(group=group)
        self._shape = tuple(tensor.shape)
        self._dtype = tensor.dtype

        self._symm_tensor = _symm_mem.empty(
            self._shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if tensor.numel() > 0:
            self._symm_tensor.copy_(tensor)

        self._handle = _symm_mem.rendezvous(self._symm_tensor, group)
        self.buffer = cast(
            torch.Tensor,
            self._handle.get_buffer(self._rank, self._shape, self._dtype),
        )
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
            self._handle.get_buffer(rank, self._shape, self._dtype),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def create_symmetric_memory_handle(
    tensor: torch.Tensor,
    group: Union[str, ProcessGroup, None] = None,
) -> SymmetricMemoryHandle:
    """Create a symmetric memory handle or raise if unavailable."""
    return SymmetricMemoryHandle(tensor, group=group)


def maybe_create_symmetric_memory_handle(
    tensor: torch.Tensor,
    group: Union[str, ProcessGroup, None] = None,
) -> Optional[SymmetricMemoryHandle]:
    """Return a symmetric memory handle when available; otherwise None."""
    if not symmetric_memory_available():
        return None
    try:
        return SymmetricMemoryHandle(tensor, group=group)
    except Exception:
        return None
