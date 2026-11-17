# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Iterable, Iterator, Sequence, Sized
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

__all__ = [
    "ShardSampler",
    "barrier_download",
    "ddp_device",
    "is_distributed",
    "is_main_rank",
    "reduce_sum",
    "sync_val_metric",
]


def is_distributed() -> bool:
    """Whether a process group with more than one rank is active"""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def is_main_rank() -> bool:
    """Whether this process is rank 0 (always True when not distributed)"""
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def ddp_device() -> torch.device:
    """Device to put collective payloads on"""
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def reduce_sum(values: Sequence[float]) -> list[float]:
    """Sum a short list of scalars across all ranks. No-op when not distributed"""
    if not is_distributed():
        return [float(v) for v in values]
    tensor = torch.tensor([float(v) for v in values], dtype=torch.float64, device=ddp_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def _gather_lists(value: list[Any]) -> list[Any]:
    """Concatenate a per-rank list into the full list, in rank order"""
    gathered: list[list[Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return [item for part in gathered if part is not None for item in part]


def sync_val_metric(
    val_metric: Any,
    counters: Iterable[str] = (),
    buffers: Iterable[str] = (),
) -> None:
    """Merge per-rank validation state so `summary()` reflects the whole validation set

    Args:
        val_metric: the metric instance to update in place
        counters: names of scalar accumulators to sum (e.g. `num_gts`, `matches`)
        buffers: names of list accumulators to concatenate (e.g. `_gts`, `_preds`)
    """
    if not is_distributed():
        return

    counter_names = [name for name in counters if hasattr(val_metric, name)]
    if counter_names:
        totals = reduce_sum([float(getattr(val_metric, name)) for name in counter_names])
        for name, total in zip(counter_names, totals):
            # Preserve int/float so downstream divisions behave as before
            setattr(val_metric, name, type(getattr(val_metric, name))(total))

    for name in buffers:
        if hasattr(val_metric, name):
            setattr(val_metric, name, _gather_lists(getattr(val_metric, name)))


def barrier_download(rank: int, distributed: bool):
    """Context manager that lets rank 0 populate the docTR cache before the others read it"""

    def _barrier():
        if torch.cuda.is_available() and dist.get_backend() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()

    class _BarrierDownload:
        def __enter__(self):
            if distributed and rank != 0:
                _barrier()  # wait for rank 0 to finish downloading
            return self

        def __exit__(self, *exc_info):
            if distributed and rank == 0:
                _barrier()  # release the other ranks
            return False

    return _BarrierDownload()


class ShardSampler(Sampler[int]):
    """Split a dataset across ranks without padding, so no sample is evaluated twice"""

    def __init__(self, dataset: Sized, rank: int | None = None, num_replicas: int | None = None):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if is_distributed() else 1
        if rank is None:
            rank = dist.get_rank() if is_distributed() else 0
        # Strided split: rank 0 takes 0, N, 2N - Every index appears on exactly one rank.
        self.indices = list(range(rank, len(dataset), num_replicas))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
