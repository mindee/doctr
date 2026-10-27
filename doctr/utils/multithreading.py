# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from multiprocessing.pool import ThreadPool
from typing import Any

from doctr.file_utils import ENV_VARS_TRUE_VALUES

__all__ = ["multithread_exec", "chunked", "prefetch"]


def multithread_exec(func: Callable[[Any], Any], seq: Iterable[Any], threads: int | None = None) -> Iterator[Any]:
    """Execute a given function in parallel for each element of a given sequence

    >>> from doctr.utils.multithreading import multithread_exec
    >>> entries = [1, 4, 8]
    >>> results = multithread_exec(lambda x: x ** 2, entries)

    Args:
        func: function to be executed on each element of the iterable
        seq: iterable
        threads: number of workers to be used for multiprocessing

    Returns:
        iterator of the function's results using the iterable as inputs

    Notes:
        This function uses ThreadPool from multiprocessing package, which uses `/dev/shm` directory for shared memory.
        If you do not have write permissions for this directory (if you run `doctr` on AWS Lambda for instance),
        you might want to disable multiprocessing. To achieve that, set 'DOCTR_MULTIPROCESSING_DISABLE' to 'TRUE'.
    """
    threads = threads if isinstance(threads, int) else min(16, mp.cpu_count())
    items = seq if isinstance(seq, (list, tuple)) else list(seq)
    # Never spawn more workers than items - single-item calls skip pool startup entirely
    threads = min(threads, len(items))
    # Single-thread
    if threads < 2 or os.environ.get("DOCTR_MULTIPROCESSING_DISABLE", "").upper() in ENV_VARS_TRUE_VALUES:
        results = map(func, items)
    # Multi-threading
    else:
        with ThreadPool(threads) as tp:
            # ThreadPool's map function returns a list, but seq could be of a different type
            # That's why wrapping result in map to return iterator
            results = map(lambda x: x, tp.map(func, items))  # noqa: C417
    return results


def chunked(seq: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Split an iterable into consecutive lists of at most `size` elements, consuming it lazily

    >>> from doctr.utils.multithreading import chunked
    >>> list(chunked(range(5), 2))
    [[0, 1], [2, 3], [4]]

    Args:
        seq: iterable
        size: the maximum number of elements per chunk

    Returns:
        iterator of chunks
    """
    if size < 1:
        raise ValueError("the chunk size must be a positive integer")
    iterator = iter(seq)
    while chunk := list(islice(iterator, size)):
        yield chunk


_EXHAUSTED = object()


def prefetch(seq: Iterable[Any], enabled: bool = True) -> Iterator[Any]:
    """Iterate over an iterable while producing its next element in a background thread

    This overlaps the production of the next element (e.g. decoding the next pages of a document) with the
    processing of the current one (e.g. running the models). At most one element is produced ahead, and the
    iterable is only ever advanced by one thread at a time.

    >>> from doctr.utils.multithreading import prefetch
    >>> for item in prefetch(expensive_generator()):
    ...     process(item)

    Args:
        seq: iterable
        enabled: whether the elements are produced in a background thread

    Returns:
        iterator over the elements of the iterable

    Notes:
        Prefetching is disabled when 'DOCTR_MULTIPROCESSING_DISABLE' is set to 'TRUE'.
    """
    if not enabled or os.environ.get("DOCTR_MULTIPROCESSING_DISABLE", "").upper() in ENV_VARS_TRUE_VALUES:
        yield from seq
        return

    iterator = iter(seq)
    try:
        # On early exit, leaving the context waits for the element being produced, so that the iterable is
        # never closed while it is advanced in the background thread
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="doctr-prefetch") as executor:
            future = executor.submit(next, iterator, _EXHAUSTED)
            while (item := future.result()) is not _EXHAUSTED:
                future = executor.submit(next, iterator, _EXHAUSTED)
                yield item
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
