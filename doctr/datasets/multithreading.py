# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from typing import Callable, Any, Optional, Iterable


__all__ = ['multithread_exec']


def multithread_exec(func: Callable[[Any], Any], iter: Iterable[Any], threads: Optional[int] = None) -> Iterable[Any]:
    """Download a file accessible via URL with mutiple retries

    Example::
        >>> from doctr.datasets.multithreading import multithread_exec
        >>> entries = [1, 4, 8]
        >>> results = multithread_exec(lambda x: x ** 2, entries)

    Args:
        func: function to be executed on each element of the iterable
        iter: iterable
        threads: number of workers to be used for multiprocessing

    Returns:
        iterable of the function's results using the iterable as inputs
    """

    threads = threads if isinstance(threads, int) else min(16, mp.cpu_count())
    # Single-thread
    if threads < 2:
        results = map(func, iter)
    # Multi-threading
    else:
        with ThreadPool(threads) as tp:
            results = tp.map(func, iter)
    return results
