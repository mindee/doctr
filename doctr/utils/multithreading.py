# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import multiprocessing as mp
import os
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Iterable, Optional

from doctr.file_utils import ENV_VARS_TRUE_VALUES

__all__ = ['multithread_exec']


def multithread_exec(func: Callable[[Any], Any], seq: Iterable[Any], threads: Optional[int] = None) -> Iterable[Any]:
    """Execute a given function in parallel for each element of a given sequence

    >>> from doctr.utils.multithreading import multithread_exec
    >>> entries = [1, 4, 8]
    >>> results = multithread_exec(lambda x: x ** 2, entries)

    Args:
        func: function to be executed on each element of the iterable
        seq: iterable
        threads: number of workers to be used for multiprocessing

    Returns:
        iterable of the function's results using the iterable as inputs

    Notes:
        This function uses ThreadPool from multiprocessing package, which uses `/dev/shm` directory for shared memory.
        If you do not have write permissions for this directory (if you run `doctr` on AWS Lambda for instance),
        you might want to disable multiprocessing. To achieve that, set 'DOCTR_MULTIPROCESSING_DISABLE' to 'TRUE'.
    """

    threads = threads if isinstance(threads, int) else min(16, mp.cpu_count())
    # Single-thread
    if threads < 2 or os.environ.get('DOCTR_MULTIPROCESSING_DISABLE', "").upper() in ENV_VARS_TRUE_VALUES:
        results = map(func, seq)
    # Multi-threading
    else:
        with ThreadPool(threads) as tp:
            results = tp.map(func, seq)  # type: ignore[assignment]
    return results
