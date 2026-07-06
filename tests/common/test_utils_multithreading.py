import os
from multiprocessing.pool import ThreadPool
from unittest.mock import patch

import pytest

from doctr.utils.multithreading import multithread_exec


@pytest.mark.parametrize(
    "input_seq, func, output_seq",
    [
        [[1, 2, 3], lambda x: 2 * x, [2, 4, 6]],
        [[1, 2, 3], lambda x: x**2, [1, 4, 9]],
        [
            ["this is", "show me", "I know"],
            lambda x: x + " the way",
            ["this is the way", "show me the way", "I know the way"],
        ],
    ],
)
def test_multithread_exec(input_seq, func, output_seq):
    assert list(multithread_exec(func, input_seq)) == output_seq
    assert list(multithread_exec(func, input_seq, 0)) == output_seq


@patch.dict(os.environ, {"DOCTR_MULTIPROCESSING_DISABLE": "TRUE"}, clear=True)
def test_multithread_exec_multiprocessing_disable():
    with patch.object(ThreadPool, "map") as mock_tp_map:
        multithread_exec(lambda x: x, [1, 2])
    assert not mock_tp_map.called


def test_multithread_exec_empty_sequence():
    assert list(multithread_exec(lambda x: x, [])) == []


def test_multithread_exec_skips_pool_for_single_item():
    # A pool brings no benefit for a single item: it must not be spawned even with many threads
    with patch.object(ThreadPool, "map") as mock_tp_map:
        assert list(multithread_exec(lambda x: 2 * x, [3], threads=16)) == [6]
    assert not mock_tp_map.called


def test_multithread_exec_uses_pool_for_multiple_items():
    with patch.object(ThreadPool, "map", return_value=[2, 4]) as mock_tp_map:
        assert list(multithread_exec(lambda x: 2 * x, [1, 2], threads=16)) == [2, 4]
    assert mock_tp_map.called
