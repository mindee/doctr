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
