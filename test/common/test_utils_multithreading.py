import pytest

import numpy as np
from doctr.utils.multithreading import multithread_exec


@pytest.mark.parametrize(
    "input_list, fn, expected_list",
    [
        [[1, 2, 3], lambda x: 2 * x, [2, 4, 6]],
        [
            ['this is', 'show me', 'I know'],
            lambda x: x + ' the way',
            ['this is the way', 'show me the way', 'I know the way']
        ],
    ],
)
def test_multithread_exec(input_list, fn, expected_list):
    assert multithread_exec(fn, input_list) == expected_list
    assert multithread_exec(fn, input_list, 0) == expected_list
