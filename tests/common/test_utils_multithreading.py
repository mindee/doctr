import pytest

from doctr.utils.multithreading import multithread_exec


@pytest.mark.parametrize(
    "input_seq, func, output_seq",
    [
        [[1, 2, 3], lambda x: 2 * x, [2, 4, 6]],
        [[1, 2, 3], lambda x: x ** 2, [1, 4, 9]],
        [
            ['this is', 'show me', 'I know'],
            lambda x: x + ' the way',
            ['this is the way', 'show me the way', 'I know the way']
        ],
    ],
)
def test_multithread_exec(input_seq, func, output_seq):
    assert multithread_exec(func, input_seq) == output_seq
    assert list(multithread_exec(func, input_seq, 0)) == output_seq
