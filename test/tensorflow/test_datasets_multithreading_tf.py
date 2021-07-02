import pytest
from doctr.datasets import multithreading


@pytest.mark.parametrize(
    "inputs, func, outputs",
    [
        [[1, 2, 3], lambda x: x**2, [1, 4, 9]],
    ],
)
def test_multithread_exec(inputs, func, outputs):
    out = multithreading.multithread_exec(func, inputs)
    assert list(out) == outputs
    singlethread_out = multithreading.multithread_exec(func, inputs, threads=1)
    assert list(singlethread_out) == outputs
