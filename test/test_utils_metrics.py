import pytest
from doctr.utils import metrics


@pytest.mark.parametrize(
    "gt, pred, ignore_case, ignore_accents, result",
    [
        [['grass', '56', 'True', 'STOP'], ['grass', '56', 'true', 'stop'], True, False, 1.0],
        [['grass', '56', 'True', 'STOP'], ['grass', '56', 'true', 'stop'], False, False, .5],
    ],
)
def test_exact_match(gt, pred, ignore_case, ignore_accents, result):
    metric = metrics.ExactMatch(ignore_case, ignore_accents)
    metric.update_state(gt, pred)
    assert metric.result() == result