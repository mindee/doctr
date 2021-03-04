from doctr.utils import metrics


def test_exact_match():
    mock_gt = ['grass', '56', 'True', 'STOP']
    mock_pred = ['grass', '56', 'true', 'stop']
    metric_a = metrics.ExactMatch(ignore_case=True, ignore_accents=False)
    metric_b = metrics.ExactMatch(ignore_case=False, ignore_accents=False)
    metric_a.update_state(mock_gt, mock_pred)
    metric_b.update_state(mock_gt, mock_pred)
    assert metric_a.result() == 1.0
    assert metric_b.result() == 0.5
