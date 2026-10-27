# doctr.utils

Complementary tools: visualization, page reconstruction and evaluation metrics.

## Visualization

```{eval-rst}
.. currentmodule:: doctr.utils.visualization

.. autofunction:: visualize_page

.. autofunction:: visualize_kie_page

.. autofunction:: draw_boxes
```

## Reconstitution

```{eval-rst}
.. currentmodule:: doctr.utils.reconstitution

.. autofunction:: synthesize_page

.. autofunction:: synthesize_kie_page
```

(task-evaluation)=
## Task evaluation

Metrics to evaluate models on each task, used by the evaluation scripts of the references and in the
[model zoo](../models/index.md).

```{eval-rst}
.. currentmodule:: doctr.utils.metrics

.. autoclass:: TextMatch
   :members: update, summary

.. autoclass:: LocalizationConfusion
   :members: update, summary

.. autoclass:: OCRMetric
   :members: update, summary

.. autoclass:: DetectionMetric
   :members: update, summary

.. autoclass:: ObjectDetectionMetric
   :members: update, summary

.. autoclass:: TableCellMetric
   :members: update, summary
```
