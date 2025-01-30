doctr.utils
===========

This module regroups non-core features that are complementary to the rest of the package.

.. currentmodule:: doctr.utils


Visualization
-------------
Easy-to-use functions to make sense of your model's predictions.

.. currentmodule:: doctr.utils.visualization

.. autofunction:: visualize_page

Reconstitution
---------------

.. currentmodule:: doctr.utils.reconstitution

.. autofunction:: synthesize_page


.. _metrics:

Task evaluation
---------------
Implementations of task-specific metrics to easily assess your model performances.

.. currentmodule:: doctr.utils.metrics

.. autoclass:: TextMatch

   .. automethod:: update
   .. automethod:: summary

.. autoclass:: LocalizationConfusion

   .. automethod:: update
   .. automethod:: summary

.. autoclass:: OCRMetric

   .. automethod:: update
   .. automethod:: summary

.. autoclass:: DetectionMetric

   .. automethod:: update
   .. automethod:: summary
