.. Bolt documentation master file.
.. title:: Bolt v0.1.0
.. image:: assets/logo.png

=========================================================

`Home <https://your-homepage.example/>`_ | `Docs <https://your-docs.example/>`_ | `GitHub <https://github.com/your-org/your-repo>`_ | `Paper <https://your-paper-link.example/>`_

Introduction
-------------------------
Bolt is an experiment runner and benchmarking toolkit for category discovery and open-set evaluation.
It provides a YAML-driven grid specification, a method registry with multi-stage pipelines, GPU-aware
parallel scheduling, robust OOM retry, and automatic result collection into summary tables.

Bolt organizes experiments around:

- **Task**: ``gcd`` (generalized category discovery) or ``openset`` (open-set / OOD scenarios)
- **Method**: a registered pipeline (1 or 2 stages) with a dedicated CLI builder
- **Dataset**: the target dataset identifier
- **Grid**: combinatorial factors such as ratios, folds, seeds, and cluster factor

The overall workflow is illustrated as follows:

.. image:: assets/pipeline.png
   :width: 700
   :align: center

.. raw:: html

   <br/>

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/install
   get_started/quick_start
   get_started/pretrained_models
   get_started/outputs

.. toctree::
   :maxdepth: 1
   :caption: Framework Features

   features/yaml_experiment_spec
   features/method_registry
   features/gpu_scheduler
   features/oom_retry
   features/auto_summary
   features/reproducibility

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user_guide/concepts
   user_guide/config_yaml
   user_guide/datasets
   user_guide/run_grid
   user_guide/outputs_and_logs
   user_guide/metrics

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/architecture
   developer_guide/add_method
   developer_guide/add_dataset
   developer_guide/add_grid_dimension

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
