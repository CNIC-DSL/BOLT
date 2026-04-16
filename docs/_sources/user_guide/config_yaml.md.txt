# Configuration File (YAML) Guide

This page explains what fields a YAML configuration should contain, what each field means, and how to scale from “small verification runs” to “large grid experiments”.

## 1. Top-Level Fields Overview

A runnable configuration file typically includes the following top-level fields:

- `maps`  
  Declares which task (`gcd` or `openset`) each method belongs to.

- `methods`  
  The list of methods to run in this experiment.

- `datasets`  
  The list of datasets to run in this experiment.

- `result_file`  
  A naming suffix for summary/index files. The framework will generate:
  - `results/summary_{result_file}.csv`
  - `results/seen_index_{result_file}.json`

- `grid`  
  The grid search space. Each dimension is a list, and the framework takes the Cartesian product to generate combos.

- `run`  
  Runtime control parameters (GPU selection, concurrency, epochs, dry-run, OOM retry, etc.).

- `paths`  
  Output directory settings (e.g., `results_dir`, `logs_dir`).

- `per_method_extra`  
  Per-method injection/override of extra parameters (method-specific hyperparameters or switches).

Missing any required field will cause the configuration to be non-runnable.

## 2. Recommendations for the `grid` Field

Common dimension examples:
- `known_cls_ratio: [0.75]`
- `labeled_ratio: [0.5]`
- `fold_types: [fold]`
- `fold_nums: [5]`
- `fold_idxs: [0]`
- `seeds: [2025]`
- `cluster_num_factor: [1.0]`

Recommended practice:
- For the first pipeline verification, keep only one value for `fold_idxs` and one value for `seeds`.
- After confirming the pipeline works correctly, gradually increase `seeds` / `folds` / `ratios` to scale up.

## 3. Recommendations for the `run` Field

Common fields:
- `gpus: [0,1,2,3]`
- `max_workers: 4`
- `slots_per_gpu: 1`
- `num_pretrain_epochs: 1`
- `num_train_epochs: 1`
- `dry_run: false`
- `retry_on_oom: true`
- `max_retries: 2`
- `retry_backoff_sec: 15`
- `only_collect: false`

Recommended practice:
- For the first run, set `dry_run=true` to verify the number of generated combos and whether command construction matches your expectations.
- For concurrency, start with “one task per GPU”, then adjust `slots_per_gpu` based on your available GPU memory headroom.

## 4. Recommendations for the `paths` Field

Common settings:
- `results_dir: results`
- `logs_dir: logs`

In most cases, keeping these as relative paths is sufficient. Relative paths will be created under the working directory (`--output-dir`), which is convenient for archiving a run as a whole.

## 5. `per_method_extra`: Injecting Parameters per Method

`per_method_extra` is used to set parameters for a specific method only. Example:

per_method_extra:
  tan:
    backbone: bert-base-uncased
  ab:
    emb_name: sbert
    temperature: 0.07

Notes:
- Keys under `per_method_extra` must match the method names listed in `methods`.
- Injected fields are merged with the base `args` and passed to the method execution script, and are also written into `ARGS_JSON` in the logs for reproducibility.

## 6. Minimal Runnable Example
```yaml
maps:
  gcd: [tan]
  openset: []

methods: [tan]
datasets: [XTopic]
result_file: demo

grid:
  known_cls_ratio: [0.75]
  labeled_ratio: [0.5]
  fold_types: [fold]
  fold_nums: [5]
  fold_idxs: [0]
  seeds: [2025]
  cluster_num_factor: [1.0]

run:
  gpus: [0]
  max_workers: 1
  slots_per_gpu: 1
  num_pretrain_epochs: 1
  num_train_epochs: 1
  dry_run: false

paths:
  results_dir: results
  logs_dir: logs

per_method_extra: {}
```