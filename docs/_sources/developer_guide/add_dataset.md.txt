# Integrating a New Dataset

This page explains how to integrate a new dataset at the level of a **dataset name (a string identifier)**, so it can be referenced in YAML and correctly loaded by different methods.

## I. Basic Conventions

1) `dataset` is a string identifier
- You write the dataset name in `datasets: [...]` in YAML
- Method scripts and configurations must be able to recognize this name and load the corresponding data

2) Default `data_dir`
- When constructing `args_json`, the framework sets `data_dir=./data`
- The exact data-reading logic is determined by the method script: some read paths from config, others read paths from CLI arguments

## II. Recommended Implementation Approaches

### Option A: Organize by Directory Structure (Recommended)

1) Create a directory for the new dataset under `./data`, for example:
./data/my_dataset/

2) Add dataset-specific configuration in the method config (recommended)  
In `configs/<task>/<method>.yaml`, add `dataset_specific_configs` if needed:

dataset_specific_configs:
  my_dataset:
    data_path: ./data/my_dataset
    max_length: 128
    num_labels: 20

Advantages:
- No hard-coded changes are needed in the runner layer
- Different methods can use different configurations for the same dataset

### Option B: Dispatch by Dataset Name Inside the Method Script

This is suitable when data-loading logic is complex and must be controlled in code.
It is recommended to consolidate dispatch logic into a unified data-loading module to avoid duplicating implementations across baselines.

## III. Notes Related to Model Selection

Some `cli_builder` implementations choose a Chinese or English BERT based on the dataset name.
If your new dataset is not included in the default Chinese/English list, it is recommended to explicitly:
- set `backbone` / `bert_model` paths in the config (if the method supports it), or
- provide a configurable `bert_model` parameter entry in the method script

## IV. Integration Verification (Recommended Workflow)

1) Choose a simple method (single-stage, clear logs)
2) Run only:
- `datasets: [my_dataset]`
- `methods: [tan or ab, etc.]`
- `seeds: [single value]`
- `fold_idxs: [single value]`
- set epochs to a small value
3) Check:
- whether you see normal loading and training behavior under `logs/`
- whether expected artifacts are produced under `outputs/`
- whether `results/{task}/{method}/results.csv` is created and contains the final result row
- whether `summary_{result_file}.csv` gets a new appended record

## V. Common Issues

1) Data files not found
- First check whether `data_path` is correct (and whether the config/arguments are actually taking effect)
- Then check whether the method script correctly parses the dataset name and maps it to the data directory

2) Inconsistent behavior across methods on the same dataset
- Use `dataset_specific_configs` to explicitly set key parameters (`max_length`, `num_labels`, text field names, etc.)
- Avoid relying on method-internal defaults, which can make differences hard to trace
