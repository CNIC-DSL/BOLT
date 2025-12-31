# Adding a New Method

This page explains how to integrate a new method into bolt-lab so that it can:
- be referenced by YAML configs and enter grid scheduling
- execute correctly (single-stage or multi-stage)
- be automatically collected after completion and written into the summary

## I. Integration Overview

Adding a new method typically involves three parts:
1) Method script (the actual training/evaluation code)
2) Method configuration (YAML under `configs/`)
3) Registration and command construction (`cli_gcd.py` or `cli_openset.py`)

## II. Step 1: Prepare the Method Script and Outputs

1) Location
- GCD methods are typically placed under: `code/gcd/...`
- OpenSet methods are typically placed under: `code/openset/...`

2) Output requirements (mandatory)
After the method finishes, it **must** write:
./results/{task}/{method}/results.csv

Requirements:
- The CSV must contain at least one row of results
- It is recommended to include an `args` column (a JSON string) for deduplication and traceability
- Metric columns are method-dependent, but it is recommended to keep them consistent with existing methods (for fair horizontal comparison)

## III. Step 2: Add a Method Configuration File

Add a default config for the new method:
- GCD: `configs/gcd/<method>.yaml`
- OpenSet: `configs/openset/<method>.yaml`

Recommendations:
- Put common parameters at the top level
- If different datasets require different parameters, add `dataset_specific_configs` (optional)

## IV. Step 3: Implement `cli_builder` (Command Builder Function)

Add a new command builder function in the corresponding file:
- GCD: `cli_gcd.py`
- OpenSet: `cli_openset.py`

Basic principles for command construction:
1) Read common fields from `args_json`:
- `dataset` / `known_cls_ratio` / `labeled_ratio`
- `fold_type` / `fold_num` / `fold_idx`
- `seed` / `gpu_id`
- `num_pretrain_epochs` / `num_train_epochs`
- `config` (default config path)
2) Read method-specific parameters from `args_json` as well (from `per_method_extra` or future grid dimensions)
3) Output must be an **argv list**, for example:
[sys.executable, "code/xxx/run.py", "--arg1", "v1", ...]

Recommended pattern:
- If your method can reuse common flags, reuse existing `_common_env` / `_common_flags` / `_epoch_flags` logic
- If extra flags are needed, define `extra_flags` (list) in `args_json` and append them in `cli_builder`

## V. Step 4: Register the Method in `METHOD_REGISTRY`

1) Choose the correct registry
- GCD: `METHOD_REGISTRY_GCD` (in `cli_gcd.py`)
- OpenSet: `METHOD_REGISTRY_OPENSET` (in `cli_openset.py`)

2) Add a new entry

"<method>": {
  "task": "gcd" or "openset",
  "stages": [
    {"entry": "<script path>", "cli_builder": <function_name>},
    ... (add more items for multi-stage methods)
  ],
  "config": "configs/<task>/<method>.yaml",
  "output_base": "./outputs/<task>/<method>",
}

Notes:
- The number of `stages` determines whether the method is single-stage or multi-stage
- `entry` is used to identify the stage entry point (mainly for readability and debugging)
- `output_base` is included in `args_json` (used for organizing output directories)

## VI. Step 5: Declare It in YAML and Test Run

1) Add the method into the corresponding task set under `maps`:
maps:
  gcd: [..., <method>]
  openset: [...]

2) Reference the method in `methods`:
methods: [<method>]

3) For the first run, use a minimal grid and `dry_run`
- Set `dry_run=true` first to verify commands and parameters
- Then set `dry_run=false` and run with 1 seed + 1 fold

## VII. Common Integration Issues

1) Run succeeds but the summary has no new record
- First check whether `./results/{task}/{method}/results.csv` was written
- Then check whether `results.csv` contains a readable last-row result

2) Parameters are not passed through
- Check `ARGS_JSON` at the beginning of `stageX.log`
- Confirm that `cli_builder` reads the corresponding fields from `args_json`
- If using `per_method_extra`, confirm the method name matches and field names are consistent
