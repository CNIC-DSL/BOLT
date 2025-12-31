# Extending Grid Dimensions

This page explains how to add a new sweepable dimension (a grid dimension), such as `backbone`, `temperature`, `reg_loss`, etc.

In the current implementation, there are two possible paths:
- Path A: The dimension is only relevant to one (or a few) methods — use `per_method_extra` (no runner changes needed)
- Path B: The dimension must participate in the global Cartesian product — you must modify the runner and `args_json` (code changes required)

## I. Path A: Inject via `per_method_extra` (Recommended First)

Applicable when:
- The new parameter only affects one or a small number of methods
- You do not require this parameter to participate in the “global combo count calculation”
- You want to integrate it with minimal changes

How to do it:
1) Add `per_method_extra` for the target method(s) in YAML:
per_method_extra:
  plm_ood:
    reg_loss: cosine
  tan:
    backbone: ./pretrained_models/bert-base-uncased

2) Read the field from `args_json` in `cli_builder` or the method script and convert it into CLI arguments
- `args_json` merges fields from `method_specs[method]` in `make_base_args`
- The field will be recorded in `ARGS_JSON` in logs for reproducibility

Advantages:
- No need to change the combo-generation logic in `run_grid.py`
- Does not affect other methods

## II. Path B: Add a Real Grid Dimension (Code Changes Required)

Applicable when:
- The parameter must participate in grid expansion to form a full Cartesian-product combo set
- You want to compare and analyze results by this dimension in the summary table
- The dimension is shared across multiple methods and you want unified control

You need to make the following changes:

### Step 1: Add a new dimension list under `grid` in YAML

Example:
grid:
  ...
  backbone: ["bert-base-uncased", "bert-base-chinese"]

### Step 2: Modify the combo-expansion logic in `run_grid.py`

Currently, `run_grid.py` explicitly reads and expands fixed dimensions (`known_cls_ratio`, `labeled_ratio`, `fold_*`, `seeds`, `cluster_num_factor`).
Therefore you need to:
- read `grid["backbone"]` into a local variable
- add a `backbone` layer into the nested loops
- include `backbone` as part of the combo tuple and pass it into `worker` / `run_combo`

### Step 3: Write the new dimension into `args_json` in `utils.make_base_args`

Example:
args_json["backbone"] = backbone

Make sure:
- `cli_builder` can read the field and pass it to the method script
- `ARGS_JSON` contains the field (for reproducibility)

### Step 4: Update the duplicate-skip matching fields (if you rely on this mechanism)

The current implementation reads historical records from `results.csv` and filters by several fields.
If `backbone` is an important dimension that distinguishes experiments, include it in the matching field set to avoid treating different backbones as the same combo.

### Step 5: Ensure result collection and the summary reflect this dimension

It is recommended that the method records `backbone` in the `args` field (JSON) when writing `results.csv`, so the summary remains traceable.

## III. Recommended Development and Verification Order

1) Start with Path A (`per_method_extra`) to get the method running and writing `results.csv` and the summary correctly
2) Only proceed to Path B if the dimension truly needs to be global/grid-expanded
3) After each change, validate with a minimal grid:
- 1 method
- 1 dataset
- 1 seed
- 1 fold
- 1–2 values for the new dimension
Ensure the combo count and summary records match expectations

## IV. Common Issues

1) Combo count is not as expected
- Check whether the new dimension actually enters the nested loops in `run_grid.py`
- Check whether the combo tuple carries this value and passes it into `run_combo`

2) Parameter passing is missing
- Check whether `ARGS_JSON` at the beginning of stage logs includes the field
- If `ARGS_JSON` includes it but behavior does not change, check whether `cli_builder` converts it into real CLI arguments

3) The summary cannot distinguish different values
- Confirm that the field is recorded in the `args` (JSON) column in `results.csv`
- If you rely on the “skip duplicated combos” mechanism, confirm that the field is included in the matching logic
