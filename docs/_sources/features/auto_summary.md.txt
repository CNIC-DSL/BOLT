# Automatic Result Collection and Summarization

After each experiment combo finishes, **bolt-lab** automatically collects the corresponding result record and appends it to a unified summary table, making later comparison and analysis easier.

## 1. Result File Convention

By default, the framework reads per-method result files from:
./results/{task}/{method}/results.csv

Where:
- `task` is `gcd` or `openset`
- `method` is the method name (e.g., `tan`, `plm_ood`, `ab`, etc.)

The framework reads the **last row** of this CSV as the “latest result record”.

Notes:
- If `results.csv` does not exist or is empty, the framework will not write anything to the summary table and will print a message.
- Therefore, each method script must ensure it writes to the location above after finishing, and that the CSV includes the key metric columns.

## 2. Summary File Location and Naming

The summary file is written under `paths.results_dir` from the YAML:
{results_dir}/summary_{result_file}.csv

An index file is also generated:
{results_dir}/seen_index_{result_file}.json

Here, `result_file` comes from the top-level YAML field `result_file`.

## 3. What Fields Are Included in the Summary Table

At minimum, the summary table includes:
- `method`, `dataset`
- `known_cls_ratio`, `labeled_ratio`, `cluster_num_factor`, `seed`
- metric columns (vary by task type)
- `args`: a JSON string recording key run parameters (used for reproducibility and deduplication)

In general:
- Common GCD metrics include `ACC`, `H-Score`, `K-ACC`, `N-ACC`, `ARI`, `NMI`, etc.
- Common OpenSet metrics include `F1`, `K-F1`, `N-F1`, etc.

## 4. Common Issue: Why Nothing Was Written to the Summary

Troubleshoot in the following order:
1) Did the sub-task fail?
   Check the stage logs for that combo under `logs/` and confirm whether there is an error stack trace or any non-zero return code.
2) Was `results.csv` generated?
   Confirm that `./results/{task}/{method}/results.csv` exists and that the CSV has at least one row of result data.
3) Are the result columns complete?
   If the columns written by the method script differ significantly from what the framework expects, the summary may be missing fields or metrics may be empty. It is recommended to standardize column names on the method side, or to fill in the required fields in the method outputs.

## 5. Usage Tips

- When comparing methods, use `summary_{result_file}.csv` directly as the primary analysis entry point.
- If you want to distinguish multiple large runs, change `result_file` to generate separate summary files and avoid overwriting each other.
