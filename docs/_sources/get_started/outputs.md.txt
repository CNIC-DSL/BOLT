# Output Directory Structure and Notes

This section explains where **bolt-lab** writes its runtime outputs, what directories are typically created after an experiment, and recommended usage patterns to help you quickly locate logs and results.

## 1. Meaning of the Working Directory (`--output-dir`)

During execution, you specify a **working directory** via `--output-dir`. The framework will create the runtime environment and output folders for the current experiment under this directory, and will write **logs, aggregated results, and training/evaluation artifacts** into it. This makes it easy to migrate, package, and reproduce runs.

It is recommended to point `--output-dir` to a dedicated experiment root directory (e.g., `~/tmp/exp_001`), rather than directly to subfolders such as `outputs/` or `logs/`.

## 2. Directories Typically Created After a Run

Using `--output-dir ~/tmp` as an example, after running you will see something like the following (example structure):

~/tmp/
  outputs/            Training/evaluation artifacts (models, predictions, caches, etc.)
  results/            Aggregations and index files (e.g., summary, seen_index, etc.)
  logs/               Runtime logs (for debugging errors and tracking stage progress)
  code/               Snapshot of the code used for this run (records code state)
  configs/            Snapshot of the config used for this run (records config state)
  data/               Data entry directory (may be empty or filled by your data pipeline)
  pretrained_models/  Model entry directory (corresponds to `--model-dir`)

Notes:
- Focus first on `logs/`, `results/`, and `outputs/`.
- Directories such as `code/` and `configs/` are used to record the environment state for the run, making reproduction and tracking easier.

## 3. Recommended Usage

### 3.1 How to Modify Configuration Files

It is recommended **not** to directly edit files under `configs/` inside the working directory as your long-term configuration source.

A more robust approach:
- Copy the example configs to your own directory and modify them there.
- Pass the absolute path of that config file at runtime via `--config`.

This avoids accidental overwrites between experiments and is easier to manage with version control.

### 3.2 Where to Look First to Diagnose Issues

- Runtime errors, hangs, or stage failures: check `logs/` first. You can usually find the error stack trace and the failed stage there.
- Experiment summary results: check `results/` (typically contains summary files, indexes, etc.).
- Model and prediction artifacts: check `outputs/`.

## 4. Common Issues and How to Avoid Them

### 4.1 Avoid Nested Output Directories

Make sure `--output-dir` is the experiment root directory.
Do not set `--output-dir` to something like `.../outputs`, otherwise you may end up with nested structures such as `outputs/outputs`, which makes searching and cleanup difficult.

### 4.2 About `--model-dir` (Additional Notes)

`--model-dir` is used to specify the model/cache directory (for reading/writing pretrained models and related caches).
The `pretrained_models/` directory under the working directory serves as the container for this entry point. It is recommended to set `--model-dir` to a stable, readable/writable location to reduce repeated downloads and preparation costs.
