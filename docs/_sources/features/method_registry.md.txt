# Method Registration and Execution Pipeline

bolt-lab maps a `method` name to an executable script, default configuration, and execution stages via a **method registry**.

This page explains:
- how a method is resolved to `gcd` / `openset`
- how stages are organized (single-stage or multi-stage)
- how parameters are passed to subprocesses for each run

## 1. From Method to Task

In YAML, `maps` declares the mapping between methods and task types:
- `maps.gcd`: the set of methods belonging to `gcd`
- `maps.openset`: the set of methods belonging to `openset`

The framework maps each method in `methods` to a task (`gcd` or `openset`) and selects the corresponding registry.

## 2. Registry Structure

Each method corresponds to one registry entry, which includes:
- `task`: `gcd` or `openset`
- `stages`: a list of stages
- `config`: the default config file path for the method (e.g., `configs/gcd/tan.yaml`)
- `output_base`: the default output root for the method (e.g., `./outputs/gcd/tan`)

`stages` is a list. Each stage typically contains:
- `entry`: the script entry point invoked by this stage (for locating/identifying the executable)
- `cli_builder`: a function that assembles command-line arguments (it ultimately generates the subprocess command)

Therefore, a method can be:
- Single-stage: `stages` has only 1 item (most baselines fall into this category)
- Multi-stage: `stages` has multiple items (e.g., a pipeline like pretrain → finetune)

## 3. How Parameters Are Passed to Subprocesses

For each run, the framework constructs an `args_json` containing key fields such as `dataset`, ratios, fold, seed, GPU, epochs, etc., and passes it in two ways:
1) Write to the environment variable `ARGS_JSON` (so it can be fully recorded in logs)
2) Pass to the script via command-line arguments (assembled by `cli_builder`)

At the beginning of a log file, you will typically see:
- `CMD`: the actual command executed
- `ARGS_JSON`: the full parameter JSON for that combo

This provides a clear basis for reproducing and debugging a single experiment run.

## 4. BERT Model Selection Rule (Default Logic)

In some tasks, `cli_builder` automatically chooses the Chinese or English BERT directory based on the dataset. For example:
- `ecdt` / `thucnews`: use `bert-base-chinese`
- otherwise: use `bert-base-uncased`

When using `bolt-grid`, it is recommended to point `--model-dir` to a stable pretrained model directory on your machine so that multiple experiments can reuse caches and weight files.

## 5. Extension: What You Need to Do to Add a New Method

In general, you need to complete the following:
1) Add a new method entry in the corresponding registry (`gcd` or `openset`):
   - specify `task`, `config`, and `output_base`
   - configure `stages` and `cli_builder`
2) Ensure the method script writes results to the location the framework can collect (see the `auto_summary` page)
3) Add the method to the corresponding task set in YAML `maps`, and reference it in `methods`
4) First validate commands and parameters with a small grid + `dry_run`, then scale up for large runs
