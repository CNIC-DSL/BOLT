# Architecture and Execution Flow

This page is intended for developers who want to extend or modify the system. It explains bolt-lab’s core execution chain, key data structures, and the responsibility boundaries across modules.

## I. End-to-End Data Flow (From YAML to Summarized Results)

### 1) Entry Point: `bolt-grid`
- The CLI accepts arguments: `--config`, `--output-dir`, `--model-dir`
- After resolving the config file to a concrete path, it hands control to the runner (which reads the YAML and starts scheduling)

### 2) Runner: `run_grid.py`
- Reads the YAML and validates required fields:
  `maps / methods / datasets / grid / run / paths / result_file / per_method_extra`
- Calls `set_paths(results_dir, logs_dir, result_file)` to initialize:
  - `results/summary_{result_file}.csv`
  - `results/seen_index_{result_file}.json`
  - the `logs/` root directory
- Expands each dimension in `grid` into experiment combinations (combos), forming a task list `combos`
- Builds a GPU token pool (a `Queue`):
  - `run.gpus` specifies the list of available GPUs
  - `run.slots_per_gpu` determines how many tokens each GPU provides
  - total tokens = `len(gpus) * slots_per_gpu`
- Schedules tasks concurrently via `ThreadPoolExecutor`:
  - each worker acquires a `gpu_id` when starting
  - returns the `gpu_id` when finished
  - concurrency = `min(max_workers, total_tokens, number_of_combos)`

### 3) Single-Combo Execution: `utils.run_combo()`
- Looks up method definitions from the two registries:
  - `cli_gcd.METHOD_REGISTRY_GCD`
  - `cli_openset.METHOD_REGISTRY_OPENSET`
- Builds `args_json` (`make_base_args`):
  - includes key fields such as method/dataset/ratios/fold/seed/GPU/epochs
  - merges method-specific extra fields from `per_method_extra[method]`
- Executes stages:
  - each stage generates an argv command via `cli_builder(args_json, stage_idx)`
  - `run_stage()` is responsible for:
    - setting the `ARGS_JSON` environment variable (for logging and reproducibility)
    - setting `CUDA_VISIBLE_DEVICES` (binding to the acquired `gpu_id`)
    - writing `CMD` and `ARGS_JSON` at the beginning of the stage log
    - launching the subprocess and waiting for the return code
  - if any stage returns a non-zero code, the combo fails and subsequent stages will not run

### 4) Result Collection and Summarization: `collect_latest_result()` + `write_summary()`
- After all stages succeed, the framework attempts to read:
  `./results/{task}/{method}/results.csv`
  and uses the last row as the “latest result”
- If successful, it appends one row to:
  `results/summary_{result_file}.csv`
- If the result file is missing or empty, it prints a message like “Finished but no results found” and does not write to the summary

## II. Key Modules and Responsibility Boundaries

### 1) `run_grid.py` (Scheduling Layer)
- Responsible for: parsing YAML, expanding combos, concurrent scheduling, OOM retry policy (optional)
- Not responsible for: method-internal training logic or metric computation

### 2) `cli_gcd.py` / `cli_openset.py` (Method Registration and Command Building)
- Responsible for: mapping a method to stages, and providing `cli_builder` for each stage
- Responsible for: defining default config paths and default `output_base`
- Not responsible for: writing results during execution (this is handled by the method scripts)

### 3) `utils.py` (Execution and Summarization Layer)
- Responsible for: constructing `args_json`, running stages, organizing log directories, collecting results, and writing the summary
- Responsible for: writing `CMD` and `ARGS_JSON` for each stage (for reproducibility and debugging)

## III. Reproducibility Conventions

### 1) `ARGS_JSON` as the Minimal Reproducibility Loop
- At the beginning of each stage log, you will see:
  - `CMD` (the actual command executed)
  - `ARGS_JSON` (the full parameter JSON)
Therefore, reproducing an experiment can usually be done by directly copying the command and parameters from the logs.

### 2) Result File Path Convention
- Automatic summarization depends on the convention:
  `./results/{task}/{method}/results.csv`
Method scripts must write this file at the end of execution; otherwise, the framework cannot automatically collect and summarize results.

## IV. Common Debugging Techniques for Developers

### 1) `dry_run`
- Set `dry_run=true` under `run` in YAML to print commands and `ARGS_JSON` without actually executing anything
- Useful for checking combo counts, parameter passing, and whether command construction matches expectations

### 2) Reproduce from Stage Logs
- Locate `stageX.log` for the failed combo
- Copy `CMD` and `ARGS_JSON`, and reproduce/debug by aligning with the method script’s argument parsing logic
