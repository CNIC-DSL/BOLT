# Running and Common Operations

This page explains how to start runs, how to control scale and concurrency, and a recommended workflow.

## 1. Basic Run Command

After installation, you can run from any directory:

bolt-grid --config grid_gcd.yaml --output-dir ~/tmp --model-dir ~/code/bolt/pretrained_models

Argument details:
- `--config`  
  A configuration filename or an absolute path.  
  If you pass something like `grid_gcd.yaml`, the framework will look for a config with the same name under the package’s built-in `configs` directory.  
  You can also pass the absolute path to a config file that you copied and modified yourself.

- `--output-dir`  
  The working directory. The framework will create outputs such as `outputs/`, `results/`, and `logs/` under this directory.

- `--model-dir`  
  The model/cache directory. It is recommended to point to a stable, readable/writable location to store pretrained models and caches.

## 2. Recommended Workflow

Step 1: Copy a config file to your working directory and modify it  
Do not treat the example config as the only long-term source. Keep your own version-controlled copy.

Step 2: Shrink the grid to a minimal size to validate the pipeline  
For the first run, keep only:
- 1 dataset
- 1 method
- 1 seed
- 1 `fold_idx`
- Set epochs to a small value

Step 3: Run with a new output directory  
For example:
bolt-grid --config /abs/path/to/your.yaml --output-dir ~/tmp/exp_001 --model-dir ~/code/bolt/pretrained_models

It is recommended to use an independent `output-dir` for each large experiment to make archiving, comparison, and cleanup easier.

## 3. `dry_run`: Print Only, Do Not Execute (Recommended for the First Use)

Set the following in `run` inside your YAML:
run:
  dry_run: true

`dry_run` prints the command and key parameters that would be executed for each combo, which helps you:
- Verify the number of generated combos
- Verify method entry points, datasets, and fold/seed settings
- Avoid discovering configuration issues only after launching a large run

## 4. `only_collect`: Collect and Update Summary Only (As Needed)

Set the following in `run` inside your YAML:
run:
  only_collect: true

This mode is useful when you are confident the outputs already exist and you only want the framework to re-collect results and update the summary.
If result files do not exist, it will not “create” result records out of nothing.

## 5. Concurrency and GPU Configuration Tips

- Typical default: multiple GPUs, one task per GPU
  run:
    gpus: [0,1,2,3]
    max_workers: 4
    slots_per_gpu: 1

- Single-GPU serial: the most stable option
  run:
    gpus: [0]
    max_workers: 1
    slots_per_gpu: 1

- Single-GPU concurrent: only use when you have enough GPU memory
  run:
    gpus: [0]
    max_workers: 2
    slots_per_gpu: 2

The effective concurrency upper bound is determined jointly by:
min(max_workers, total GPU slots, number of combos)

## 6. OOM Retry (Optional)

In `run` inside your YAML, you can set:
- `retry_on_oom: true`
- `max_retries: 2`
- `retry_backoff_sec: 15`

This is useful for large grid experiments. However, if a combo keeps running into OOM, you should still adjust the configuration by reducing concurrency, lowering batch size/sequence length, reducing epochs, etc.
