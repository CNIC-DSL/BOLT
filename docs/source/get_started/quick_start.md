# Quick Start

Goal of this section: complete a reproducible run with the fewest steps, and clearly understand three key arguments: `--config`, `--output-dir`, and `--model-dir`.

## 0. (Optional) Initialize the Workspace Only (`--init-only`)

If you want to set up the working directory structure first (without running any experiments), use `--init-only`:

bolt-grid --config grid_gcd.yaml --output-dir ~/tmp --model-dir ~/code/bolt/pretrained_models --init-only

What it does:
- Creates/initializes the workspace under `--output-dir` (e.g., `logs/`, `results/`, `outputs/`, and linked directories like `code/`, `configs/`, `data/`, `pretrained_models/`)
- Useful for checking paths/permissions and confirming your workspace layout before launching a full run

Note:
- `--init-only` does **not** run combos, does **not** produce stage logs, and does **not** append new records into the summary table.

## 1. Run with a Single Command

After installation, run the following directly (example uses the GCD configuration):

bolt-grid --config grid_gcd.yaml --output-dir ~/tmp --model-dir ~/code/bolt/pretrained_models

Argument details:
- `--config`
  Specifies the configuration file. You can provide:
  1) A config filename (e.g., `grid_gcd.yaml`): the framework will look for a config with the same name in its built-in `configs` directory.
  2) An absolute path to a config file: used to run a config that you copied and modified yourself.
- `--output-dir`
  Specifies the working directory for this run. All logs (`logs/`), aggregated results (`results/`), and training/evaluation artifacts (`outputs/`) will be written under this directory.
- `--model-dir`
  Specifies the pretrained model and cache directory. It is recommended to point to a stable, readable/writable location so you can reuse already-downloaded model files.

## 2. Recommended First-Run Workflow

To avoid an overly large and time-consuming setup on your first run, use the following workflow:

Step 0 (optional): Initialize the workspace
bolt-grid --config /abs/path/to/your_config.yaml --output-dir ~/tmp/exp_001 --model-dir ~/code/bolt/pretrained_models --init-only

Step 1: Copy a config to your own directory
For example, copy an internal/built-in config out (choose the path according to your setup), then modify the copied version.

Step 2: Shrink the config to a “minimal runnable scale”
For the first run, it is recommended to keep only:
- 1 dataset
- 1 method
- 1 seed
- 1 fold
- Set training epochs to a small value (to validate the pipeline)

Step 3: Run your copied config using an absolute path
bolt-grid --config /abs/path/to/your_config.yaml --output-dir ~/tmp/exp_001 --model-dir ~/code/bolt/pretrained_models

This ensures:
- The config source is always clear for each run;
- Different experiments do not overwrite each other;
- The output directory structure stays consistent, making comparison and cleanup easier.

## 3. How to Tell Whether the Run Succeeded

During execution:
- The standard output will continuously print progress and key messages;
- If an error occurs, it usually tells you to check the log directory for detailed error information.

After completion (a real run):
- You should see directories such as `logs/`, `results/`, and `outputs/` under `--output-dir`;
- You should be able to find stage logs under `logs/` for executed combos;
- The summary file under `results/` should append new records when combos successfully produce results.

After completion (init-only):
- You should see the workspace directory skeleton and linked directories under `--output-dir`;
- It is normal that there are no combo stage logs and no new summary records.

Next, open `outputs.md` to learn how to locate logs and summary results, and the recommended troubleshooting order.

## 4. Common Notes

- Make sure `--output-dir` points to the experiment root directory. Do not set it directly to an `outputs` subdirectory, otherwise you may create nested structures that make later searching and cleanup harder.
- It is recommended to keep `--model-dir` fixed to a long-term directory to avoid repeatedly preparing models and caches.
