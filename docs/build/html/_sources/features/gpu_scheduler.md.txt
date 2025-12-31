# GPU Concurrency Scheduling

bolt-lab’s concurrency model is built around a **GPU token pool**: before a task starts, it acquires a token; after the task finishes, it returns the token. This controls both the concurrency level and the GPU binding behavior.

## 1. Key Parameters

- `run.gpus`  
  Specifies the list of available GPUs, e.g., `[0,1,2]`.

- `run.slots_per_gpu` (default: 1)  
  How many tokens each GPU provides. `slots_per_gpu=2` means the same GPU is allowed to run 2 tasks concurrently (useful for small models / small batch sizes).

- `run.max_workers`  
  The maximum number of concurrent workers. The effective concurrency is:
  effective_concurrency = min(max_workers, total_GPU_tokens, number_of_combos)

## 2. Runtime Behavior

- Before a combo starts, it acquires a `gpu_id` from the token pool.
- When the subprocess launches, it sets `CUDA_VISIBLE_DEVICES` to bind to the acquired `gpu_id`.
- After the combo finishes, it returns the token so later combos can reuse it.

This mechanism has the following properties:
- Concurrency is bounded, preventing unlimited parallelism that could exhaust GPU memory
- With multiple GPUs, tasks are naturally distributed across different GPUs
- `slots_per_gpu` can increase throughput when GPU memory allows

## 3. Recommended Configuration Examples

1) Single GPU, strictly serial (most stable)
run:
  gpus: [0]
  max_workers: 1
  slots_per_gpu: 1

2) Multiple GPUs, one task per GPU (common default)
run:
  gpus: [0,1,2,3]
  max_workers: 4
  slots_per_gpu: 1

3) Multiple tasks on a single GPU (only when memory is sufficient)
run:
  gpus: [0]
  max_workers: 2
  slots_per_gpu: 2

## 4. Use `dry_run` to Check Before Enabling Concurrency

When configuring concurrency for the first time, it is recommended to set `dry_run=true` first:
- It only prints the commands and parameters that would be executed
- It helps confirm whether method entry points, datasets, fold/seed combos, etc. match your expectations
- It avoids discovering configuration errors only after launching a large number of combos
