# OOM Retry Mechanism

In GPU training, CUDA OOM (out-of-memory) is a common failure mode. bolt-lab provides a controllable automatic OOM retry mechanism to improve the completion rate of large-scale grid experiments.

## 1. Trigger Conditions

If a subprocess raises a `RuntimeError` and the error message contains:
- "CUDA out of memory"
- or "out of memory"
the framework will treat it as an OOM event.

## 2. Configurable Parameters (in `run`)

- `retry_on_oom` (default: true)  
  Whether to enable OOM retries.

- `max_retries` (default: 2)  
  Maximum number of retry attempts.

- `retry_backoff_sec` (default: 15.0)  
  Seconds to wait before each retry, to allow GPU memory and buffers to be released.

Example:
run:
  retry_on_oom: true
  max_retries: 3
  retry_backoff_sec: 30

## 3. Retry Behavior

- After an OOM occurs, the framework waits for the backoff seconds and then restarts the combo.
- If multiple GPUs are configured, a retry may acquire a different GPU token, effectively enabling “retry on another GPU”.
- If the combo still fails after reaching `max_retries`, it is marked as failed and the pipeline stops for that combo (manual debugging or configuration adjustments are needed before rerunning).

## 4. Recommended Actions to Handle OOM (Priority Order)

If the same combo keeps hitting OOM, it is recommended to address it in the following order:
1) Reduce concurrency: decrease `max_workers` or `slots_per_gpu`
2) Reduce training cost: lower batch size / `max_length` / `num_train_epochs`
3) Adjust the model/backbone: switch to a smaller backbone or disable memory-heavy features
4) Use a GPU with more memory, or reduce the number of tasks running on the same GPU
