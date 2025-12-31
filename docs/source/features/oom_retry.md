# OOM 重试机制

在 GPU 训练中，CUDA OOM（显存不足）属于常见失败原因。bolt-lab 提供了可控的 OOM 自动重试机制，用于提高大规模网格实验的完成率。

## 1. 触发条件

当子任务抛出 RuntimeError 且错误信息包含：
- "CUDA out of memory"
- 或 "out of memory"
框架会将其判定为 OOM。

## 2. 可配置参数（run 节）

- retry_on_oom（默认 true）
  是否启用 OOM 重试。

- max_retries（默认 2）
  最多重试次数。

- retry_backoff_sec（默认 15.0）
  每次重试前等待的秒数，用于释放显存与缓冲。

示例：
run:
  retry_on_oom: true
  max_retries: 3
  retry_backoff_sec: 30

## 3. 重试行为说明

- OOM 发生后，框架会等待 backoff 秒数，然后重新启动该组合。
- 若配置了多张 GPU，重试时可能会领取到不同的 GPU token，从而实现“换卡重试”的效果。
- 若达到 max_retries 仍失败，则该组合会被视为失败并终止（需要人工排查或调整配置后再重跑）。

## 4. 处理 OOM 的优先建议

如果同一组合持续 OOM，建议优先按以下顺序处理：
1) 降低并发：减少 max_workers 或 slots_per_gpu
2) 降低训练开销：减小 batch / max_length / num_train_epochs
3) 调整模型或 backbone：更换更小的 backbone 或关闭部分耗显存功能
4) 使用更大显存的 GPU 或减少同卡任务数
