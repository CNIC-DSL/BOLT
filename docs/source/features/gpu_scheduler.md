# GPU 并发调度

bolt-lab 的并发模型以“GPU token（令牌）池”为核心：每个并发任务启动前会领取一个 token，任务结束后归还 token，从而控制并发量与 GPU 绑定关系。

## 1. 关键参数

- run.gpus
  指定可用 GPU 列表。例如 [0,1,2]。

- run.slots_per_gpu（默认 1）
  每张 GPU 提供多少个 token。slots_per_gpu=2 表示允许同一张 GPU 同时跑 2 个任务（适用于小模型/小 batch）。

- run.max_workers
  最大并发 worker 数。最终有效并发取决于：
  有效并发 = min(max_workers, GPU token 总数, 组合数)

## 2. 运行时行为

- 每个组合（combo）开始执行前，从 token 池中领取一个 gpu_id。
- 子进程启动时，会设置 CUDA_VISIBLE_DEVICES 绑定到领取到的 gpu_id。
- 组合执行结束后，归还 token，供后续组合继续使用。

该机制具有以下特性：
- 能够保证并发数量受控，不会无限并发占满显存
- 多 GPU 情况下会自然地分配任务到不同 GPU 上
- 通过 slots_per_gpu 可以在显存允许时提升吞吐

## 3. 推荐配置示例

1) 单卡、严格串行（最稳妥）
run:
  gpus: [0]
  max_workers: 1
  slots_per_gpu: 1

2) 多卡、每卡 1 个任务（常用默认）
run:
  gpus: [0,1,2,3]
  max_workers: 4
  slots_per_gpu: 1

3) 单卡多任务（仅在显存充足时使用）
run:
  gpus: [0]
  max_workers: 2
  slots_per_gpu: 2

## 4. dry_run 用于并发前检查

首次配置并发时，建议先设置 dry_run=true：
- 仅打印将要执行的命令与参数
- 用于确认方法入口、数据集、fold/seed 等组合是否符合预期
- 避免大规模组合启动后才发现配置错误
