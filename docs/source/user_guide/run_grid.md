# 运行方式与常用操作

本页说明如何启动运行、如何控制规模与并发，以及建议的工作流程。

## 1. 基本运行命令

安装完成后，在任意目录均可运行：

bolt-grid --config grid_gcd.yaml --output-dir ~/tmp --model-dir ~/code/bolt/pretrained_models

参数说明：
- --config
  配置文件名称或绝对路径。
  传入类似 grid_gcd.yaml 时，会从包内自带的 configs 目录查找同名配置；
  也可以传入你自行复制修改后的配置文件绝对路径。

- --output-dir
  工作目录。框架会在该目录下生成 outputs/、results/、logs/ 等产出。

- --model-dir
  模型/缓存目录。建议指向一个稳定且可读写的位置，用于存放预训练模型与缓存。

## 2. 建议的运行流程

步骤 1：先复制一份配置文件到你的工作目录进行修改
不要把示例配置当作唯一来源长期直接修改。保留一份可版本管理的“你的配置副本”。

步骤 2：将网格缩小到最小规模，验证流程
建议首次只保留：
- 1 个 dataset
- 1 个 method
- 1 个 seed
- 1 个 fold_idx
- epochs 设为较小值

步骤 3：使用新的输出目录运行
例如：
bolt-grid --config /abs/path/to/your.yaml --output-dir ~/tmp/exp_001 --model-dir ~/code/bolt/pretrained_models

建议每次大实验使用独立 output-dir，便于归档、对比与清理。

## 3. dry_run：只打印不执行（建议首次使用）

在 YAML 的 run 中设置：
run:
  dry_run: true

dry_run 会打印每个组合将要执行的命令与关键参数，用于：
- 核对组合数是否正确
- 核对方法入口、数据集、fold/seed 是否符合预期
- 避免大规模启动后才发现配置错误

## 4. only_collect：仅收集与写汇总（按需）

在 YAML 的 run 中设置：
run:
  only_collect: true

该模式用于你确信输出结果已存在，只希望框架重新收集并更新 summary 的场景。
若结果文件不存在，则不会凭空产生结果记录。

## 5. 并发与 GPU 配置建议

- 典型默认：多卡每卡 1 个任务
  run:
    gpus: [0,1,2,3]
    max_workers: 4
    slots_per_gpu: 1

- 单卡串行：最稳妥
  run:
    gpus: [0]
    max_workers: 1
    slots_per_gpu: 1

- 单卡并发：仅在显存充足时使用
  run:
    gpus: [0]
    max_workers: 2
    slots_per_gpu: 2

并发的实际生效上限由三者共同决定：
min(max_workers, GPU 总槽位数, 组合数)

## 6. OOM 重试（可选）

在 YAML 的 run 中可设置：
- retry_on_oom: true
- max_retries: 2
- retry_backoff_sec: 15

适合大规模网格实验，但若某组合持续 OOM，仍应从降低并发、降低 batch/长度、减少 epochs 等方向调整配置。
