# 配置文件（YAML）说明

本页面向使用者说明：一个 YAML 配置应包含哪些字段、每个字段的含义是什么、以及如何从“小规模验证”逐步扩展到“大规模网格实验”。

## 1. 顶层字段一览

一个可运行的配置文件通常包含以下顶层字段：

- maps
  声明方法属于哪个任务（gcd 或 openset）。

- methods
  本次运行的方法列表。

- datasets
  本次运行的数据集列表。

- result_file
  汇总文件的命名后缀。最终会生成：
  - results/summary_{result_file}.csv
  - results/seen_index_{result_file}.json

- grid
  网格搜索空间。每个维度为列表，框架做笛卡尔积生成 combos。

- run
  运行控制参数（GPU、并发、epochs、dry_run、OOM 重试等）。

- paths
  输出目录设置（results_dir、logs_dir）。

- per_method_extra
  按方法注入/覆写额外参数（用于方法特定超参或开关）。

缺少任一必需字段会导致配置无法运行。

## 2. grid 字段建议

常用维度示例：
- known_cls_ratio: [0.75]
- labeled_ratio: [0.5]
- fold_types: [fold]
- fold_nums: [5]
- fold_idxs: [0]
- seeds: [2025]
- cluster_num_factor: [1.0]

建议实践：
- 首次验证流程时，fold_idxs 与 seeds 尽量各保留 1 个值。
- 确认流程正确后，再逐步增加 seeds / folds / ratios 扩大规模。

## 3. run 字段建议

常用字段：
- gpus: [0,1,2,3]
- max_workers: 4
- slots_per_gpu: 1
- num_pretrain_epochs: 1
- num_train_epochs: 1
- dry_run: false
- retry_on_oom: true
- max_retries: 2
- retry_backoff_sec: 15
- only_collect: false

建议实践：
- 首次运行建议 dry_run=true，先检查组合数量与命令拼接是否符合预期。
- 并发策略建议从“每卡 1 个任务”开始，再根据显存余量调整 slots_per_gpu。

## 4. paths 字段建议

常见设置：
- results_dir: results
- logs_dir: logs

通常保持为相对路径即可。相对路径会落在工作目录（--output-dir）下，便于整体归档。

## 5. per_method_extra：按方法注入参数

per_method_extra 用于对某个方法单独设置参数，示例：

per_method_extra:
  tan:
    backbone: bert-base-uncased
  ab:
    emb_name: sbert
    temperature: 0.07

说明：
- per_method_extra 下的键必须与 methods 列表中的方法名一致。
- 注入的字段会与基础 args 合并后传给方法执行脚本，并写入日志中的 ARGS_JSON，便于复现。

## 6. 最小可运行示例

maps:
  gcd: [tan]
  openset: []

methods: [tan]
datasets: [XTopic]
result_file: demo

grid:
  known_cls_ratio: [0.75]
  labeled_ratio: [0.5]
  fold_types: [fold]
  fold_nums: [5]
  fold_idxs: [0]
  seeds: [2025]
  cluster_num_factor: [1.0]

run:
  gpus: [0]
  max_workers: 1
  slots_per_gpu: 1
  num_pretrain_epochs: 1
  num_train_epochs: 1
  dry_run: false

paths:
  results_dir: results
  logs_dir: logs

per_method_extra: {}
