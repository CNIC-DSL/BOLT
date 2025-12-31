# YAML 实验规格

bolt-lab 的批量实验由一个 YAML 文件完整描述。该 YAML 主要由 7 个部分组成：maps、methods、datasets、grid、run、paths、per_method_extra。

本页说明这些字段的含义、取值方式，以及一份可直接复用的最小示例。

## 1. 顶层字段说明

1) maps
用于声明“方法属于哪类任务”，当前包含两类任务：
- gcd：GCD 任务的方法集合
- openset：OpenSet 任务的方法集合

maps 的作用是让框架能够将 method 解析为 task（gcd 或 openset），并选择正确的执行入口。

2) methods
本次要运行的方法列表。框架会根据 maps 过滤/识别有效方法，并按 grid 组合运行。

3) datasets
本次要运行的数据集列表。每个 dataset 会与 methods、grid 组合构成最终的实验组合。

4) result_file
汇总文件的命名后缀。最终会在 results_dir 下生成：
- summary_{result_file}.csv
- seen_index_{result_file}.json（用于记录已运行/已收集的组合信息）

5) grid
网格搜索空间。框架会对 grid 中的每个维度做笛卡尔积，生成所有组合（combo）。
常见维度包括：
- known_cls_ratio：已知类别比例
- labeled_ratio：标注比例
- fold_types / fold_nums / fold_idxs：交叉验证切分方式与索引
- seeds：随机种子
- cluster_num_factor：聚类簇数倍率（如方法使用该参数）

6) run
运行控制参数，常用包括：
- gpus：可用 GPU 列表（例如 [0,1,2]）
- max_workers：最大并发 worker 数
- slots_per_gpu：每张 GPU 同时允许占用的“槽位”数量（默认 1）
- dry_run：仅打印命令，不实际执行（默认 false）
- retry_on_oom：是否在 OOM 时重试（默认 true）
- max_retries：OOM 最大重试次数（默认 2）
- retry_backoff_sec：每次重试前等待秒数（默认 15）

此外还包括训练轮数：
- num_pretrain_epochs
- num_train_epochs

7) paths
输出路径设置：
- results_dir：汇总文件目录（例如 results）
- logs_dir：日志目录（例如 logs）

8) per_method_extra
按方法覆写/追加参数。该字段用于将额外参数注入到某个 method 的运行参数中，例如：
- 为某个方法单独指定 backbone、reg_loss、extra_flags 等
- 或者为 OpenSet LLM 方法指定 llm_model、llm_api_base 等

## 2. 最小示例（可直接复制）

以下示例展示一个最小可运行的 YAML 结构。首次建议将 methods/datasets/grid 都缩到最小规模用于验证流程。

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

## 3. 组合数如何计算

组合数 = methods 数量 × datasets 数量 ×
        known_cls_ratio × labeled_ratio × fold_types × fold_nums × fold_idxs × seeds × cluster_num_factor 的笛卡尔积大小。

当组合数较大时，建议先使用 dry_run 检查命令与参数是否符合预期，再逐步扩大网格规模。

## 4. per_method_extra 的使用示例

per_method_extra:
  plm_ood-llm:
    llm_ood: true
    llm_model: gpt-4.1-mini
    llm_temperature: 0.0
    llm_batch_size: 16
  plm_ood:
    reg_loss: cosine
  ab:
    emb_name: sbert
    extra_flags: ["--some_flag", "1"]
