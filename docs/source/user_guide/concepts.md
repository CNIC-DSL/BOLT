# 核心概念

本页统一解释 bolt-lab 文档中反复出现的关键术语，便于你快速理解配置与输出。

## 1. Task：gcd 与 openset

bolt-lab 当前支持两类任务（task）：
- gcd：Generalized Category Discovery（广义类别发现）
- openset：Open-set Recognition / Open-set Classification（开放集识别）

在配置文件中，方法（method）会被归类到某个 task 下，框架据此选择对应的执行入口与指标字段。

## 2. Method：方法（算法实现）

method 是你要运行的算法名，例如 tan、ab、plm_ood 等。
同一个 method 在不同数据集、不同划分与不同超参组合上可以运行多次。

方法与 task 的关系由 YAML 的 maps 字段定义：
- maps.gcd：属于 gcd 的方法集合
- maps.openset：属于 openset 的方法集合

## 3. Dataset：数据集

dataset 是数据集标识符（字符串）。
框架会把 methods × datasets × grid 的笛卡尔积展开为实验组合，并逐个调度运行。

## 4. Grid：网格搜索空间

grid 用来描述“要扫描哪些参数维度”。常见维度包括：
- known_cls_ratio：已知类别比例
- labeled_ratio：已标注比例
- fold_types / fold_nums / fold_idxs：数据划分方式与索引
- seeds：随机种子
- cluster_num_factor：簇数倍率（部分方法使用）

每个维度都是一个列表，框架会对所有维度做笛卡尔积。

## 5. Combo：实验组合（一次最小运行单元）

一个 combo 对应一次最小运行单元，通常由以下参数唯一确定：
- task、method、dataset
- known_cls_ratio、labeled_ratio、cluster_num_factor
- fold_type、fold_num、fold_idx
- seed
- 训练轮数（num_pretrain_epochs、num_train_epochs）
- 以及 per_method_extra 注入的额外参数

框架对每个 combo 产生日志、输出产物，并收集结果写入汇总表。

## 6. Stage：阶段（单方法多阶段流水线）

部分方法由多个阶段组成（例如先预训练再微调），每个阶段称为一个 stage。
框架会按顺序执行 stage1、stage2...，任何阶段失败都会导致该 combo 失败并停止后续阶段。

## 7. Workdir：工作目录（--output-dir）

运行时通过 --output-dir 指定工作目录。
框架会在该目录下组织本次运行的 logs/、results/、outputs/ 等产出，并保留与本次运行相对应的配置与环境快照，便于复现与归档。

## 8. Summary：汇总表（summary_{result_file}.csv）

每个 combo 完成后，框架会尝试收集该方法的最新结果记录，并追加写入汇总表：
results/summary_{result_file}.csv

汇总表是后续对比与分析的首选入口。
