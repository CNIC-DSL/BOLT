# 指标说明

本页解释汇总表中常见指标的含义与阅读方式。不同 task 的指标字段不同，以下为常用约定。

## 1. 汇总表字段结构

summary_{result_file}.csv 通常包含两类字段：
- 实验设置字段：method、dataset、known_cls_ratio、labeled_ratio、cluster_num_factor、seed（以及可能的 fold 信息、额外超参）
- 结果指标字段：随 task 不同而变化（例如 ACC、H-Score、F1 等）
- args：一段 JSON 字符串，记录本次运行的关键参数（用于复现与去重）

建议阅读方式：
- 横向比较：固定 dataset / ratios / folds，只比较 method 的指标差异
- 纵向比较：固定 method，比较 ratios / seeds / folds 的稳定性与趋势

## 2. GCD 常见指标（示例）

- ACC
  聚类/分类整体准确率（具体定义依方法实现而定，通常反映总体分配质量）。

- H-Score
  用于平衡“已知类（Known）”与“新类（Novel）”表现的综合指标，常用于综合评价。

- K-ACC
  已知类（Known classes）上的准确率。

- N-ACC
  新类（Novel classes）上的准确率。

- ARI（Adjusted Rand Index）
  衡量聚类结果与真实标签的一致性，考虑随机一致性的校正。

- NMI（Normalized Mutual Information）
  衡量聚类与真实标签的互信息一致性，范围通常在 [0,1]。

阅读建议：
- 若方法在 K-ACC 高但 N-ACC 低，通常意味着对新类发现能力不足；
- H-Score 往往更适合做综合排序，但仍需结合 K/N 的分解指标解释原因。

## 3. OpenSet 常见指标（示例）

- F1
  总体 F1（精确率与召回率的调和平均），用于综合评价。

- K-F1
  已知类（Known）上的 F1。

- N-F1
  未知/新类（Novel / Unknown）上的 F1。

阅读建议：
- K-F1 与 N-F1 往往存在权衡关系；
- 只看总体 F1 可能掩盖对未知类识别的不足，建议同时查看 K-F1 与 N-F1。

## 4. 与方法实现的关系

不同方法可能在“结果文件 results.csv”中写出更多列（例如 loss、时间、额外超参）。
框架汇总时通常会保留核心字段并附带 args 以保证可追溯性。
如你需要额外指标进入汇总表，优先在方法侧统一结果列名，并确保写入 results/{task}/{method}/results.csv。
