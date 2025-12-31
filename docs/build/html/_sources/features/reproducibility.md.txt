# 可复现性与重复运行控制

大规模网格实验常见需求包括：
- 能够准确复现实验
- 避免重复运行同一组合
- 能够在需要时强制重跑

本页说明 bolt-lab 在这三方面的设计与推荐实践。

## 1. 可复现性的基础要素

每个实验组合由以下参数共同定义：
- method / dataset
- known_cls_ratio / labeled_ratio / cluster_num_factor
- fold_type / fold_num / fold_idx
- seed
- 训练轮数（num_pretrain_epochs / num_train_epochs）
- 以及 per_method_extra 注入的额外参数（如 backbone、reg_loss、extra_flags 等）

运行时，这些信息会被记录为 ARGS_JSON，并写入对应 stage 的日志开头，作为复现依据。

## 2. 输出隔离的推荐做法

建议每次大实验使用独立的 --output-dir，例如：
- ~/tmp/exp_gcd_001
- ~/tmp/exp_openset_20251225

这样可以确保：
- logs/results/outputs 互不干扰
- 运行环境与配置副本与本次实验对应
- 便于整体打包、迁移与归档

同时建议：
- 将使用的 YAML 配置保存在版本管理中（或与 output-dir 一起归档）
- 使用固定的 --model-dir 复用模型缓存

## 3. 避免重复运行：已有记录的跳过

当框架检测到某方法已有 results.csv，并且其中存在与当前组合参数匹配的记录时，会跳过该组合的执行，以避免重复计算。

建议将“是否跳过重复组合”的判断视为默认行为：在大规模实验中能显著节省时间与资源。

## 4. 如何强制重跑

若你需要重跑同一组合（例如代码更新、修复 bug、改了方法实现），建议使用以下方式之一：

1) 更换 result_file
通过更换 YAML 的 result_file，生成新的 summary 与索引文件，便于与旧结果区分。

2) 使用新的 --output-dir
将本次运行输出写入新的工作目录，避免覆盖旧目录下的产物与日志。

3) 清理目标方法的 results.csv
在明确知道影响范围的情况下，可以清理：
./results/{task}/{method}/results.csv
然后重跑对应组合，使框架重新收集结果。

在实际使用中，优先推荐前两种方式（更安全、可追踪）。
