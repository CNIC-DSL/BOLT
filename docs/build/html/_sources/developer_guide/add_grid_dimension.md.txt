# 扩展网格维度

本文说明如何增加一个新的可扫描维度（grid dimension），例如 backbone、temperature、reg_loss 等。

在当前实现中，你有两种实现路径：
- 路径 A：该维度仅与某个 method 相关，使用 per_method_extra 即可（无需改 runner）
- 路径 B：该维度需要参与全局网格笛卡尔积，必须改 runner 与 args_json（需要改代码）

一、路径 A：用 per_method_extra 注入（推荐优先）

适用场景：
- 新参数只对某一个/少数方法有效
- 不要求该参数参与“全局组合数计算”
- 你希望以最小改动完成接入

做法：
1）在 YAML 中为目标方法增加 per_method_extra：
per_method_extra:
  plm_ood:
    reg_loss: cosine
  tan:
    backbone: ./pretrained_models/bert-base-uncased

2）在 cli_builder 或方法脚本中从 args_json 读取该字段并转成命令行参数
- args_json 会在 make_base_args 中合并 method_specs[method] 的字段
- 日志里的 ARGS_JSON 会记录该字段，便于复现

优点：
- 不需要改 run_grid.py 的组合生成逻辑
- 不会影响其他方法

二、路径 B：增加真正的 grid 维度（需要改代码）

适用场景：
- 该参数需要参与网格展开，形成完整的笛卡尔积组合
- 你希望在 summary 中按该维度进行对比分析
- 该维度对多个方法通用，且希望统一控制

你需要完成以下改动：

步骤 1：在 YAML 的 grid 中增加新维度列表
例如：
grid:
  ...
  backbone: ["bert-base-uncased", "bert-base-chinese"]

步骤 2：修改 run_grid.py 的组合展开逻辑
当前 run_grid.py 是显式读取并展开固定维度（known_cls_ratio、labeled_ratio、fold_*、seeds、cluster_num_factor）。
因此你需要：
- 读取 grid["backbone"] 到本地变量
- 在嵌套循环中加入 backbone 这一层
- 将 backbone 作为 combo 元组的一部分传入 worker / run_combo

步骤 3：在 utils.make_base_args 中把新维度写入 args_json
例如：
args_json["backbone"] = backbone

确保：
- cli_builder 能读取该字段并传入方法脚本
- ARGS_JSON 中包含该字段（便于复现）

步骤 4：更新“重复运行跳过”的匹配字段（如你依赖该机制）
当前实现会从 results.csv 中读取历史记录并按若干字段过滤。
如果 backbone 是区分实验的重要维度，应将其纳入匹配字段集合，避免不同 backbone 被误判为同一组合。

步骤 5：确保结果收集与 summary 中能体现该维度
建议方法在写入 results.csv 时把 backbone 记录在 args 字段中（JSON），这样 summary 侧可追溯。

三、建议的开发与验证顺序

1）先用路径 A（per_method_extra）把方法跑通并能写出 results.csv 与 summary
2）确认该维度确实需要全局网格化，再走路径 B
3）每次改动后用最小网格验证：
- 1 method
- 1 dataset
- 1 seed
- 1 fold
- 1~2 个新维度取值
确保组合数与 summary 记录符合预期

四、常见问题

1）组合数不符合预期
- 检查 run_grid.py 中新维度是否真的进入嵌套循环
- 检查 combos 元组是否携带了该值并传到了 run_combo

2）参数传递丢失
- 查看 stage 日志开头 ARGS_JSON 是否包含该字段
- 如 ARGS_JSON 有但方法行为未变化，检查 cli_builder 是否把它转成了实际命令行参数

3）summary 无法区分不同取值
- 确认 results.csv 的 args 中记录了该字段
- 如依赖“重复运行跳过”机制，确认该字段被纳入匹配逻辑
