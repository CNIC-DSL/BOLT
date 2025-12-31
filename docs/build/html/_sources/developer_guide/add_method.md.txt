# 新增方法

本文说明如何在 bolt-lab 中接入一个新方法（method），使其能够：
- 被 YAML 配置引用并进入网格调度
- 正确执行（单阶段或多阶段）
- 运行结束后被框架自动收集结果并写入 summary

一、接入路径概览

新增方法通常涉及三个部分：
1）方法脚本（真正训练/评测的代码）
2）方法配置（configs 下的 YAML 配置）
3）注册与命令构建（cli_gcd.py 或 cli_openset.py）

二、步骤 1：准备方法脚本与输出

1）放置位置
- GCD 方法通常放在：code/gcd/...
- OpenSet 方法通常放在：code/openset/...

2）输出要求（强制）
方法运行结束后必须写入：
./results/{task}/{method}/results.csv

要求：
- CSV 至少包含一行结果
- 建议包含 args 字段（JSON 字符串），用于后续去重与追溯
- 指标列由方法决定，但建议与现有方法保持一致（便于横向对比）

三、步骤 2：新增方法配置文件

为新方法新增默认配置：
- GCD：configs/gcd/<method>.yaml
- OpenSet：configs/openset/<method>.yaml

建议：
- 将通用参数放在顶层
- 如对不同 dataset 需要不同参数，可增加 dataset_specific_configs 字段（可选）

四、步骤 3：实现 cli_builder（命令构建函数）

在对应文件中新增命令构建函数：
- GCD：cli_gcd.py
- OpenSet：cli_openset.py

命令构建的基本原则：
1）从 args_json 中读取通用字段：
- dataset / known_cls_ratio / labeled_ratio
- fold_type / fold_num / fold_idx
- seed / gpu_id
- num_pretrain_epochs / num_train_epochs
- config（默认配置路径）
2）将 method 特有参数也从 args_json 读取（来自 per_method_extra 或未来扩展字段）
3）输出必须是 “argv 列表”，例如：
[sys.executable, "code/xxx/run.py", "--arg1", "v1", ...]

建议模式：
- 若你的方法可复用通用参数集，可以复用已有的 _common_env / _common_flags / _epoch_flags 逻辑
- 若需要额外 flag，可在 args_json 中定义 extra_flags（list），并在 cli_builder 中追加

五、步骤 4：在 METHOD_REGISTRY 中注册方法

1）选择 registry
- GCD：METHOD_REGISTRY_GCD（cli_gcd.py）
- OpenSet：METHOD_REGISTRY_OPENSET（cli_openset.py）

2）新增条目结构

"<method>": {
  "task": "gcd" 或 "openset",
  "stages": [
    {"entry": "<脚本路径>", "cli_builder": <函数名>},
    ...（多阶段时继续添加）
  ],
  "config": "configs/<task>/<method>.yaml",
  "output_base": "./outputs/<task>/<method>",
}

说明：
- stags 数量决定该方法是单阶段还是多阶段
- entry 用于标识阶段入口（主要用于可读性与排查）
- output_base 会进入 args_json（用于输出目录组织）

六、步骤 5：在 YAML 中声明并试跑

1）在 YAML 的 maps 中把方法加入对应任务集合：
maps:
  gcd: [..., <method>]
  openset: [...]

2）在 methods 列表中引用该方法：
methods: [<method>]

3）首次建议使用最小网格并 dry_run
- 先 dry_run=true 检查命令与参数
- 再 dry_run=false 真跑 1 个 seed + 1 个 fold

七、常见接入问题

1）运行成功但 summary 没有新增
- 优先检查：是否写出了 ./results/{task}/{method}/results.csv
- 再检查：results.csv 是否有最后一行可读结果

2）参数没传进去
- 查看 stageX.log 开头的 ARGS_JSON
- 确认 cli_builder 是否从 args_json 读取了对应字段
- 如使用 per_method_extra 注入字段，确认 method 名称匹配且字段名一致
