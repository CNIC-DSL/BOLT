# 方法注册与执行流水线

bolt-lab 通过“方法注册表（registry）”将 method 名称映射到可执行脚本、默认配置与执行阶段（stages）。

本页说明：
- method 如何被解析为 gcd / openset
- stages 如何组织（单阶段或多阶段）
- 每次运行时参数如何传入子进程

## 1. 从 method 到 task

YAML 中的 maps 声明了方法与任务类型的对应关系：
- maps.gcd：属于 gcd 的方法集合
- maps.openset：属于 openset 的方法集合

框架会将 methods 中的每个方法映射为 task（gcd 或 openset），并选择对应的 registry。

## 2. Registry 的结构

每个方法在 registry 中对应一条记录，包含：
- task：gcd 或 openset
- stages：阶段列表
- config：该方法默认使用的配置文件路径（例如 configs/gcd/tan.yaml）
- output_base：该方法默认的输出根目录（例如 ./outputs/gcd/tan）

stages 是一个列表。每个 stage 包含：
- entry：该阶段调用的脚本入口（用于定位）
- cli_builder：用于拼接命令行参数的函数（最终生成子进程命令）

因此一个方法可以是：
- 单阶段：stages 只有 1 项（大多数 baseline 属于该类型）
- 多阶段：stages 有多项（例如先 pretrain 再 finetune 的流程）

## 3. 参数如何传入子进程

每次运行会构造一份 args_json（包含 dataset/ratio/fold/seed/gpu/epochs 等关键字段），并以两种方式传递：
1) 写入环境变量 ARGS_JSON（便于在日志中完整记录）
2) 通过命令行参数传入脚本（cli_builder 负责拼接）

日志文件的开头通常会包含：
- CMD：实际执行命令
- ARGS_JSON：该组合的完整参数 JSON

这使得单次实验的复现与排查具备明确依据。

## 4. BERT 模型选择规则（默认逻辑）

在部分任务中，cli_builder 会按数据集自动选择中文或英文 BERT 目录。例如：
- ecdt / thucnews：使用 bert-base-chinese
- 其他：使用 bert-base-uncased

在使用 bolt-grid 时，建议通过 --model-dir 指向你本机稳定的预训练模型目录，使得多次实验可以复用缓存与权重文件。

## 5. 扩展：新增一个方法需要做什么

一般需要完成以下事项：
1) 在对应 registry（gcd 或 openset）中新增方法条目：
   - 指定 task、config、output_base
   - 配置 stages 与 cli_builder
2) 确保方法脚本在执行结束后，将结果写入框架可收集的位置（见 auto_summary 页）
3) 在 YAML 的 maps 中将该方法加入对应任务集合，并在 methods 中引用该方法
4) 先用小网格 + dry_run 验证命令与参数，再扩大规模运行
