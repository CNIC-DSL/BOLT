# 接入新数据集

本文说明如何以“数据集名称（dataset 字符串）”为单位接入新数据集，使其能够被 YAML 引用并在各方法中正确加载。

一、基本约定

1）dataset 是一个字符串标识符
- YAML 的 datasets: [...] 中写入 dataset 名称
- 方法脚本与配置需要能够识别该名称并加载对应数据

2）默认 data_dir
- 框架构造 args_json 时会设置 data_dir=./data
- 具体如何读数据由方法脚本决定：有的从 config 中读路径，有的从命令行参数读路径

二、推荐的落地方式

方式 A：以目录结构组织（推荐）
1）在 ./data 下为新数据集建立目录，例如：
./data/my_dataset/

2）在方法配置中新增该数据集的专用配置（推荐）
在 configs/<task>/<method>.yaml 中加入 dataset_specific_configs（如需要）：

dataset_specific_configs:
  my_dataset:
    data_path: ./data/my_dataset
    max_length: 128
    num_labels: 20

优势：
- 不需要在 runner 层加任何硬编码
- 不同方法可以对同一 dataset 采用不同配置

方式 B：方法脚本内部按 dataset 名称分发
适用于数据加载逻辑复杂、且必须在代码侧控制的情况。
建议将分发逻辑收敛到一个统一的数据加载模块，避免每个 baseline 重复实现。

三、与模型选择相关的注意事项

部分 cli_builder 会按 dataset 名称决定使用中文或英文 BERT。
如果你接入的新数据集不是中文/英文的默认列表，建议明确：
- 在配置中指定 backbone / bert_model 路径（如果方法支持）
或
- 在方法脚本中提供可配置的 bert_model 参数入口

四、接入验证（建议流程）

1）选择一个最简单的方法（单阶段、日志清晰）
2）只跑：
- datasets: [my_dataset]
- methods: [tan 或 ab 等]
- seeds: [单个]
- fold_idxs: [单个]
- epochs 设为小值
3）检查：
- logs 下是否能看到正常的加载与训练过程
- outputs 下是否产生预期产物
- results/{task}/{method}/results.csv 是否产生并包含最后一行结果
- summary_{result_file}.csv 是否新增记录

五、常见问题

1）找不到数据文件
- 优先检查 data_path 是否正确（config / 参数是否生效）
- 再检查方法脚本是否正确解析 dataset 名称并映射到数据目录

2）不同方法对同一数据集行为不一致
- 建议用 dataset_specific_configs 将关键参数显式写出来（max_length、num_labels、文本字段名等）
- 避免依赖方法内部默认值导致差异难以追踪
