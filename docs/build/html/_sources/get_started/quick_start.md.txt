# 快速开始

本节目标：用最少的步骤完成一次可复现的运行，并明确三个关键参数：--config、--output-dir、--model-dir。

## 1. 一条命令跑通

安装完成后，直接运行（示例以 GCD 配置为例）：

bolt-grid --config grid_gcd.yaml --output-dir ~/tmp --model-dir ~/code/bolt/pretrained_models

参数说明：
- --config
  指定配置文件。可以传入：
  1）配置文件名（例如 grid_gcd.yaml）：框架会从内置的 configs 目录中查找同名配置；
  2）配置文件绝对路径：用于运行你自行复制并修改后的配置文件。
- --output-dir
  指定本次运行的工作目录。所有日志（logs/）、结果汇总（results/）与训练/评测产物（outputs/）都会写入该目录。
- --model-dir
  指定预训练模型与缓存目录。建议指向一个稳定且可读写的位置，便于复用已下载的模型文件。

## 2. 推荐的第一次运行方式

为避免首次运行组合过多、耗时过长，建议采用以下流程：

步骤 1：复制一份配置到你自己的目录
例如将内置配置复制出来（路径按你的实际情况选择），然后在副本上做修改。

步骤 2：将配置缩小到“最小可运行规模”
建议首次只保留：
- 1 个 dataset
- 1 个 method
- 1 个 seed
- 1 个 fold
- 训练 epoch 设为较小值（用于验证流程）

步骤 3：使用绝对路径运行你的配置副本
bolt-grid --config /abs/path/to/your_config.yaml --output-dir ~/tmp/exp_001 --model-dir ~/code/bolt/pretrained_models

这样可以保证：
- 每次运行的配置来源清晰；
- 不同实验之间互不覆盖；
- 输出目录结构一致，便于对比与清理。

## 3. 如何判断“运行成功”

运行过程中：
- 标准输出会持续打印执行进度与关键提示信息；
- 如出现错误，通常会提示到日志目录中查看详细错误信息。

运行结束后：
- 你应当能在 --output-dir 下看到 logs/、results/、outputs/ 等目录；
- 下一步请打开 outputs.md 查看如何定位日志与汇总结果，以及常见问题的排查顺序。

## 4. 常见注意事项

- 请确保 --output-dir 指向“运行根目录”，不建议直接设为某个 outputs 子目录，以免产生嵌套结构，影响后续查找与清理。
- 建议将 --model-dir 固定到一个长期目录，避免重复准备模型与缓存。
