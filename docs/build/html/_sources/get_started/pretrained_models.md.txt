# 预训练模型准备（Pretrained Models）

本项目运行时会从 `--model-dir` 指定的目录读取预训练模型与向量编码模型。为保证可复现与避免运行时自动下载失败，建议在首次运行前手动将以下模型下载到本地。

## 1. 必备模型列表

需要用户自行下载并放入 `--model-dir` 的模型如下（目录名建议与下表一致）：

1) bert-base-chinese
- 用途：中文数据集相关方法的 PLM/backbone
- 推荐来源：Hugging Face（google-bert/bert-base-chinese）

2) bert-base-uncased
- 用途：英文数据集相关方法的 PLM/backbone
- 推荐来源：Hugging Face（google-bert/bert-base-uncased）

3) stsb-roberta-base
- 用途：部分方法使用的句向量/语义编码模型（Sentence Transformer）
- 推荐来源：Hugging Face（sentence-transformers/stsb-roberta-base-v2）
  说明：如你的配置/代码中固定写死目录名为 `stsb-roberta-base`，可以将下载后的目录重命名为 `stsb-roberta-base` 以保持兼容。

4) Meta-Llama-3.1-8B-Instruct
- 用途：LLM 相关方法（若启用）
- 推荐来源：Hugging Face（meta-llama/Llama-3.1-8B-Instruct）
  说明：该模型通常需要在 Hugging Face 账号上完成许可协议确认/申请访问权限后才能下载。

## 2. 推荐目录结构

假设你将 `--model-dir` 设为 `/path/to/pretrained_models`，推荐组织如下：

/path/to/pretrained_models/
  bert-base-chinese/
  bert-base-uncased/
  stsb-roberta-base/
  Meta-Llama-3.1-8B-Instruct/

运行时示例：
bolt-grid --config grid_gcd.yaml --output-dir ~/tmp/exp_001 --model-dir /path/to/pretrained_models

## 3. 下载方式（示例）

方式 A：使用 huggingface-cli（推荐）
1) 安装并登录
pip install -U huggingface_hub
huggingface-cli login

2) 下载到指定目录
huggingface-cli download google-bert/bert-base-chinese --local-dir /path/to/pretrained_models/bert-base-chinese
huggingface-cli download google-bert/bert-base-uncased --local-dir /path/to/pretrained_models/bert-base-uncased
huggingface-cli download sentence-transformers/stsb-roberta-base-v2 --local-dir /path/to/pretrained_models/stsb-roberta-base
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir /path/to/pretrained_models/Meta-Llama-3.1-8B-Instruct

方式 B：使用 git lfs（可选）
适用于你希望以 git 仓库方式管理权重文件的场景。注意需要先安装 git-lfs，并保证网络与磁盘空间充足。

## 4. 常见问题

- 找不到模型或下载被拒绝：
  优先确认是否需要登录 Hugging Face，以及是否需要在网页端接受许可协议（LLM 模型常见）。
- 运行时仍然尝试联网下载：
  请检查配置/代码里引用的模型目录名是否与你本地目录一致（尤其注意大小写与连字符）。
