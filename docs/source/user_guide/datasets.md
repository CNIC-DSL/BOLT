# 数据集介绍（Datasets）

本项目的实验覆盖多种文本分类/意图识别数据集，包含英文与中文场景。本文对当前已支持的数据集做简要说明，并给出常用缩写与来源。

说明：
- 文中“我们构建的版本”指论文实验设置中对原始数据集进行抽样/筛选后的版本（例如抽样 10,000 条样本）。
- 具体如何下载与放置数据文件，请以项目的数据准备脚本或仓库说明为准；本页主要提供背景与来源。

## 1. 数据集列表（按论文描述整理）

1) 20NG（20 Newsgroups）
- 简介：约 20,000 篇新闻组帖子，近似均匀分布在 20 个类别。
- 我们构建的版本：抽样 10,000 条文本用于实验。
- 来源：https://people.csail.mit.edu/jrennie/20Newsgroups/

2) THUCNews
- 简介：大规模中文新闻分类语料，来源于 2005–2011 年间的新浪新闻 RSS 历史数据。
- 我们构建的版本：抽样 10,000 条文本用于实验。
- 来源：http://thuctc.thunlp.org/

3) Yahoo（Yahoo! Answers Topic Classification）
- 简介：Yahoo! Answers 主题分类数据集，包含 10 个大类主题。
- 我们构建的版本：抽样 10,000 条文本用于实验。
- 来源：https://www.kaggle.com/datasets/bhavikardeshna/yahoo-email-classification/data

4) BANK（BANKING77）
- 简介：英文银行客服查询（intent）数据集，包含 13,083 条语句，标注为 77 个细粒度意图类别。
- 来源：https://huggingface.co/datasets/PolyAI/banking77

5) S.O.（StackOverflow）
- 简介：来自 Stack Overflow 的短文本标题数据集，共 20,000 条，标注为 20 个技术主题标签。
- 来源：https://github.com/jacoxu/StackOverflow

6) CLINC（CLINC150）
- 简介：意图分类数据集，覆盖 150 个 in-domain 意图类别，常用于评估 out-of-domain/开放集场景。
- 来源：https://archive.ics.uci.edu/dataset/570/clinc150

7) HWU（HWU-64）
- 简介：英文多领域用户表达（news、IoT 等）意图检测数据集，覆盖 64 个意图，常用于对话系统研究。
- 来源：https://service.tib.eu/ldmservice/en/dataset/hwu64

8) ECDT（SMP-ECDT）
- 简介：中文意图分类数据集（SMP-ECDT），论文中用于中文意图识别场景。
- 来源：https://arxiv.org/pdf/1709.10217

9) TREC
- 简介：开放域事实型问题分类数据集（Question Classification）。
- 来源：https://huggingface.co/datasets/CogComp/trec

10) M-CID
- 简介：多语言 COVID-19 意图检测数据集，包含 6,871 条语句与 16 个意图；本工作仅使用其中英文语句构建实验版本。
- 来源：https://openreview.net/pdf?id=Ku-nv600bNM

11) DBpedia
- 简介：从 Wikipedia 抽取的结构化知识库项目 DBpedia 的本体/类别相关文本分类数据集；论文实验中构建了抽样版本用于对比。
- 我们构建的版本：抽样 10,000 条文本用于实验。
- 来源：https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes

## 2. 在配置文件中的常用写法（缩写提示）

在 YAML 中通常以字符串形式指定数据集名称，例如：
datasets: [20NG, THUCNews, Yahoo, BANK, S.O., CLINC, HWU, ECDT, TREC, M-CID, DBpedia]

不同仓库版本对数据集名称的具体拼写可能存在差异（例如 “S.O.” vs “StackOverflow”）。
如遇到 “找不到数据集” 的报错，建议先在代码的数据加载处确认实际支持的 dataset name 列表，并保持 YAML 中一致。
