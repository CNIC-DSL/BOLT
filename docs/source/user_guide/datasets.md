# Datasets Overview

This project covers a variety of text classification / intent recognition datasets, including both English and Chinese scenarios. This page briefly introduces the datasets currently supported, and provides common abbreviations and sources.

Notes:
- “Our constructed version” refers to the version used in the paper’s experimental setup, where the original dataset is sampled/filtered (e.g., sampling 10,000 examples).
- For how to download and place the data files, please follow the project’s data preparation scripts or repository instructions. This page mainly provides background and sources.

## 1. Dataset List (Organized Based on the Paper Description)

1) 20NG (20 Newsgroups)
- Description: About 20,000 newsgroup posts, distributed approximately evenly across 20 categories.
- Our constructed version: We sampled 10,000 texts for experiments.
- Source: https://people.csail.mit.edu/jrennie/20Newsgroups/

2) THUCNews
- Description: A large-scale Chinese news classification corpus, collected from Sina News RSS history data between 2005 and 2011.
- Our constructed version: We sampled 10,000 texts for experiments.
- Source: http://thuctc.thunlp.org/

3) Yahoo (Yahoo! Answers Topic Classification)
- Description: A topic classification dataset from Yahoo! Answers, containing 10 coarse-grained topic classes.
- Our constructed version: We sampled 10,000 texts for experiments.
- Source: https://www.kaggle.com/datasets/bhavikardeshna/yahoo-email-classification/data

4) BANK (BANKING77)
- Description: An English banking customer-service intent dataset with 13,083 utterances annotated with 77 fine-grained intent classes.
- Source: https://huggingface.co/datasets/PolyAI/banking77

5) S.O. (StackOverflow)
- Description: A short-text title dataset from Stack Overflow with 20,000 examples, labeled with 20 technical topic tags.
- Source: https://github.com/jacoxu/StackOverflow

6) CLINC (CLINC150)
- Description: An intent classification dataset covering 150 in-domain intents, commonly used to evaluate out-of-domain / open-set scenarios.
- Source: https://archive.ics.uci.edu/dataset/570/clinc150

7) HWU (HWU-64)
- Description: An English multi-domain intent detection dataset (e.g., news, IoT, etc.) covering 64 intents, commonly used in dialogue system research.
- Source: https://service.tib.eu/ldmservice/en/dataset/hwu64

8) ECDT (SMP-ECDT)
- Description: A Chinese intent classification dataset (SMP-ECDT), used in the paper for Chinese intent recognition scenarios.
- Source: https://arxiv.org/pdf/1709.10217

9) TREC
- Description: An open-domain factoid question classification dataset (Question Classification).
- Source: https://huggingface.co/datasets/CogComp/trec

10) M-CID
- Description: A multilingual COVID-19 intent detection dataset with 6,871 utterances and 16 intents. In this work, we only use the English utterances to construct the experimental version.
- Source: https://openreview.net/pdf?id=Ku-nv600bNM

11) DBpedia
- Description: A text classification dataset related to ontology/classes from DBpedia, a structured knowledge base extracted from Wikipedia. The paper constructs a sampled version for comparison.
- Our constructed version: We sampled 10,000 texts for experiments.
- Source: https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes

## 2. Common Usage in Configuration Files (Abbreviation Tips)

In YAML, datasets are typically specified as strings, for example:
datasets: [20NG, THUCNews, Yahoo, BANK, S.O., CLINC, HWU, ECDT, TREC, M-CID, DBpedia]

Different repository versions may use slightly different spellings for dataset names (e.g., “S.O.” vs “StackOverflow”).
If you encounter an error like “dataset not found”, it is recommended to first check the actual supported dataset name list in the data-loading code, and keep the YAML names consistent with it.
