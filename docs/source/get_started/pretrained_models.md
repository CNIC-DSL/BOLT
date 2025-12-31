# Pretrained Model Preparation (Pretrained Models)

During execution, this project reads pretrained models and embedding/encoder models from the directory specified by `--model-dir`. To ensure reproducibility and to avoid failures caused by automatic downloads at runtime, it is recommended to manually download the following models to your local machine before the first run.

## 1. Required Model List

The following models must be downloaded by the user and placed under `--model-dir` (it is recommended to keep the directory names consistent with the list below):

1) bert-base-chinese
- Purpose: PLM/backbone for methods related to Chinese datasets
- Recommended source: Hugging Face (`google-bert/bert-base-chinese`)

2) bert-base-uncased
- Purpose: PLM/backbone for methods related to English datasets
- Recommended source: Hugging Face (`google-bert/bert-base-uncased`)

3) stsb-roberta-base
- Purpose: Sentence embedding / semantic encoder model used by some methods (Sentence Transformer)
- Recommended source: Hugging Face (`sentence-transformers/stsb-roberta-base-v2`)
  Note: If your configuration/code hard-codes the directory name as `stsb-roberta-base`, you can rename the downloaded folder to `stsb-roberta-base` to maintain compatibility.

4) Meta-Llama-3.1-8B-Instruct
- Purpose: LLM-related methods (if enabled)
- Recommended source: Hugging Face (`meta-llama/Llama-3.1-8B-Instruct`)
  Note: This model typically requires accepting a license agreement / requesting access via your Hugging Face account before you can download it.

## 2. Recommended Directory Layout

Assume you set `--model-dir` to `/path/to/pretrained_models`. The recommended structure is:

/path/to/pretrained_models/
  bert-base-chinese/
  bert-base-uncased/
  stsb-roberta-base/
  Meta-Llama-3.1-8B-Instruct/

Runtime example:
bolt-grid --config grid_gcd.yaml --output-dir ~/tmp/exp_001 --model-dir /path/to/pretrained_models

## 3. Download Methods (Examples)

Method A: Use `huggingface-cli` (recommended)
1) Install and log in
pip install -U huggingface_hub
huggingface-cli login

2) Download to the target directories
huggingface-cli download google-bert/bert-base-chinese --local-dir /path/to/pretrained_models/bert-base-chinese
huggingface-cli download google-bert/bert-base-uncased --local-dir /path/to/pretrained_models/bert-base-uncased
huggingface-cli download sentence-transformers/stsb-roberta-base-v2 --local-dir /path/to/pretrained_models/stsb-roberta-base
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir /path/to/pretrained_models/Meta-Llama-3.1-8B-Instruct

Method B: Use `git lfs` (optional)
This is suitable if you want to manage weight files as a Git repository. Make sure `git-lfs` is installed, and ensure your network connection and disk space are sufficient.

## 4. Troubleshooting

- Model not found or download access denied:
  First confirm whether you need to log in to Hugging Face, and whether you must accept a license agreement on the website (common for LLM models).

- The program still tries to download online at runtime:
  Check whether the model directory names referenced in your configuration/code match your local folder names exactly (pay special attention to case sensitivity and hyphens).
