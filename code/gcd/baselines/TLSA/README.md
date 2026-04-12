# TLSA: LLM-Guided Text-Label Space Alignment with Contrastive Learning for Generalized Category Discovery

This repository contains the implementation of TLSA, a method for Generalized Category Discovery (GCD) in NLP tasks.

## Overview

TLSA addresses the challenge of discovering novel categories in unlabeled data while leveraging knowledge from known categories. The method uses:
- Text-Label contrastive learning with BERT encoders
- LLM-assisted label generation for novel categories
- Confidence-based sample selection and label refinement

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```
## LLM Configuration

Set environment variables for LLM API access:
```bash
export LLM_MODEL="your-model-name"
export LLM_BASE_URL="https://your-api-endpoint/v1"
export LLM_API_KEY="your-api-key"
```

## Training

Run the training script:
```bash
bash scripts/run_banking_0.5.sh
```

## Training Pipeline

1. **Phase 1 (Supervised Warmup)**: Train on labeled known-class data with contrastive loss
2. **Phase 2 (Semi-supervised)**: 
   - Cluster unlabeled data
   - Generate labels for novel clusters via LLM
   - Refine labels using similarity-based grouping
   - Train on combined labeled and pseudo-labeled data
