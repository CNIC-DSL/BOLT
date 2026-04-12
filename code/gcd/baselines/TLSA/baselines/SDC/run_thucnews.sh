#!/usr/bin/env bash
set -o errexit
export CUDA_VISIBLE_DEVICES='3'

for SEED in 0 1 2
do
for DATASET in thucnews
do
for KNOWN_CLS_RATIO in 0.75 0.5 0.25
do
OPENBLAS_NUM_THREADS=24 python train.py \
    --dataset $DATASET \
    --known_cls_ratio $KNOWN_CLS_RATIO \
    --seed $SEED \
    --pretrain_dir 'baseline_models' \
    --train_dir 'sdc_models' \
    --train_batch_size 24 \
    --pretrain_batch_size 24 \
    --eval_batch_size 24 \
    --bert_model ./pretrained_models/bert-base-chinese \
    --tokenizer ./pretrained_models/bert-base-chinese
done
done
done
