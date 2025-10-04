#!/usr/bin bash
set -o errexit

for s in 0 1 2 3 4
do
for dataset in thucnews
do
for rate in 0.25 0.5 0.75
do
    python ADB.py \
        --dataset $dataset \
        --known_cls_ratio $rate \
        --labeled_ratio 1.0 \
        --seed $s \
        --num_train_epochs 100 \
        --gpu_id 3 \
        --bert_model ../../llms/bert-base-chinese
done
done
done