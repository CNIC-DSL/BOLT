#!/usr/bin/env bash

for dataset in thucnews
do
for known_cls_ratio in 0.75
do
for s in 3 4 5 6
do
    OPENBLAS_NUM_THREADS=16 python DeepAligned.py \
        --dataset $dataset \
        --known_cls_ratio $known_cls_ratio \
        --cluster_num_factor 1 \
        --labeled_ratio 0.1 \
        --seed $s \
        --gpu_id 5 \
        --pretrain \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --freeze_bert_parameters
done
done
done
