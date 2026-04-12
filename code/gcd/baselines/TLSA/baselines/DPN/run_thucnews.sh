#!/usr/bin/env bash

for s in 3 4 5
do
for dataset in thucnews
do
for known_cls_ratio in 0.5 0.75
do
    OPENBLAS_NUM_THREADS=48 python DPN.py \
        --dataset $dataset \
        --known_cls_ratio $known_cls_ratio \
        --cluster_num_factor 1 \
        --labeled_ratio 0.1 \
        --seed $s \
        --gpu_id 1 \
        --pretrain_batch_size 32 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --freeze_bert_parameters \
        --pretrain
done
done
done
