#!/usr/bin/env bash
for s in 10 11 12 13 14 15 16 17
do
for dataset in thucnews
do
for known_cls_ratio in 0.75 0.5 0.25
do
    OPENBLAS_NUM_THREADS=24 python main.py \
    --dataset $dataset \
    --known_cls_ratio $known_cls_ratio \
    --seed $s \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --gpu_id 3
done
done
done
