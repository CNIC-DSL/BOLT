#!/usr/bin/env bash
# OPENBLAS_NUM_THREADS=16 python PTJN.py \
#     --dataset banking \
#     --known_cls_ratio 0.75 \
#     --cluster_num_factor 1 \
#     --seed 0 \
#     --gpu_id 2 \
#     --freeze_bert_parameters \
#     --pretrain \

for known_cls_ratio in 0.75 0.5 0.25
do
for dataset in thucnews
do
for s in 0 1 2
do
    OPENBLAS_NUM_THREADS=24 python PTJN.py \
        --dataset $dataset \
        --known_cls_ratio $known_cls_ratio \
        --cluster_num_factor 1 \
        --seed $s \
        --gpu_id 0 \
        --freeze_bert_parameters \
        --pretrain \
        --save_results_path ./results_thucnews \
        --train_batch_size 32 \
        --eval_batch_size 32 \

done
done
done
