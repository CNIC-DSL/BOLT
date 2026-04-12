#!/usr/bin/env bash
for d in 'thucnews'
do
for known_cls_ratio in 0.5 0.25
do
for seed in 3 4 5
do
python TAN.py \
    --dataset $d \
    --gpu_id 0 \
    --known_cls_ratio $known_cls_ratio \
    --cluster_num_factor 1 \
    --seed $seed \
    --freeze_bert_parameters \
    --bert_model ./pretrained_models/bert-base-chinese \
    --save_model \
    --pretrain_batch_size 24 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --save_results_path ./results_thucnews \
    --pretrain
done
done
done
