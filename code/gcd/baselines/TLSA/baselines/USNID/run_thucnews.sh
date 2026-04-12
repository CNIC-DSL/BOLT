#!/usr/bin/env bash
for dataset in 'thucnews'
do
for known_cls_ratio in 0.75 0.5 0.25
do
for seed in 0 1 2
do
    python run.py \
    --dataset $dataset \
    --method 'USNID' \
    --setting 'semi_supervised' \
    --known_cls_ratio $known_cls_ratio \
    --seed $seed \
    --train \
    --tune \
    --backbone 'bert_USNID' \
    --config_file_name 'SemiUSNID' \
    --gpu_id '5' \
    --results_file_name 'results_thucnews.csv' \
    --save_results \

done
done
done
