#!/usr/bin/env bash
export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0
export LLM_MODEL="your-llm-model"
export LLM_BASE_URL="https://your-api-endpoint.example.com/v1"
export LLM_API_KEY="YOUR_API_KEY_HERE"

for dataset in thucnews
do
for seed in 0 1 2
do
for known_cls_ratio in 0.25 0.5 0.75
do
python loop.py \
	--data_dir ./data \
	--dataset $dataset \
	--known_cls_ratio $known_cls_ratio \
	--labeled_ratio 0.1 \
	--seed $seed \
	--lr '1e-5' \
	--save_results_path 'outputs_thucnews' \
	--view_strategy 'rtr' \
	--update_per_epoch 5 \
	--save_premodel \
	--save_model \
	--pretrain_batch_size 24 \
	--train_batch_size 24 \
	--eval_batch_size 24 \
	--bert_model ./pretrained_models/bert-base-chinese \
	--tokenizer ./pretrained_models/bert-base-chinese
done
done
done
