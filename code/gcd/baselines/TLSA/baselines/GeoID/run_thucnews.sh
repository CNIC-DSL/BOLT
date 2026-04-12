#!/usr/bin/env bash
export OMP_NUM_THREADS=24
export CUDA_VISIBLE_DEVICES=4
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
for d in 'thucnews'
do
for k in  0.5 0.25
do
if [ "$k" = "0.5" ]; then
for s in 2
do
	python geoid.py \
	--dataset $d \
	--known_cls_ratio $k \
	--seed $s \
	--lr '1e-3' \
	--save_results_path 'results/thucnews' \
	--view_strategy 'rtr' \
	--update_per_epoch 5 \
	--topk 256 \
	--num_train_epochs 200 \
	--num_warm_epochs  20 \
    --pretrain_batch_size 32 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--feat_dim   64 \
	--num_pretrain_epochs 50\
	--bert_model ./pretrained_models/bert-base-chinese \
	--tokenizer ./pretrained_models/bert-base-chinese \
	>> exper/thucnews${k}_${s}.txt
done
else
for s in 0 1 2
do
	python geoid.py \
	--dataset $d \
	--known_cls_ratio $k \
	--seed $s \
	--lr '1e-3' \
	--save_results_path 'results/thucnews' \
	--view_strategy 'rtr' \
	--update_per_epoch 5 \
	--topk 256 \
	--num_train_epochs 200 \
	--num_warm_epochs  20 \
    --pretrain_batch_size 32 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--feat_dim   64 \
	--num_pretrain_epochs 50\
	--bert_model ./pretrained_models/bert-base-chinese \
	--tokenizer ./pretrained_models/bert-base-chinese \
	>> exper/thucnews${k}_${s}.txt
done
fi
done
done
done
