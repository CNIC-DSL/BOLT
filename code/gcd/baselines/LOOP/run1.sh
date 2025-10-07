#!/usr/bin/env bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=4
export OPENAI_API_KEY=6c5c96209c4ef126a87fbe9840fab7c346b4c6fb6a57529fce7dea01c683fd1b

for seed in 0
do
	for dataset in 'stackoverflow' 'clinc' 'banking' 'hwu' ''mcid
    do
		for known_cls_ratio in 0.25 0.5 0.75
        do
            echo "===> Running: Dataset=${dataset}, KnownClsRatio=${known_cls_ratio}, Seed=${seed}"

            python loop.py \
                --data_dir ../../data \
                --dataset $dataset \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio 0.1 \
                --seed $seed \
                --lr '1e-5' \
                --save_results_path 'outputs' \
                --view_strategy 'rtr' \
                --update_per_epoch 5 \
                --save_premodel \
                --save_model    \
                --api_base 'http://localhost:8864/v1' \
                --llm_model_name 'qwen27B' \
                --num_pretrain_epochs 100 \
                --num_train_epochs 50
                # --api_base 'https://uni-api.cstcloud.cn/v1' \
                # --llm_model_name 'deepseek-v3:671b-gw' 
        done
    done
done