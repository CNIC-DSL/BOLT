#!/usr/bin/env bash

# ===============================================================
# 1. Environment Configuration (LLM and GPU)
# ===============================================================
export LLM_MODEL="your-llm-model"
export LLM_BASE_URL="https://your-api-endpoint.example.com/v1"
export LLM_API_KEY="YOUR_API_KEY_HERE"
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=24

# ===============================================================
# 2. Run Parameters
# ===============================================================

dataset="thucnews"
for seed in 0 1 2; do
    for known_cls_ratio in 0.25 0.5 0.75; do

        echo "===> Running Glean on THUCNews: Dataset=${dataset}, KnownClsRatio=${known_cls_ratio}, Seed=${seed}"
        echo "===> Using LLM: ${LLM_MODEL} at ${LLM_BASE_URL}"

        mkdir -p outputs_thucnews

        python GCDLLMs.py \
            --dataset $dataset \
            --api_base $LLM_BASE_URL \
            --llm $LLM_MODEL \
            --api_key $LLM_API_KEY \
            --seed $seed \
            --known_cls_ratio $known_cls_ratio \
            --labeled_ratio 0.1 \
            --running_method 'GLEAN' \
            --architecture 'Loop' \
            --weight_cluster_instance_cl 0.05 \
            --weight_ce_unsup 0.1 \
            --query_samples 500 \
            --options 5 \
            --options_cluster_instance_ratio 0.5 \
            --sampling_strategy 'highest' \
            --num_train_epochs 25 \
            --lr '1e-5' \
            --train_batch_size 12 \
            --pretrain_batch_size 24 \
            --eval_batch_size 32 \
            --update_per_epoch 5 \
            --view_strategy 'rtr' \
            --save_premodel \
            --save_model \
            --save_results_path 'outputs_thucnews' \
            --experiment_name "${dataset}_kcr_${known_cls_ratio}"

    done
done
