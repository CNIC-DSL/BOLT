#!/usr/bin/env bash

# ALUP Full Pipeline Script - THUCNews Dataset
# Step 1: Pretrain  |  Step 2: AL Finetune

method="alup"
prefix="outs"
pretrain_suffix="alup_pretrain"
suffix="alup"

# Environment variables
export OPENBLAS_NUM_THREADS=24
export OMP_NUM_THREADS=24

# LLM configuration
export LLM_MODEL="your-llm-model"
export LLM_BASE_URL="https://your-api-endpoint.example.com/v1"
export LLM_API_KEY="YOUR_API_KEY_HERE"

# ============================================================
# Step 1: Pretrain
# ============================================================
echo "===== Step 1: Pretrain ====="
for dataset in "thucnews"
do
    for known_cls_ratio in 0.75 0.5 0.25
    do
        for seed in 0 1 2
        do
            CUDA_VISIBLE_DEVICES=0 python run.py \
            --dataset $dataset \
            --method ${method} \
            --known_cls_ratio $known_cls_ratio \
            --seed $seed \
            --config_file_name "./methods/${method}/configs/config_${dataset}.yaml" \
            --save_results \
            --log_dir "./${prefix}/${method}/logs/${dataset}_${pretrain_suffix}" \
            --output_dir "./${prefix}/${method}" \
            --dataset_dir "datasets_${known_cls_ratio}/data_${known_cls_ratio}_${dataset}_${seed}.pkl" \
            --data_dir "./data" \
            --model_file_name "model_${known_cls_ratio}_${dataset}_${seed}_${pretrain_suffix}" \
            --result_dir "./${prefix}/${method}/results_${dataset}_${pretrain_suffix}" \
            --results_file_name "results_${known_cls_ratio}_${dataset}_${seed}_${pretrain_suffix}.csv"  \
            --cl_loss_weight 1.0 \
            --semi_cl_loss_weight 1.0
        done
    done
done

# ============================================================
# Step 2: AL Finetune
# ============================================================
echo "===== Step 2: AL Finetune ====="
for dataset in "thucnews"
do
    for known_cls_ratio in 0.75
    do
        for seed in 0 1 2
        do
            # Find the latest epoch model file (excluding best)
            model_dir="./${prefix}/${method}/models_${known_cls_ratio}"
            latest_model_path=$(ls ${model_dir}/model_${known_cls_ratio}_${dataset}_${seed}_${pretrain_suffix}_epoch_*.pt 2>/dev/null | grep -v "best" | sort -V | tail -n 1)

            if [ -z "$latest_model_path" ]; then
                echo "Error: No latest pretrain model found for ${dataset} ${known_cls_ratio} ${seed} (excluding best). Exiting."
                exit 1
            else
                pretrained_model_filename=$(basename "$latest_model_path")
                echo "Using latest pretrain model: $pretrained_model_filename"
            fi

            CUDA_VISIBLE_DEVICES=0 python run.py \
            --dataset $dataset \
            --method ${method} \
            --known_cls_ratio $known_cls_ratio \
            --seed $seed \
            --config_file_name "./methods/${method}/configs/config_${dataset}_al_finetune.yaml" \
            --save_results \
            --log_dir "./${prefix}/${method}/logs/${dataset}_${suffix}" \
            --output_dir "./${prefix}/${method}" \
            --dataset_dir "datasets_${known_cls_ratio}/data_${known_cls_ratio}_${dataset}_${seed}.pkl" \
            --data_dir "./data" \
            --model_file_name "model_${known_cls_ratio}_${dataset}_${seed}_${suffix}.pt" \
            --pretrained_nidmodel_file_name "$pretrained_model_filename" \
            --result_dir "./${prefix}/${method}/results_${dataset}_${suffix}_kcr${known_cls_ratio}" \
            --results_file_name "results_${known_cls_ratio}_${dataset}_${seed}_${suffix}.csv"  \
            --cl_loss_weight 1.0 \
            --semi_cl_loss_weight 1.0
        done
    done
done
