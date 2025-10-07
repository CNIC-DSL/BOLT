#!/usr/bin/env bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY=6c5c96209c4ef126a87fbe9840fab7c346b4c6fb6a57529fce7dea01c683fd1b

# --- 配置测试参数 ---
dataset='banking' # 1. 只选用一个数据集
seed=0

# !!! 请根据你的 init_parameter.py 和 mtp.py 确认下面的参数名 !!!
PRETRAIN_EPOCH_ARG="--num_pretrain_epochs"
TRAIN_EPOCH_ARG="--num_train_epochs"

echo "--- Running Fast Test for LOOP (Full Pipeline) ---"

# 2. 传入所有必要的参数，并将epoch设为1
python loop.py \
    --data_dir ../../../data \
    --dataset $dataset \
    --known_cls_ratio 0.75 \
    --labeled_ratio 0.1 \
    --seed $seed \
    --lr '1e-5' \
    --save_results_path 'outputs' \
    --view_strategy 'rtr' \
    --update_per_epoch 1 \
    --save_premodel \
    --save_model \
    --api_base 'https://uni-api.cstcloud.cn/v1' \
    --llm_model_name 'deepseek-v3:671b-gw'  \
    $PRETRAIN_EPOCH_ARG 2 \
    $TRAIN_EPOCH_ARG 5

echo "--- LOOP Fast Test Finished ---"