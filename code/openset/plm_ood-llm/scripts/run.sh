set -o errexit
dataset_name=$1
rate=$2
backbone=$3
seed=$4
config_file=$5
export CUDA_VISIBLE_DEVICES=$6
reg_loss=$7

python pretrain.py --dataset_name $dataset_name --backbone $backbone --rate $rate --seed $seed --reg_loss $reg_loss
python train_ood.py --dataset_name $dataset_name --backbone $backbone  --rate $rate --seed $seed --reg_loss $reg_loss