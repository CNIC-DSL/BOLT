config_file='../../configs/yaml/7916-deepspeed_config_z3_qlora.yaml'
set -o errexit
gpu_id='0'
# for reg_loss in 'normal' 'npo'  'vos'
for reg_loss in 'normal'
do
for backbone in Meta-Llama-3.1-8B-Instruct
do
for seed in 0
do
for dataset_name in 'banking'
do
for rate in 0.25
do
sh scripts/run.sh  $dataset_name $rate $backbone $seed $config_file $gpu_id $reg_loss
done
done
done
done
done