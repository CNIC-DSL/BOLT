# baselines/analogy/args.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='clinc', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--prompt_method', type=str, default='analogy')
    parser.add_argument('--mask_ratio', type=float, default=0.25, choices=[0.25, 0.5, 0.75], help='Label mask ratio')
    parser.add_argument('--llm_url', type=str, default='https://openrouter.ai/api/v1', help='VLLM model URL')
    parser.add_argument('--llm_name', type=str, default='deepseek-v3:671b-gw', help='Served model name')
    parser.add_argument('--data_dir', type=str, default='../../data', help='Data root path')
    parser.add_argument('--num_threads', type=int, default=10, help='Number of threads for parallel inference')
    args = parser.parse_args()
    args.output_path = f'outputs/results/{args.prompt_method}/{args.llm_name}_{args.dataset}_{args.mask_ratio}.tsv'
    return args