import os
import sys
import logging
import argparse
import datetime
import pickle as pkl

from utils import set_seed, load_yaml_config
from easydict import EasyDict
from dataloaders.base_data import BaseDataNew




def set_logger(args):

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"{args.method}_{args.dataset}_{args.known_cls_ratio}_{time}_{args.labeled_ratio}.log"

    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(args.log_dir, file_name))
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='open_intent_discovery',
                        help="Type for methods")

    parser.add_argument("--dataset", default='banking', type=str,
                        help="The name of the dataset to train selected")

    parser.add_argument("--dataset_dir", default='', type=str,
                        help="The saving path of the dataset")

    parser.add_argument("--data_dir", default='', type=str,
                        help="Project data root directory that contains <dataset>/train.tsv etc. If empty, use config data_dir.")

    parser.add_argument("--known_cls_ratio", default=0.75, type=float,
                        help="The number of known classes")

    parser.add_argument("--labeled_ratio", default=0.1, type=float,
                        help="The ratio of labeled samples in the training set")

    parser.add_argument("--cluster_num_factor", default=1.0, type=float,
                        help="The factor (magnification) of the number of clusters K.")

    parser.add_argument("--cluster_k", default=1, type=int,
                        help="The factor (magnification) of the number of clusters K.")

    parser.add_argument("--method", type=str, default='alnid',
                        help="which method to use")

    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")

    parser.add_argument("--config_file_name", type=str, default = 'Prompt.py',
                        help="The config file name for the model.")

    parser.add_argument("--bert_model", default='', type=str,
                        help="HuggingFace model path/name for AutoModel/Tokenizer. If empty, use config bert_model.")

    parser.add_argument("--llm_model_name", type=str, default='',
                        help="LLM model name used in al_finetune. If empty, read from env LLM_MODEL.")

    parser.add_argument("--api_base", type=str, default='',
                        help="LLM API base URL (OpenAI-compatible), e.g. https://uni-api.cstcloud.cn/v1. If empty, read from env LLM_BASE_URL.")

    parser.add_argument('--log_dir', type=str, default='logs',
                        help="Logger directory.")

    parser.add_argument("--output_dir", default= './outputs', type=str,
                        help="The output directory where all train data will be written.")

    parser.add_argument("--result_dir", type=str, required=True,
                        default = 'results', help="The path to save results")

    parser.add_argument("--results_file_name", type=str, default = 'results.csv',
                        help="The file name of all the results.")

    parser.add_argument("--model_file_name", type=str, default = '',
                        help="The file name of trained model.")

    parser.add_argument("--pretrained_nidmodel_file_name", type=str, default = '',
                        help="The file name of pretrained_nidmodel_file_name.")

    parser.add_argument("--save_results", action="store_true",
                        help="save final results for open intent detection")

    parser.add_argument("--cl_loss_weight", default=1.0, type=float,
                        help="loss_weight")

    parser.add_argument("--semi_cl_loss_weight", default=1.0, type=float,
                        help="loss_weight")

    args = parser.parse_args()

    return args


def run():

    command_args = parse_arguments()
    configs = load_yaml_config(command_args.config_file_name)
    args = EasyDict(dict(vars(command_args), **configs))

    if getattr(command_args, 'data_dir', ''):
        args.data_dir = command_args.data_dir
    if getattr(command_args, 'bert_model', ''):
        args.bert_model = command_args.bert_model

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    if getattr(args, 'data_dir', None):
        if not os.path.isabs(args.data_dir) and (str(args.data_dir).startswith('.') or str(args.data_dir).startswith('..')):
            cand = os.path.join(repo_root, args.data_dir)
            if os.path.exists(cand):
                args.data_dir = cand

    if getattr(args, 'bert_model', None):
        if not os.path.isabs(args.bert_model) and (str(args.bert_model).startswith('.') or str(args.bert_model).startswith('..')):
            cand = os.path.join(repo_root, args.bert_model)
            if os.path.exists(cand):
                args.bert_model = cand

    if not getattr(args, 'llm_model_name', ''):
        args.llm_model_name = os.environ.get('LLM_MODEL', '')
    if not getattr(args, 'api_base', ''):
        args.api_base = os.environ.get('LLM_BASE_URL', '')

    logger = set_logger(args)

    logger.debug("="*30+" Params "+"="*30)
    for k in args.keys():
        logger.debug(f"{k}:\t{args[k]}")
    logger.debug("="*30+" End Params "+"="*30)

    set_seed(args.seed)

    logger.info('Data and Model Preparation...')

    dataset_dir = args.dataset_dir
    if not dataset_dir:
        dataset_dir = os.path.join(
            f"datasets_{args.known_cls_ratio}",
            f"data_{args.known_cls_ratio}_{args.dataset}_{args.seed}.pkl",
        )

    dataset_path = os.path.join(args.output_dir, dataset_dir)
    try:
        logger.info('Loading the processed data...')
        with open(dataset_path, 'rb') as f:
            data = pkl.load(f)
    except (FileNotFoundError, EOFError, pkl.UnpicklingError, IsADirectoryError, OSError) as e:
        logger.info('Dataset not found or corrupted. Re-processing the training data.')

        dataset_cache_dir = os.path.dirname(dataset_path)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        data = BaseDataNew(args)
        with open(dataset_path, 'wb') as f:
            pkl.dump(data, f)

    if args.mode == 'train':
        from methods.alup.pretrain_manager import PretrainManager
        from methods.alup.manager import Manager

        logger.info('Pretrain Begin...')
        pretrain_manager = PretrainManager(args, data)
        pretrain_manager.train(args)
        if args.finish_pretrain is not None and args.finish_pretrain:
            return
        manager = Manager(args, data, pretrained_model=pretrain_manager.model)
        manager.train(args, data)
        logger.info('Pretrain Finished...')
    elif args.mode == 'al_finetune':
        from methods.alup.al_manager import ALManager
        logger.info('AL Fine-tuning Begin...')
        model_path = os.path.join(args.output_dir, f'models_{args.known_cls_ratio}', args.pretrained_nidmodel_file_name)
        logger.info(f"Loading pretrained model from: {model_path}")
        finetune_manager = ALManager(args, data, model_path)
        finetune_manager.al_finetune(args)


if __name__ == '__main__':

    run()
