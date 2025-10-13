import argparse

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_model():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None, help="Path to the YAML config file.")
    parser.add_argument("--output_dir", type=str, default='./outputs/openset/adb', help="The unified output directory for all experiment artifacts.")

    parser.add_argument("--data_dir", default='./data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--save_results_path", type=str, default='./results', help="the path to save results")
    # parser.add_argument("--pretrain_dir", default='./outputs/adb/models', type=str, 
    #                     help="The output directory where the model predictions and checkpoints will be written.") 
    parser.add_argument("--bert_model", default="./pretrained_models/bert-base-uncased", type=str, help="The path for the pre-trained bert model.")
    
    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--freeze_bert_parameters", default=True, type=str_to_bool, help="Freeze the last parameters of BERT") 

    parser.add_argument("--save_model", type=str_to_bool, default=False, help="save trained-model")
    parser.add_argument("--save_results", type=str_to_bool, default=False, help="save test results")

    parser.add_argument("--dataset", default="news", type=str, 
                        help="The name of the dataset to train selected")
    parser.add_argument("--known_cls_ratio", default=0.25, type=float, help="The number of known classes")
    parser.add_argument("--labeled_ratio", default=1.0, type=float, help="The ratio of labeled samples in the training set")
    parser.add_argument("--method", type=str, default=None, help="which method to use")
    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")
    parser.add_argument("--lr", default=2e-5, type=float,help="The learning rate of BERT.")    
    parser.add_argument("--cluster_num_factor", default=1.0, type=float,help="The learning rate of BERT.")    
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.") 
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Batch size for evaluation.")    
    parser.add_argument("--wait_patient", default=10, type=int,
                        help="Patient steps for Early Stop.")    
    parser.add_argument("--lr_boundary", type=float, default=0.05, help="The learning rate of the decision boundary.")

    parser.add_argument("--fold_idx", default=0, type=int)
    parser.add_argument("--fold_num", default=5, type=int)
    parser.add_argument("--fold_type", type=str, default="fold", help="", choices=['imbalance_fold', 'fold'])

    return parser