# Preprocessing
import sys
import random
import time
import json
import os
import argparse
from utils import *
import numpy as np
import pandas as pd
from tqdm import tqdm_gui
from tqdm import tqdm
import nltk
nltk.data.path.append("/data/code/bolt/pretrained_models/nltk_data")
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
from thop import profile
# Modeling
import torch
torch.backends.cudnn.enabled = False
from model import BiLSTM
from model import PGD_contrastive
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
from keras import backend as K

# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
import yaml


def apply_config_updates(args, config_dict, parser):
    """
    使用配置字典中的值更新 args 对象，同时进行类型转换。
    命令行中显式给出的参数不会被覆盖。
    """
    type_map = {action.dest: action.type for action in parser._actions}
    for key, value in config_dict.items():
        # bool 类型的特殊处理
        if key in type_map and isinstance(type_map[key], type(lambda x: x)):
             if isinstance(value, str):
                 value = (value.lower() == 'true')
                 
        if f'--{key}' not in sys.argv and hasattr(args, key):
            expected_type = type_map.get(key)
            if expected_type and value is not None:
                try:
                    value = expected_type(value)
                except (TypeError, ValueError):
                    pass # 如果类型转换失败，则保留从YAML加载的原始值
            setattr(args, key, value)

# Parse Arguments
def define_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", type=str, help="Path to the YAML config file")
    
    parser.add_argument("--dataset", type=str, default='news')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--known_cls_ratio', type=float, default=0.25)
    parser.add_argument('--labeled_ratio', type=float, default=1.0)
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--fold_num', type=int, default=5)
    parser.add_argument("--gpu_id", type=str, default="0", help="The gpu device to use.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Mini-batch size for train and validation")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # parser.add_argument("--proportion", type=int, default=25,
    #                     help="The proportion of seen classes, range from 0 to 100.")
    parser.add_argument("--seen_classes", type=str, nargs="+", default=None,
                        help="The specific seen classes.")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both", "find_threshold"], default="test",
                        help="Specify running mode: only train, only test or both.")
    parser.add_argument("--setting", type=str, nargs="+", default='lof',
                        help="The settings to detect ood samples, e.g. 'lof' or 'gda_lsqr")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="The directory contains model file (.h5), requried when test only.")
    parser.add_argument("--seen_classes_seed", type=int, default=None,
                        help="The random seed to randomly choose seen classes.")
    # default arguments
    parser.add_argument("--cuda", action="store_true",
                        help="Whether to use GPU or not.")
    # parser.add_argument("--gpu_device", type=str, default="0,1",
    #                     help="The gpu device to use.")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="The directory to store training models & logs.")
    parser.add_argument("--experiment_No", type=str, default="vallian",
                        help="Manually setting of experiment number.")
    # model hyperparameters
    parser.add_argument("--embedding_file", type=str,
                        default="./glove_embeddings/glove.6B.300d.txt",
                        help="The embedding file to use.")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="The dimension of hidden state.")
    parser.add_argument("--contractive_dim", type=int, default=32,
                        help="The dimension of hidden state.")
    parser.add_argument("--embedding_dim", type=int, default=300,
                        help="The dimension of word embeddings.")
    parser.add_argument("--max_seq_len", type=int, default=64,
                        help="The max sequence length. When set to None, it will be implied from data.")
    parser.add_argument("--max_num_words", type=int, default=10000,
                        help="The max number of words.")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="The layers number of lstm.")
    parser.add_argument("--do_normalization", type=bool, default=True,
                        help="whether to do normalization or not.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="relative weights of classified loss.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="relative weights of adversarial classified loss.")
    parser.add_argument("--unseen_proportion", type=int, default=100,
                        help="proportion of unseen class examples to add in, range from 0 to 100.")
    parser.add_argument("--mask_proportion", type=int, default=0,
                        help="proportion of seen class examples to mask, range from 0 to 100.")
    parser.add_argument("--ood_loss", action="store_true",
                        help="whether ood examples to backpropagate loss or not.")
    parser.add_argument("--adv", action="store_true",
                        help="whether to generate perturbation through adversarial attack.")
    parser.add_argument("--cont_loss", action="store_true",
                        help="whether to backpropagate contractive loss or not.")
    parser.add_argument("--norm_coef", type=float, default=0.1,
                        help="coefficients of the normalized adversarial vectors")
    parser.add_argument("--cluster_num_factor", type=float, default=1.0,
                        help="coefficients of the normalized adversarial vectors")
    parser.add_argument("--n_plus_1", action="store_true",
                        help="treat out of distribution examples as the N+1 th class")
    parser.add_argument("--augment", action="store_true",
                        help="whether to use back translation to enhance the ood data")
    parser.add_argument("--cl_mode", type=int, default=1,
                        help="mode for computing contrastive loss")
    parser.add_argument("--lmcl", action="store_true",
                        help="whether to use LMCL loss")
    parser.add_argument("--cont_proportion", type=float, default=1.0,
                        help="coefficients of the normalized adversarial vectors")
    parser.add_argument("--dataset_proportion", type=float, default=100,
                        help="proportion for each in-domain data")
    parser.add_argument("--use_bert", action="store_true",
                        help="whether to use bert")
    parser.add_argument("--sup_cont", action="store_true",
                    help="whether to add supervised contrastive loss")
    # training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="Max epoches when in-domain pre-training.")
    parser.add_argument("--num_pretrain_epochs", type=int, default=100,
                        help="Max epoches when in-domain supervised contrastive pre-training.")
    parser.add_argument("--aug_pre_epoches", type=int, default=100,
                        help="Max epoches when adversarial contrastive training.")
    parser.add_argument("--finetune_epoches", type=int, default=20,
                        help="Max epoches when finetune model")
    parser.add_argument("--patience", type=int, default=20,
                        help="Patience when applying early stop.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Mini-batch size for train and validation")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="weight_decay")
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--save_results_path', type=str, default='results/openset/scl', help='gradient clipping')
    # args = parser.parse_args()
    return parser

def load_and_process_data_for_scl(args):
        """
        SOP标准化的数据加载与预处理函数 for SCL.
        """
        # 1. 从标准路径加载数据
        origin_train_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'train.tsv')
        origin_dev_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'dev.tsv')
        origin_test_path = os.path.join(args.data_dir, args.dataset, 'origin_data', 'test.tsv')

        labeled_train_path = os.path.join(args.data_dir, args.dataset, 'labeled_data', str(args.labeled_ratio), 'train.tsv')
        labeled_dev_path = os.path.join(args.data_dir, args.dataset, 'labeled_data', str(args.labeled_ratio), 'dev.tsv')

        origin_train_df = pd.read_csv(origin_train_path, sep='\t')
        origin_dev_df = pd.read_csv(origin_dev_path, sep='\t')
        test_df = pd.read_csv(origin_test_path, sep='\t')

        labeled_train_df = pd.read_csv(labeled_train_path, sep='\t')
        labeled_dev_df = pd.read_csv(labeled_dev_path, sep='\t')
        
        train_df = labeled_train_df
        train_df['text'] = origin_train_df['text']
        
        dev_df = labeled_dev_df
        dev_df['text'] = origin_dev_df['text']

        # 2. 加载标准化的已知类列表
        known_label_path = os.path.join(args.data_dir, args.dataset, 'label', f'fold{args.fold_num}', f'part{args.fold_idx}', f'label_known_{args.known_cls_ratio}.list')
        y_cols_seen = pd.read_csv(known_label_path, header=None)[0].tolist()
        y_cols_all = train_df['label'].unique().tolist()
        y_cols_unseen = [l for l in y_cols_all if l not in y_cols_seen]
        n_class_seen = len(y_cols_seen)
        print(f"Loaded {n_class_seen} known classes.")

        # 2.5 (新增步骤) 先对 train_df 和 dev_df 进行完整筛选
        train_df_filtered = train_df[(train_df['label'].isin(y_cols_seen)) & (train_df['labeled'].astype(bool))]
        dev_df_filtered = dev_df[(dev_df['label'].isin(y_cols_seen)) & (dev_df['labeled'].astype(bool))]


        # 3. (关键修改) 仅在训练集上构建词汇表，防止数据泄露
        # train_texts = train_df['text'].astype(str).apply(lambda s: " ".join(word_tokenize(s)))
        train_texts = train_df_filtered['text'].astype(str).apply(lambda s: " ".join(word_tokenize(s)))
        tokenizer = Tokenizer(num_words=args.max_num_words, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~')
        tokenizer.fit_on_texts(train_texts)
        word_index = tokenizer.word_index

        # 4. 文本序列化和填充
        def texts_to_padded_sequences(texts):
            tokenized_texts = texts.astype(str).apply(lambda s: " ".join(word_tokenize(s)))
            sequences = tokenizer.texts_to_sequences(tokenized_texts)
            return pad_sequences(sequences, maxlen=args.max_seq_len, padding='post', truncating='post')

        X_train = texts_to_padded_sequences(train_df['text'])
        X_valid = texts_to_padded_sequences(dev_df['text'])
        X_test = texts_to_padded_sequences(test_df['text'])
        
        y_train = train_df.label
        y_valid = dev_df.label
        y_test = test_df.label

        # 5. 按照已知/未知类别划分数据集索引 (保留SCL原有逻辑)
        train_seen_idx = train_df_filtered.index
        train_ood_idx = y_train[y_train.isin(y_cols_unseen)].index
        
        valid_seen_idx = dev_df_filtered.index
        valid_ood_idx = y_valid[y_valid.isin(y_cols_unseen)].index

        test_seen_idx = y_test[y_test.isin(y_cols_seen)].index
        test_ood_idx = y_test[y_test.isin(y_cols_unseen)].index

        # 6. 创建最终使用的数据集
        X_train_seen, y_train_seen = X_train[train_seen_idx], y_train[train_seen_idx]
        X_train_ood, y_train_ood = X_train[train_ood_idx], y_train[train_ood_idx]
        X_valid_seen, y_valid_seen = X_valid[valid_seen_idx], y_valid[valid_seen_idx]
        X_valid_ood, y_valid_ood = X_valid[valid_ood_idx], y_valid[valid_ood_idx]
        X_test_seen, y_test_seen = X_test[test_seen_idx], y_test[test_seen_idx]
        X_test_ood, y_test_ood = X_test[test_ood_idx], y_test[test_ood_idx]
        
        # 获取原始文本数据给DataLoader使用
        train_seen_text = train_df['text'][train_seen_idx].tolist()
        valid_seen_text = dev_df['text'][valid_seen_idx].tolist()
        valid_unseen_text = dev_df['text'][valid_ood_idx].tolist()
        test_text = test_df['text'].tolist()

        print("Train seen : Valid seen : Test seen = %d : %d : %d" % (len(X_train_seen), len(X_valid_seen), len(X_test_seen)))

        # 7. 标签编码与One-Hot转换 (保留SCL原有逻辑)
        le = LabelEncoder()
        le.fit(y_train_seen)
        y_train_idx = le.transform(y_train_seen)
        y_valid_idx = le.transform(y_valid_seen)
        y_test_idx = le.transform(y_test_seen)
        
        y_train_onehot = to_categorical(y_train_idx, num_classes=n_class_seen)
        y_valid_onehot = to_categorical(y_valid_idx, num_classes=n_class_seen)
        y_test_onehot = to_categorical(y_test_idx, num_classes=n_class_seen)

        y_train_ood_onehot = np.array([[0.0] * n_class_seen for _ in range(len(X_train_ood))])
        y_valid_ood_onehot = np.array([[0.0] * n_class_seen for _ in range(len(X_valid_ood))])
        
        y_test_mask = y_test.copy()
        y_test_mask[y_test_mask.isin(y_cols_unseen)] = 'unseen'
        
        # 8. 组合最终数据集给DataLoader
        train_data_raw = (X_train_seen, y_train_onehot)
        valid_data_raw = (X_valid_seen, y_valid_onehot)
        valid_data_ood = (X_valid_ood, np.zeros(len(X_valid_ood))) # label for ood doesn't matter much here
        train_data = (np.concatenate((X_train_seen, X_train_ood)), np.concatenate((y_train_onehot, y_train_ood_onehot)))
        valid_data = (np.concatenate((X_valid_seen, X_valid_ood)), np.concatenate((y_valid_onehot, y_valid_ood_onehot)))
        test_data = (X_test, y_test_mask)

        return (train_data_raw, valid_data_raw, valid_data_ood, test_data, train_data, valid_data,
                train_seen_text, valid_seen_text, valid_unseen_text, test_text,
                word_index, le, y_cols_unseen, n_class_seen, y_train_seen, y_test, y_test_mask)

def main(args):
    import tensorflow as tf

    # 为了兼容原代码，我们把标准参数名映射回原有的参数名
    args.proportion = int(args.known_cls_ratio * 100)
    args.batch_size = args.train_batch_size


    # --- 调用新的数据加载函数 ---
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    (train_data_raw, valid_data_raw, valid_data_ood, test_data, train_data, valid_data,
    train_seen_text, valid_seen_text, valid_unseen_text, test_text,
    word_index, le, y_cols_unseen, n_class_seen, y_train_seen, y_test, y_test_mask) = load_and_process_data_for_scl(args)

    # 兼容原脚本后续对全局变量的使用
    BETA = args.beta
    ALPHA = args.alpha
    DO_NORM = args.do_normalization
    NUM_LAYERS = args.num_layers
    HIDDEN_DIM = args.hidden_dim
    BATCH_SIZE = args.batch_size
    EMBEDDING_FILE = args.embedding_file
    MAX_SEQ_LEN = args.max_seq_len
    MAX_NUM_WORDS = args.max_num_words
    EMBEDDING_DIM = args.embedding_dim
    CON_DIM = args.contractive_dim
    OOD_LOSS = args.ood_loss
    CONT_LOSS = args.cont_loss
    ADV = args.adv
    NORM_COEF = args.norm_coef
    LMCL = args.lmcl
    CL_MODE = args.cl_mode
    USE_BERT = args.use_bert
    SUP_CONT = args.sup_cont
    CUDA = args.cuda

    class DataLoader(object):
        def __init__(self, data, batch_size, mode='train', use_bert=False, raw_text=None):
            self.use_bert = use_bert
            if self.use_bert:
                self.inp = list(raw_text)
            else:
                self.inp = data[0]
            self.tgt = data[1]
            self.batch_size = batch_size
            self.n_samples = len(data[0])
            self.n_batches = self.n_samples // self.batch_size
            self.mode = mode
            self._shuffle_indices()

        def _shuffle_indices(self):
            if self.mode == 'test':
                self.indices = np.arange(self.n_samples)
            else:
                self.indices = np.random.permutation(self.n_samples)
            self.index = 0
            self.batch_index = 0

        def _create_batch(self):
            batch = []
            n = 0
            while n < self.batch_size:
                _index = self.indices[self.index]
                batch.append((self.inp[_index],self.tgt[_index]))
                self.index += 1
                n += 1
            self.batch_index += 1
            seq, label = tuple(zip(*batch))
            if not self.use_bert:
                seq = torch.LongTensor(seq)
            if self.mode not in ['test','augment']:
                label = torch.FloatTensor(label)
            elif self.mode == 'augment':
                label = torch.LongTensor(label)

            return seq, label

        def __len__(self):
            return self.n_batches

        def __iter__(self):
            for _ in range(self.n_batches):
                if self.batch_index == self.n_batches:
                    raise StopIteration()
                yield self._create_batch()

    if args.mode in ["train", "both"]:
        # GPU setting
        set_allow_growth(device=args.gpu_id)

        timestamp = str(time.time())  # strftime("%m%d%H%M")

        output_dir = args.output_dir 
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "seen_classes.txt"), "w") as f_out:
            f_out.write("\n".join(le.classes_))
        with open(os.path.join(output_dir, "unseen_classes.txt"), "w") as f_out:
            f_out.write("\n".join(y_cols_unseen))

        if not USE_BERT:
            print("Load pre-trained GloVe embedding...")
            MAX_FEATURES = min(MAX_NUM_WORDS, len(word_index)) + 1  # +1 for PAD
            def get_coefs(word, *arr):
                return word, np.asarray(arr, dtype='float32')
            embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
            # all_embs = np.stack(embeddings_index.values())
            all_embs = np.stack(list(embeddings_index.values()))
            emb_mean, emb_std = all_embs.mean(), all_embs.std()
            embedding_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES, EMBEDDING_DIM))
            for word, i in word_index.items():
                if i >= MAX_FEATURES: continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix = None

        filepath = os.path.join(output_dir, 'model_best.pkl')
        model = BiLSTM(embedding_matrix, BATCH_SIZE, HIDDEN_DIM, CON_DIM, NUM_LAYERS, n_class_seen, DO_NORM, ALPHA, BETA, OOD_LOSS, ADV, CONT_LOSS, NORM_COEF, CL_MODE, LMCL, use_bert=USE_BERT, sup_cont=SUP_CONT, use_cuda=CUDA)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
        if args.cuda:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            model.cuda()

        #in-domain pre-training
        best_f1 = 0
        patience_counter = 0 # <--- 新增：初始化耐心计数器

        if args.sup_cont:
            for epoch in range(1,args.num_pretrain_epochs+1):
                global_step = 0
                losses = []
                train_loader = DataLoader(train_data_raw, BATCH_SIZE, use_bert=USE_BERT, raw_text=train_seen_text)
                train_iterator = tqdm(
                    train_loader, initial=global_step,
                    desc="Iter (loss=X.XXX)")
                model.train()
                for j, (seq, label) in enumerate(train_iterator):
                    if args.cuda:
                        if not USE_BERT:
                            seq = seq.cuda()
                        label = label.cuda()
                    loss = model(seq, None, label, mode='ind_pre')
                    train_iterator.set_description('Iter (sup_cont_loss=%5.3f)' % (loss.item()))
                    losses.append(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    global_step += 1
                print('Epoch: [{0}] :  Loss {loss:.4f}'.format(
                    epoch, loss=sum(losses)/global_step))
                torch.save(model, filepath)

        for epoch in range(1,args.num_train_epochs+1):
            global_step = 0
            losses = []
            train_loader = DataLoader(train_data_raw, BATCH_SIZE, use_bert=USE_BERT, raw_text=train_seen_text)
            train_iterator = tqdm(
                train_loader, initial=global_step,
                desc="Iter (loss=X.XXX)")
            valid_text = valid_seen_text + valid_unseen_text
            valid_loader = DataLoader(valid_data, BATCH_SIZE, use_bert=USE_BERT, raw_text=valid_text)
            model.train()
            for j, (seq, label) in enumerate(train_iterator):
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                    label = label.cuda()
                if epoch == 1:
                    loss = model(seq, None, label, mode='finetune')
                else:
                    loss = model(seq, None, label, sim=sim, mode='finetune')
                train_iterator.set_description('Iter (ce_loss=%5.3f)' % (loss.item()))
                losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                global_step += 1
            print('Epoch: [{0}] :  Loss {loss:.4f}'.format(
                epoch, loss=sum(losses)/global_step))

            model.eval()
            predict = []
            target = []
            if args.cuda:
                sim = torch.zeros((n_class_seen, HIDDEN_DIM*2)).cuda()
            else:
                sim = torch.zeros((n_class_seen, HIDDEN_DIM * 2))
            for j, (seq, label) in enumerate(valid_loader):
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                    label = label.cuda()
                output = model(seq, None, label, mode='validation')
                predict += output[0]
                target += output[1]
                sim += torch.mm(label.T, output[2])
            sim = sim / len(predict)
            n_sim = sim.norm(p=2, dim=1, keepdim=True)
            sim = (sim @ sim.t()) / (n_sim * n_sim.t()).clamp(min=1e-8)
            if args.cuda:
                sim = sim - 1e4 * torch.eye(n_class_seen).cuda()
            else:
                sim = sim - 1e4 * torch.eye(n_class_seen)
            sim = torch.softmax(sim, dim=1)
            f1 = metrics.f1_score(target, predict, average='macro')
            if f1 > best_f1:
                torch.save(model, filepath)
                best_f1 = f1
                patience_counter = 0  # <--- 关键：性能提升，重置耐心计数器
            else:
                patience_counter += 1 # <--- 关键：性能未提升，耐心计数器加 1

            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs with no improvement.")
                break # <--- 关键：达到耐心极限，跳出训练循环


    if args.mode in ["test", "both", "find_threshold"]:

        if args.n_plus_1:
            test_loader = DataLoader(test_data_4np1, BATCH_SIZE, use_bert=USE_BERT)
            torch.no_grad()
            model.eval()
            predict = []
            target = []
            for j, (seq, label) in enumerate(test_loader):
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                    label = label.cuda()
                output = model(seq, label, 'valid')
                predict += output[1]
                target += output[0]
            m = np.zeros((len(y_cols_seen),len(y_cols_seen)))
            for i in range(len(predict)):
                m[target[i]][predict[i]] += 1
            m[[ood_index, len(y_cols_seen) - 1], :] = m[[len(y_cols_seen) - 1, ood_index], :]
            m[:, [ood_index, len(y_cols_seen) - 1]] = m[:, [len(y_cols_seen) - 1, ood_index]]
            print(get_score(m))


        else:
            if args.mode in ["test","find_threshold"]:
                model_dir = args.model_dir
            else:
                model_dir = output_dir
            if args.cuda:
                model = torch.load(os.path.join(model_dir, "model_best.pkl"), map_location='cuda:0', weights_only=False)
            else:
                model = torch.load(os.path.join(model_dir, "model_best.pkl"), map_location='cpu', weights_only=False)
            train_loader = DataLoader(train_data_raw, BATCH_SIZE, 'test', use_bert=USE_BERT, raw_text=train_seen_text)
            valid_loader = DataLoader(valid_data_raw, BATCH_SIZE, use_bert=USE_BERT, raw_text=valid_seen_text)
            valid_ood_loader = DataLoader(valid_data_ood, BATCH_SIZE, 'test', use_bert=USE_BERT, raw_text=valid_unseen_text)
            test_loader = DataLoader(test_data, BATCH_SIZE, 'test', use_bert=USE_BERT, raw_text=test_text)
            torch.no_grad()
            model.eval()
            predict = []
            target = []
            for j, (seq, label) in enumerate(valid_loader):
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                    label = label.cuda()
                output = model(seq, None, label, mode='validation')
                predict += output[1]
                target += output[0]
            f1 = metrics.f1_score(target, predict, average='macro')
            print(f"in-domain f1:{f1}")

            valid_loader = DataLoader(valid_data_raw, BATCH_SIZE, 'test', use_bert=USE_BERT, raw_text=valid_seen_text)
            classes = list(le.classes_) + ['unseen']
            #print(list(le.classes_))
            #classes = list(le.classes_)
            feature_train = None
            feature_valid = None
            feature_valid_ood = None
            feature_test = None
            prob_train = None
            prob_valid = None
            prob_valid_ood = None
            prob_test = None
            for j, (seq, label) in enumerate(train_loader):
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                output = model(seq, None, None, mode='test')
                if feature_train != None:
                    feature_train = torch.cat((feature_train,output[1]),dim=0)
                    prob_train = torch.cat((prob_train,output[0]),dim=0)
                else:
                    feature_train = output[1]
                    prob_train = output[0]
            for j, (seq, label) in enumerate(valid_loader):
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                output = model(seq, None, None, mode='test')
                if feature_valid != None:
                    feature_valid = torch.cat((feature_valid,output[1]),dim=0)
                    prob_valid = torch.cat((prob_valid,output[0]),dim=0)
                else:
                    feature_valid = output[1]
                    prob_valid = output[0]
            for j, (seq, label) in enumerate(valid_ood_loader):
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                output = model(seq, None, None, mode='test')
                if feature_valid_ood != None:
                    feature_valid_ood = torch.cat((feature_valid_ood,output[1]),dim=0)
                    prob_valid_ood = torch.cat((prob_valid_ood,output[0]),dim=0)
                else:
                    feature_valid_ood = output[1]
                    prob_valid_ood = output[0]
            for j, (seq, label) in enumerate(test_loader):
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                output = model(seq, None, None, mode='test')
                if feature_test != None:
                    feature_test = torch.cat((feature_test,output[1]),dim=0)
                    prob_test = torch.cat((prob_test, output[0]), dim=0)
                else:
                    feature_test = output[1]
                    prob_test = output[0]
            feature_train = feature_train.cpu().detach().numpy()
            feature_valid = feature_valid.cpu().detach().numpy()
            feature_valid_ood = feature_valid_ood.cpu().detach().numpy()
            feature_test = feature_test.cpu().detach().numpy()
            prob_train = prob_train.cpu().detach().numpy()
            prob_valid = prob_valid.cpu().detach().numpy()
            prob_valid_ood = prob_valid_ood.cpu().detach().numpy()
            prob_test = prob_test.cpu().detach().numpy()
            if args.mode == 'find_threshold':
                settings = ['gda_lsqr_'+str(10.0+1.0*(i)) for i in range(20)]
            else:
                settings_arg = args.setting
                
                # ================================================================
                # --- 最终版修正逻辑 ---
                # ================================================================
                # 1. 将破损的列表零件合并成一个完整的字符串
                settings_str = "".join(settings_arg)  # 例如，"['gda','lof','msp']"
                
                # 2. 清理这个完整的字符串，去掉所有杂质
                cleaned_str = settings_str.replace('[', '').replace(']', '').replace("'", "").replace('"', '').replace(' ', '')
                
                # 3. 按逗号分割，得到最终干净的列表
                settings = cleaned_str.split(',')
                # ================================================================
                # --- 修正结束 ---
                # ================================================================

            for setting in settings:
                pred_dir = os.path.join(model_dir, f"{setting}")
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)
                setting_fields = setting.split("_")
                ood_method = setting_fields[0]
                
                print(ood_method)
                assert ood_method in ("lof", "gda", "msp")

                if ood_method == "lof":
                    method = 'LOF (LMCL)'
                    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
                    lof.fit(feature_train)
                    l = len(feature_test)
                    y_pred_lof = pd.Series(lof.predict(feature_test))
                    test_info = get_test_info(texts=test_text,
                                            label=y_test[:l],
                                            label_mask=y_test_mask[:l],
                                            softmax_prob=prob_test,
                                            softmax_classes=list(le.classes_),
                                            lof_result=y_pred_lof,
                                            save_to_file=True,
                                            output_dir=pred_dir)
                    pca_visualization(feature_test, y_test_mask[:l], classes, os.path.join(pred_dir, "pca_test.png"))
                    df_seen = pd.DataFrame(prob_test, columns=le.classes_)
                    df_seen['unseen'] = 0

                    y_pred = df_seen.idxmax(axis=1)
                    y_pred[y_pred_lof[y_pred_lof == -1].index] = 'unseen'
                    # cm = confusion_matrix(y_test_mask[:l], y_pred, classes)

                    # f, f_seen, f_unseen, p_unseen, r_unseen = get_score(cm)
                    # plot_confusion_matrix(pred_dir, cm, classes, normalize=False, figsize=(9, 6),
                    #                       title=method + ' on ' + dataset + ', f1-macro=' + str(f))
                    # print(cm)
                    # log_pred_results(f, f_seen, f_unseen, p_unseen, r_unseen, classes, pred_dir, cm, OOD_LOSS, ADV, CONT_LOSS)
                elif ood_method == "gda":
                    solver = setting_fields[1] if len(setting_fields) > 1 else "lsqr"
                    threshold = setting_fields[2] if len(setting_fields) > 2 else "auto"
                    distance_type = setting_fields[3] if len(setting_fields) > 3 else "mahalanobis"
                    assert solver in ("svd", "lsqr")
                    assert distance_type in ("mahalanobis", "euclidean")
                    l = len(feature_test)
                    method = 'GDA (LMCL)'
                    gda = LinearDiscriminantAnalysis(solver=solver, shrinkage=None, store_covariance=True)
                    gda.fit(prob_train, y_train_seen[:len(prob_train)])
                    # print(np.max(gda.covariance_class.diagonal()))
                    # print(np.min(gda.covariance_class.diagonal()))
                    # print(np.mean(gda.covariance_class.diagonal()))
                    # print(np.median(gda.covariance_class.diagonal()))
                    # print(np.max(np.linalg.norm(gda.covariance_, axis=0)))
                    # print(np.min(np.linalg.norm(gda.covariance_, axis=0)))
                    # print(np.mean(np.linalg.norm(gda.covariance_, axis=0)))
                    # print(np.median(np.linalg.norm(gda.covariance_, axis=0)))
                    # dis_matrix = np.matmul(gda.means_, gda.means_.T)
                    # K = [1,5,10,30,50]
                    # for k in K:
                    #     knn = naive_arg_topK(dis_matrix, k, axis=1)
                    #     sum = 0
                    #     for i in range(knn.shape[0]):
                    #         for j in knn[i]:
                    #             sum += dis_matrix[i][j]
                    #     print(sum/(k*knn.shape[0]))
                    if threshold == "auto":
                        # feature_valid_seen = get_deep_feature.predict(valid_data[0])
                        # valid_unseen_idx = y_valid[~y_valid.isin(y_cols_seen)].index
                        # feature_valid_ood = get_deep_feature.predict(X_valid[valid_unseen_idx])
                        seen_m_dist = confidence(prob_valid, gda.means_, distance_type, gda.covariance_).min(axis=1)
                        unseen_m_dist = confidence(prob_valid_ood, gda.means_, distance_type, gda.covariance_).min(axis=1)
                        threshold = estimate_best_threshold(seen_m_dist, unseen_m_dist)
                        # seen_m_dist = confidence(feature_valid, gda.means_, distance_type, gda.covariance_).min(axis=1)
                        # unseen_m_dist = confidence(feature_valid_ood, gda.means_, distance_type, gda.covariance_).min(axis=1)
                        # threshold = estimate_best_threshold(seen_m_dist, unseen_m_dist)
                    else:
                        threshold = float(threshold)

                    y_pred = pd.Series(gda.predict(prob_test))
                    gda_result = confidence(prob_test, gda.means_, distance_type, gda.covariance_)
                    test_info = get_test_info(texts=test_text,
                                            label=y_test[:l],
                                            label_mask=y_test_mask[:l],
                                            softmax_prob=prob_test,
                                            softmax_classes=list(le.classes_),
                                            gda_result=gda_result,
                                            gda_classes=gda.classes_,
                                            save_to_file=True,
                                            output_dir=pred_dir)
                    #pca_visualization(prob_test, y_test_mask[:l], classes, os.path.join(pred_dir, "pca_test.png"))
                    #pca_visualization(prob_train, y_train[:15000], classes, os.path.join(pred_dir, "pca_test.png"))
                    #pca_visualization(feature_test, y_test_mask[:l], classes, os.path.join(pred_dir, "pca_test.png"))
                    y_pred_score = pd.Series(gda_result.min(axis=1))
                    y_pred[y_pred_score[y_pred_score > threshold].index] = 'unseen'

                    # cm = confusion_matrix(y_test_mask[:l], y_pred, classes)
                    # f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen = get_score(cm)
                    # # plot_confusion_matrix(pred_dir, cm, classes, normalize=False, figsize=(9, 6),
                    # #                       title=method + ' on ' + dataset + ', f1-macro=' + str(f))
                    # print(cm)
                    # #log_pred_results(f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen, classes, pred_dir, cm, OOD_LOSS, ADV, CONT_LOSS, threshold)
                elif ood_method == "msp":
                    threshold = setting_fields[1] if len(setting_fields) > 1 else "auto"
                    method = 'MSP (LMCL)'
                    l = len(feature_test)
                    if threshold == "auto":
                        #prob_valid_seen = model.predict(valid_data[0])
                        #valid_unseen_idx = y_valid[~y_valid.isin(y_cols_seen)].index
                        #prob_valid_unseen = model.predict(X_valid[valid_unseen_idx])
                        seen_conf = prob_valid.max(axis=1) * -1.0
                        unseen_conf = prob_valid_ood.max(axis=1) * -1.0
                        threshold = -1.0 * estimate_best_threshold(seen_conf, unseen_conf)
                    else:
                        threshold = float(threshold)

                    df_seen = pd.DataFrame(prob_test, columns=le.classes_)
                    df_seen['unseen'] = 0

                    y_pred = df_seen.idxmax(axis=1)
                    y_pred_score = df_seen.max(axis=1)
                    y_pred[y_pred_score[y_pred_score < threshold].index] = 'unseen'

                    # cm = confusion_matrix(y_test_mask[:l], y_pred, classes)

                    # f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen = get_score(cm)
                    # plot_confusion_matrix(pred_dir, cm, classes, normalize=False, figsize=(9, 6),
                    #                       title=method + ' on ' + dataset + ', f1-macro=' + str(f))
                    # print(cm)
                    # log_pred_results(f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen,
                    #                  classes, pred_dir, cm, OOD_LOSS, ADV, CONT_LOSS, threshold)
                    

                # 打印详细报告到控制台，方便实时查看 (增加 zero_division=0 避免警告)
                print(classification_report(y_test_mask[:l], y_pred, zero_division=0))
                
                # 将报告转换为字典格式以提取指标
                # metrics = classification_report(y_test_mask[:l], y_pred, output_dict=True, zero_division=0)
                report_dict = classification_report(y_test_mask[:l], y_pred, output_dict=True, zero_division=0)

                
                # --- 步骤1：创建“单行”结果字典 (模仿ADB和DOC的逻辑) ---
                final_results = {}

                # a. 添加实验参数元数据
                final_results['dataset'] = args.dataset
                final_results['seed'] = args.seed
                final_results['known_cls_ratio'] = args.known_cls_ratio
                final_results['ood_method'] = setting # 记录本次使用的是哪种OOD检测方法

                # b. 添加核心的总览性能指标
                final_results['ACC'] = report_dict['accuracy']
                final_results['F1'] = report_dict['macro avg']['f1-score']
                final_results['args'] = json.dumps(vars(args), ensure_ascii=False)

                # c. 添加自定义的 K-F1 (已知类F1) 和 N-F1 (未知类F1) 指标
                # 获取所有已知类的标签名
                seen_class_labels = [str(c) for c in le.classes_]
                
                known_f1_scores = [report_dict[label]['f1-score'] for label in seen_class_labels if label in report_dict]
                if known_f1_scores:
                    final_results['K-F1'] = sum(known_f1_scores) / len(known_f1_scores)
                else:
                    final_results['K-F1'] = 0.0

                if 'unseen' in report_dict:
                    final_results['N-F1'] = report_dict['unseen']['f1-score']
                else:
                    final_results['N-F1'] = 0.0

                # --- 步骤2：将“单行”结果追加保存到主 results.csv 文件 ---
                # 定义标准化的 metrics 输出目录和文件路径
                # metric_dir = os.path.join(args.output_dir, 'metrics')
                os.makedirs(args.save_results_path, exist_ok=True)
                results_path = os.path.join(args.save_results_path, 'results.csv')

                df_to_save = pd.DataFrame([final_results])
                df_to_save['method'] = 'SCL'
                cols = ['method','dataset','known_cls_ratio','labeled_ratio','cluster_num_factor','seed','ACC','F1','K-F1','N-F1','args']
                for col in cols:
                    if col in df_to_save:
                        continue
                    df_to_save[col] = getattr(args, col)
                df_to_save = df_to_save[cols]

                if not os.path.exists(results_path):
                    # 如果文件不存在，直接创建并写入（包含表头）
                    df_to_save.to_csv(results_path, index=False)
                else:
                    # 如果文件存在，则读取->追加->写回
                    existing_df = pd.read_csv(results_path)
                    new_row_df = df_to_save
                    updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                    updated_df.to_csv(results_path, index=False)

                print(f"\nResults have been saved to: {results_path}")
                print("Appended new result row:")
                print(pd.DataFrame([final_results]))
                # ========================================================================
                # --- 结果保存逻辑结束 ---
                # ========================================================================


if __name__ == '__main__':
    # 1. 初始化并解析所有命令行参数
    parser = define_parser()
    args = parser.parse_args()

    # 2. 加载 YAML 并智能覆盖
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # 应用通用配置
        apply_config_updates(args, yaml_config, parser)
        
        if 'dataset_specific_configs' in yaml_config:
            dataset_configs = yaml_config['dataset_specific_configs'].get(args.dataset, {})
            apply_config_updates(args, dataset_configs, parser)
            
    # 3. 运行主逻辑
    main(args)