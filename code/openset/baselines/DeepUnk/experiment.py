import os
import argparse
import json
import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
import yaml
import sys

from models import BiLSTM_LMCL
from utils import set_allow_growth


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    set_allow_growth(args.gpu_id)

    np.random.seed(args.seed)
    rn.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("Loading standardized data...")
    known_label_path = os.path.join(
        args.data_dir,
        args.dataset,
        "label",
        f"{args.fold_type}{args.fold_num}",
        f"part{args.fold_idx}",
        f"label_known_{args.known_cls_ratio}.list",
    )
    seen_labels = pd.read_csv(known_label_path, header=None)[0].tolist()
    n_class_seen = len(seen_labels)

    origin_train_df = pd.read_csv(
        os.path.join(args.data_dir, args.dataset, "origin_data", "train.tsv"), sep="\t"
    )
    origin_valid_df = pd.read_csv(
        os.path.join(args.data_dir, args.dataset, "origin_data", "dev.tsv"), sep="\t"
    )
    origin_test_df = pd.read_csv(
        os.path.join(args.data_dir, args.dataset, "origin_data", "test.tsv"), sep="\t"
    )
    labeled_train_df = pd.read_csv(
        os.path.join(
            args.data_dir,
            args.dataset,
            "labeled_data",
            str(args.labeled_ratio),
            "train.tsv",
        ),
        sep="\t",
    )
    labeled_valid_df = pd.read_csv(
        os.path.join(
            args.data_dir,
            args.dataset,
            "labeled_data",
            str(args.labeled_ratio),
            "dev.tsv",
        ),
        sep="\t",
    )

    df_train, df_valid, df_test = (
        labeled_train_df.copy(),
        labeled_valid_df.copy(),
        origin_test_df.copy(),
    )
    df_train["text"], df_valid["text"] = (
        origin_train_df["text"],
        origin_valid_df["text"],
    )

    train_seen_df = df_train[
        (df_train["label"].isin(seen_labels)) & (df_train["labeled"])
    ]
    valid_seen_df = df_valid[df_valid["label"].isin(seen_labels)]

    print("Preprocessing text data...")
    tokenizer = Tokenizer(
        num_words=10000, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~'
    )
    tokenizer.fit_on_texts(train_seen_df["text"].astype(str))
    word_index = tokenizer.word_index

    X_train_seen = pad_sequences(
        tokenizer.texts_to_sequences(train_seen_df["text"].astype(str)),
        padding="post",
        truncating="post",
        maxlen=4096,
    )
    X_valid_seen = pad_sequences(
        tokenizer.texts_to_sequences(valid_seen_df["text"].astype(str)),
        padding="post",
        truncating="post",
        maxlen=4096,
    )
    X_test = pad_sequences(
        tokenizer.texts_to_sequences(df_test["text"].astype(str)),
        padding="post",
        truncating="post",
        maxlen=4096,
    )

    le = LabelEncoder()
    le.fit(train_seen_df["label"])
    y_train_idx = le.transform(train_seen_df["label"])
    y_valid_idx = le.transform(valid_seen_df["label"])

    y_train_onehot = to_categorical(y_train_idx, num_classes=n_class_seen)
    y_valid_onehot = to_categorical(y_valid_idx, num_classes=n_class_seen)

    y_test_mask = df_test["label"].copy()
    y_test_mask[~y_test_mask.isin(seen_labels)] = "unseen"

    print("Loading GloVe embeddings...")
    MAX_FEATURES = min(10000, len(word_index)) + 1

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    embeddings_index = dict(
        get_coefs(*o.strip().split())
        for o in open(args.embedding_file, encoding="utf-8")
    )
    emb_mean, emb_std = np.mean(list(embeddings_index.values()), axis=0), np.std(
        list(embeddings_index.values()), axis=0
    )
    embedding_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES, 300))
    for word, i in word_index.items():
        if (
            i < MAX_FEATURES
            and (embedding_vector := embeddings_index.get(word)) is not None
        ):
            embedding_matrix[i] = embedding_vector

    print("Training model...")
    ckpt_dir = os.path.join(args.output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    filepath = os.path.join(ckpt_dir, "model.h5")

    checkpoint = ModelCheckpoint(
        filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="auto",
        save_weights_only=False,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=20, mode="auto")

    model = BiLSTM_LMCL(
        max_seq_len=None,
        max_features=MAX_FEATURES,
        embedding_dim=300,
        output_dim=n_class_seen,
        embedding_matrix=embedding_matrix,
        learning_rate=args.learning_rate,
    )
    model.fit(
        X_train_seen,
        y_train_onehot,
        epochs=args.num_train_epochs,
        batch_size=args.train_batch_size,
        validation_data=(X_valid_seen, y_valid_onehot),
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint, early_stop],
    )

    print("Evaluating model...")
    y_pred_proba = model.predict(X_test)

    get_deep_feature = Model(inputs=model.inputs, outputs=model.layers[-3].output)
    feature_train = get_deep_feature.predict(X_train_seen)
    feature_test = get_deep_feature.predict(X_test)

    lof = LocalOutlierFactor(
        n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1
    )
    lof.fit(feature_train)
    y_pred_lof = pd.Series(lof.predict(feature_test))

    df_seen = pd.DataFrame(y_pred_proba, columns=le.classes_)
    y_pred = df_seen.idxmax(axis=1)
    y_pred[y_pred_lof[y_pred_lof == -1].index] = "unseen"

    print("Saving results...")
    report = classification_report(
        y_test_mask, y_pred, output_dict=True, zero_division=0
    )

    final_results = {
        "dataset": args.dataset,
        "seed": args.seed,
        "known_cls_ratio": args.known_cls_ratio,
        "ACC": report["accuracy"],
        "F1": report["macro avg"]["f1-score"],
        "args": json.dumps(vars(args), ensure_ascii=False),
    }
    known_f1_scores = [
        report[label]["f1-score"] for label in le.classes_ if label in report
    ]
    final_results["K-F1"] = (
        sum(known_f1_scores) / len(known_f1_scores) if known_f1_scores else 0.0
    )
    final_results["N-F1"] = report["unseen"]["f1-score"] if "unseen" in report else 0.0

    metric_dir = args.save_results_path
    os.makedirs(metric_dir, exist_ok=True)
    results_path = os.path.join(metric_dir, "results.csv")

    df_to_save = pd.DataFrame([final_results])

    df_to_save["method"] = "DeepUnk"
    cols = [
        "method",
        "dataset",
        "known_cls_ratio",
        "labeled_ratio",
        "cluster_num_factor",
        "seed",
        "ACC",
        "F1",
        "K-F1",
        "N-F1",
        "args",
    ]
    for col in cols:
        if col in df_to_save:
            continue
        df_to_save[col] = getattr(args, col)
    df_to_save = df_to_save[cols]

    if not os.path.exists(results_path):
        df_to_save.to_csv(results_path, index=False)
    else:
        pd.concat([pd.read_csv(results_path), df_to_save], ignore_index=True).to_csv(
            results_path, index=False
        )

    print(f"\nResults have been saved to: {results_path}")
    print("Appended new result row:")
    print(df_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, default=None, help="Path to the YAML config file."
    )
    parser.add_argument("--dataset", type=str, default="banking")
    parser.add_argument("--known_cls_ratio", type=float, default=0.25)
    parser.add_argument("--cluster_num_factor", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument("--num_pretrain_epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--labeled_ratio", type=float, default=1.0)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--fold_num", type=int, default=5)
    parser.add_argument(
        "--embedding_file", type=str, default="./pretrained_models/glove.6B.300d.txt"
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/openset/deepunk")

    parser.add_argument("--model_name_or_path", type=str, default=".")
    parser.add_argument(
        "--save_results_path", type=str, default="./results/openset/deepunk"
    )
    parser.add_argument(
        "--fold_type",
        type=str,
        default="fold",
        help="",
        choices=["imbalance_fold", "fold"],
    )

    args = parser.parse_args()

    def apply_config_updates(args, config_dict, parser):
        type_map = {action.dest: action.type for action in parser._actions}
        for key, value in config_dict.items():
            if f"--{key}" in sys.argv or not hasattr(args, key):
                continue
            expected_type = type_map.get(key)
            if expected_type and value is not None:
                try:
                    if expected_type is bool:
                        value = str(value).lower() in ("true", "1", "t", "yes")
                    else:
                        value = expected_type(value)
                except (ValueError, TypeError):
                    print(
                        f"Warning: Could not cast YAML value '{value}' for key '{key}' to type {expected_type}."
                    )
            setattr(args, key, value)

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        apply_config_updates(args, yaml_config, parser)
        if "dataset_specific_configs" in yaml_config:
            dataset_configs = yaml_config["dataset_specific_configs"].get(
                args.dataset, {}
            )
            apply_config_updates(args, dataset_configs, parser)

    main(args)
