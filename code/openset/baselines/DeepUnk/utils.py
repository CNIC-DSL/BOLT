import pandas as pd
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
import random as rn
import os


def set_allow_growth(device="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth set to True for devices: {gpus}")
        except RuntimeError as e:
            print(e)


def get_score(cm):
    fs = []
    n_class = cm.shape[0]
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        fs.append(f * 100)

    f = np.mean(fs).round(2)
    f_seen = np.mean(fs[:-1]).round(2)
    f_unseen = round(fs[-1], 2)
    print("Overall(macro): ", f)
    print("Seen(macro): ", f_seen)
    print("=====> Uneen(Experiment) <=====: ", f_unseen)

    return f, f_seen, f_unseen


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    figsize=(12, 10),
    cmap=plt.cm.Blues,
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    np.set_printoptions(precision=2)
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("img/mat.png")
