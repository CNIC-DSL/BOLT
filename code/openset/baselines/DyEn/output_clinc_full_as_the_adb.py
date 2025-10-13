import os

import pandas as pd


def read_tsv_file(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep="\t", dtype=str)
    return df


def save_tsv_file(filename: str, df: pd.DataFrame) -> None:
    df.to_csv(filename, sep="\t", index=False)


def main():
    original_root = "./data/clinc_full"

    new_root = "./data/clinc"
    if not os.path.exists(new_root):
        os.makedirs(new_root)

    data = read_tsv_file(os.path.join(original_root, "train.tsv"))

    assert data.label.isin(["oos", "ood"]).values.sum() == 0
    save_tsv_file(os.path.join(new_root, "train.tsv"), data)

    data = read_tsv_file(os.path.join(original_root, "valid.tsv"))

    assert data.label.isin(["oos", "ood"]).values.sum() == 0
    save_tsv_file(os.path.join(new_root, "valid.tsv"), data)

    knn_con_train_oos_data = read_tsv_file(os.path.join(original_root, "train_oos.tsv"))
    knn_con_valid_oos_data = read_tsv_file(os.path.join(original_root, "valid_oos.tsv"))
    knn_con_test_data = read_tsv_file(os.path.join(original_root, "test.tsv"))

    knn_con_test_oos_indices = knn_con_test_data.label.isin(["oos"])
    new_test_data = pd.concat(
        [
            knn_con_test_data[~knn_con_test_oos_indices],
            knn_con_train_oos_data,
            knn_con_valid_oos_data,
            knn_con_test_data[knn_con_test_oos_indices],
        ],
        ignore_index=True,
    )

    save_tsv_file(os.path.join(new_root, "test.tsv"), new_test_data)


if __name__ == "__main__":
    main()
