import copy
import os.path
import webbrowser
from pathlib import PurePosixPath
from typing import Dict, Any, List, Optional
from os import listdir
from os.path import isfile, join
import statistics

import pandas as pd


def get_one_result(path: str) -> Dict[str, Dict[int, Any]]:
    df = pd.read_csv(path)
    return df.to_dict()


def filter_files(root: str, prefix: str, postfix: str) -> List[str]:

    answer = []

    if not os.path.isdir(root):
        return answer

    for directory_name in listdir(root):
        file_path = join(root, directory_name)
        if (
            isfile(file_path)
            and directory_name.startswith(prefix)
            and PurePosixPath(directory_name).stem.endswith(postfix)
        ):
            answer.append(file_path)

    return answer


def valid_all_datasets(paths: List[str]) -> None:

    for path in paths:
        one_result = get_one_result(path)

        column_nums = len(one_result.keys())
        columns = list(one_result.keys())
        row_nums = len(one_result["strategy"].values())
        rows = [
            (a, b)
            for a, b in zip(
                one_result["strategy"].values(), one_result["layer"].values()
            )
        ]

        break
    else:
        raise ValueError("no file")

    for path in paths:

        cur = get_one_result(path)

        assert column_nums == len(cur.keys()), f"行列有问题"
        assert row_nums == len(cur["layer"].values()), f"行列有问题"
        assert columns == list(cur.keys())
        assert rows == [
            (a, b) for a, b in zip(cur["strategy"].values(), cur["layer"].values())
        ]


def display_in_browser(df: pd.DataFrame, dataset_name: str) -> None:

    html = df.to_html()
    path = os.path.abspath(f"../tmp/{dataset_name}.html")
    url = "file://" + path

    with open(path, "w") as f:
        f.write(html)
    webbrowser.open(url)


def check(
    root: str,
    datasets_names: Optional[List[str]] = None,
    wanted_unique_strategy_name: str = "esm_-1",
):

    norm_scores_methods = [
        "knn",
        "lof_cosine",
    ]

    if datasets_names is None:
        datasets_names = [
            "banking_0.25",
            "banking_0.75",
            "clinc_0.25",
            "clinc_0.75",
            "clincnooos_0.25",
            "clincnooos_0.75",
            "clinc_full_0",
            "clinc_small_0",
            "mcid_0.25",
            "mcid_0.75",
            "stackoverflow_0.25",
            "stackoverflow_0.75",
        ]

    unique_strategy_names = ["esm_-1", "each_layer_2", "each_layer_1"]

    seeds = [0, 1, 2, 3, 4]

    metric_names = ["speedup", "ACC-all", "F1", "F1-ood", "F1-ind"]

    all_paths = [
        path
        for norm_scores_method in norm_scores_methods
        for dataset_name in datasets_names
        for path in filter_files(root, f"{norm_scores_method}_{dataset_name}", "")
    ]

    datasets_names = [
        dataset_name
        for dataset_name in datasets_names
        if any(dataset_name in path for path in all_paths)
    ]

    if len(all_paths) == 0:
        return

    valid_all_datasets(all_paths)

    files = {path: get_one_result(path) for path in all_paths}

    files_results = {}
    for dataset_name in datasets_names:
        for unique_strategy_name in unique_strategy_names:
            for norm_scores_method in norm_scores_methods:

                metrics = {}

                for seed in seeds:
                    path = os.path.join(
                        root, f"{norm_scores_method}_{dataset_name}_{seed}.csv"
                    )
                    file_result = files[path]

                    line_nums = max(file_result["strategy"].keys())

                    for line_num in range(0, line_nums + 1):
                        cur_line_unique_strategy_name = f'{file_result["strategy"][line_num]}_{file_result["layer"][line_num]}'
                        if cur_line_unique_strategy_name == unique_strategy_name:
                            wanted_line_num = line_num
                            break
                    else:
                        raise ValueError(f"NO {unique_strategy_name}")

                    for metric_name in metric_names:
                        if metric_name not in metrics:
                            metrics[metric_name] = []
                        metrics[metric_name].append(
                            file_result[metric_name][wanted_line_num]
                        )

                files_results[
                    f"{dataset_name}+{unique_strategy_name}+{norm_scores_method}"
                ] = metrics

    wanted_norm_scores_methods = ["lof_cosine"]

    for dataset_name in datasets_names:
        df_answer: Dict[str, Dict[str, Any]] = {}
        for norm_scores_method in wanted_norm_scores_methods:
            cur_result = files_results[
                f"{dataset_name}+{wanted_unique_strategy_name}+{norm_scores_method}"
            ]
            for metric_name in metric_names:
                if metric_name not in df_answer:
                    df_answer[metric_name] = {}

                assert len(cur_result[metric_name]) == 3

                if isinstance(cur_result[metric_name][0], str):
                    assert (
                        cur_result[metric_name][0] == cur_result[metric_name][1]
                        and cur_result[metric_name][1] == cur_result[metric_name][2]
                    )
                    df_answer[metric_name][f"{norm_scores_method}_avg"] = cur_result[
                        metric_name
                    ][0]
                    df_answer[metric_name][f"{norm_scores_method}_var"] = "NONE"
                else:
                    df_answer[metric_name][f"{norm_scores_method}_avg"] = (
                        statistics.mean(cur_result[metric_name])
                    )
                    df_answer[metric_name][f"{norm_scores_method}_var"] = (
                        statistics.variance(cur_result[metric_name])
                    )

                df_answer[metric_name][f"-{norm_scores_method}-"] = "-"

        df_answer_ = copy.deepcopy(df_answer)
        for key0, values0 in df_answer.items():
            for key1, value1 in values0.items():
                if isinstance(value1, float):
                    df_answer_[key0][f"{key1}:.2f"] = f"{round(value1, 2):.2f}"
        df_answer = copy.deepcopy(df_answer_)

        df = pd.DataFrame(df_answer)

        print(dataset_name, root)
        print(df)

        display_in_browser(df, f"{dataset_name}")

        print("-" * 80)


def main():
    pd.options.display.max_rows = 500
    pd.options.display.max_columns = 500
    pd.options.display.expand_frame_repr = False

    bests = {
        "banking_0.25": "./model_output/banking_0.25/ad30.10.2_lr2.0e-05__epoch50__lossce_and_div_drop-last-layer__batchsize16__lambda0.1__scale1.51.2",
        "banking_0.75": "./model_output/banking_0.75/ad30.10.2_lr2.0e-05__epoch50__lossce_and_div__batchsize32__lambda0.1__scale1.51.4",
        "stackoverflow_0.25": "./model_output/stackoverflow_0.25/ad20.150.4_lr2.0e-05__epoch50__lossce_and_div_drop-last-layer__batchsize32__lambda0.1__scale1.51.4",
        "stackoverflow_0.75": "./model_output/stackoverflow_0.75/ad30.10.3_lr2.0e-05__epoch50__lossce_and_div__batchsize32__lambda0.1__scale1.41.5",
        "clinc_0.25": "./model_output/clinc_0.25/ad30.150.4_lr5.0e-06__epoch50__lossce_and_div_drop-last-layer__batchsize32__lambda0.1__scale1.81.7",
        "clinc_0.75": "./model_output/clinc_0.75/ad30.10.2_lr5.0e-05__epoch50__lossce_and_div_drop-last-layer__batchsize32__lambda0.1__scale1.5",
    }

    for dataset_name, path in bests.items():

        check(path, [dataset_name], "esm_-1")

        print("#" * 80)


if __name__ == "__main__":
    main()
