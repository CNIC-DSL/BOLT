from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    (已根据SOP进行标准化改造 - 最终版)
    """
    # --- 步骤1: 先定义所有【没有】默认值的参数 ---
    dataset: str = field(
        metadata={"help": "The name of the dataset to use."}
    )
    known_cls_ratio: float = field(
        metadata={"help": "The ratio of known classes."}
    )

    # --- 步骤2: 再定义所有【有】默认值的参数 ---
    data_dir: str = field(
        default="./data",
        metadata={"help": "The input data dir."}
    )
    labeled_ratio: float = field(
        default=1.0,
        metadata={"help": "The ratio of labeled data to use."}
    )
    fold_idx: int = field(
        default=0,
        metadata={"help": "The index of the fold for cross-validation."}
    )
    fold_num: int = field(
        default=5,
        metadata={"help": "The total number of folds for cross-validation."}
    )
    # --- 新增：补上遗漏的 gpu_id 参数 ---
    gpu_id: str = field(
        default="0",
        metadata={"help": "The GPU ID to use."}
    )

    # --- 保留原有的其他非冲突参数 ---
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help": "FORCE Overwrite the cached preprocessed datasets or not."}
    )

@dataclass
class OtherArguments:
    """
    模型的训练参数、其他超参等 可以从 json 里面设置，其余的均为自动生成或者写死
    """
    # --- 将所有打算从 YAML 加载的必需参数，全部改为 Optional 并设置 default=None ---
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # num_train_epochs: Optional[int] = field(
    #     default=None,
    #     metadata={"help": "训练几个 epoch"}
    # )
    loss_type: Optional[str] = field(
        default=None,
        metadata={"help": "损失函数形式"}
    )
    diversity_loss_weight: Optional[float] = field(
        default=None,
        metadata={"help": "diversity_loss 权重"}
    )
    scale: Optional[float] = field(
        default=None,
        metadata={"help": "ensemble scale_ind 参数"}
    )
    adv_k: Optional[int] = field(
        default=None,
        metadata={"help": "Adv k"}
    )
    adv_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Adv lr"}
    )
    adv_init_mag: Optional[float] = field(
        default=None,
        metadata={"help": "Adv init mag"}
    )
    adv_max_norm: Optional[float] = field(
        default=None,
        metadata={"help": "Adv max norm"}
    )

    # --- 其他本身就有默认值的字段保持不变 ---
    config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the YAML config file for base configuration."}
    )
    scale_ood: float = field(
        default=-1,
        metadata={"help": "ensemble scale_ood 参数"}
    )

    ########################################################################
    # 写死

    cache_dir: Optional[str] = field(
        default=None,
        init=False,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    fitlog_dir: str = field(
        default="./logs",
        init=False
    )

    save_results_path: str = field(
        default="./results/openset/dyen",
    )

    clip: float = field(
        default=0.25,
        init=False
    )
    cluster_num_factor: float = field(
        default=1.0,
    )

    # TODO: 可能未来支持?
    # load_trained_model: bool = field(
    #     default=False
    # )

    def __post_init__(self):
        # 向下兼容（早期版本，scale_ind 和 scale_ood 不分家的）
        if self.scale_ood == -1:
            self.scale_ood = self.scale


@dataclass
class FitLogArguments:
    task: str = field(default='AUC', init=False)
