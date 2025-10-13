import argparse
import os
import logging


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="stackoverflow", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument(
        "--reg_loss", default="npo", type=str, choices=["normal", "vos", "npo"]
    )
    parser.add_argument("--known_cls_ratio", default=0.25, type=float)
    parser.add_argument("--labeled_ratio", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--fold_idx", default=0, type=int)
    parser.add_argument("--fold_num", default=5, type=int)
    parser.add_argument("--fold_type", default="fold", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--root", default="data", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--backbone", default="Meta-Llama-3.1-8B-Instruct", type=str)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--cluster_num_factor", default=1.0, type=float)
    parser.add_argument(
        "--save_results_path", default="results/openset/plm_ood", type=str
    )
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="Patience for early stopping.",
    )
    parser.add_argument(
        "--early_stop_delta",
        type=float,
        default=0.0,
        help="Minimum change to qualify as an improvement for early stopping.",
    )
    return parser


def finalize_config(args):

    args.data_identity = f"{args.dataset}_{args.labeled_ratio}_{args.known_cls_ratio}_{args.fold_num}_{args.fold_idx}"
    run_identity = f"{args.data_identity}{args.backbone}_{args.seed}"

    base_path = os.path.join(
        args.output_dir,
        args.reg_loss,
        args.dataset,
        str(args.labeled_ratio),
        run_identity,
    )

    args.log_dir = os.path.join(base_path, "logs")
    args.checkpoint_path = base_path
    args.case_path = os.path.join(base_path, "case_study")
    args.vector_path = os.path.join(base_path, "case_study")

    os.makedirs(args.save_results_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.case_path, exist_ok=True)

    args.metric_file = f"{args.save_results_path}/results.csv"

    if args.model_path is None:
        logging.warning(
            "model_path not specified, generating a fallback path. It is recommended to specify this in the YAML file."
        )
        args.model_path = f"./pretrained_models/{args.backbone}"
    else:
        logging.info(f"Using model_path specified in config: {args.model_path}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"{args.log_dir}/epoch_{args.num_train_epochs}_seed_{args.seed}.log",
                mode="w",
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logging.info(f"Final configuration: {args}")

    return args
