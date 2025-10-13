from configs import args
import os
import pandas as pd
import torch
from tqdm import tqdm
import logging
from model import Model
from load_dataset import test_loader
from sklearn.cluster import KMeans
from utils import clustering_score
import numpy as np
import json

if __name__ == "__main__":
    model = Model(args).to(args.device)
    with torch.no_grad():
        model.eval()
        preds, golds, logits, features = [], [], [], []
        for batch in tqdm(test_loader):
            y = batch["labels"].to(args.device)
            batch = {i: v.to(args.device) for i, v in batch.items() if i != "labels"}
            logit = model(batch)
            feature = model.features(batch)
            pred = logit.max(dim=1).indices
            preds.append(pred)
            logits.append(logit)
            golds.append(y)
            features.append(feature)
        logits = torch.concat(logits).detach().cpu().numpy()
        preds = torch.concat(preds).detach().cpu().numpy()
        golds = torch.concat(golds).detach().cpu().numpy()
        features = torch.concat(features).detach().cpu().numpy()

    kmeans = KMeans(n_clusters=args.all_num_labels)
    kmeans.fit(features)
    y_kmeans = kmeans.predict(features)
    metrics = clustering_score(golds, y_kmeans, np.arange(args.num_labels))

    record = dict(metrics)
    record = {
        "method": "PLM_GCD",
        "dataset": args.dataset,
        "known_cls_ratio": args.known_cls_ratio,
        "labeled_ratio": args.labeled_ratio,
        "cluster_num_factor": 1,
        "seed": args.seed,
        "ACC": metrics.get("ACC", 0),
        "H-Score": metrics.get("H-Score", 0),
        "K-ACC": metrics.get("K-ACC", 0),
        "N-ACC": metrics.get("N-ACC", 0),
        "ARI": metrics.get("ARI", 0),
        "NMI": metrics.get("NMI", 0),
        "args": json.dumps(
            {k: str(v) for k, v in vars(args).items() if k != "device"},
            ensure_ascii=False,
        ),
    }

    df = pd.DataFrame([record])

    os.makedirs(os.path.dirname(args.metric_file), exist_ok=True)
    write_header = (not os.path.exists(args.metric_file)) or os.path.getsize(
        args.metric_file
    ) == 0

    df.to_csv(
        args.metric_file,
        sep=",",
        index=False,
        mode="a",
        header=write_header,
        encoding="utf-8",
    )
