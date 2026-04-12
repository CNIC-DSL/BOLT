#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_sim_splits.py (cluster-based hard split, in-place)

What it does
- DOES NOT create a new dataset directory.
- Reads the existing dataset at:  <data_root>/<dataset>/
- Keeps ALL instances unchanged (origin_data/, labeled_data/ untouched).
- Only writes a new split folder under: <dataset>/label/<out_fold_type><fold_num>/
  e.g. data/20NG/label/sim_fold5/part0/label_known_0.5.list

Core idea
1) Use TRAINING data only to compute a centroid vector for each label (class centroid).
2) Cluster label centroids into k semantic clusters (k-means on cosine similarity).
3) For each KCR, allocate the total number of known classes across clusters (roughly proportional),
   while trying to ensure each cluster (size>=2) has both known and unknown (hard unknown).
4) Pick the actual known labels within each cluster (closest-to-center or random).
5) Unknown labels are the complement: all_labels - known_labels (your loader already does this).

This matches your loader (A): it only needs label_known_*.list; unknown is computed as complement.
"""

import os
import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


# ---------------------------
# Text embedding utilities
# ---------------------------

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-6)                        # [B,1]
    return summed / counts


@torch.inference_mode()
def embed_texts(
    texts,
    tokenizer,
    model,
    batch_size=32,
    max_length=256,
    device="cpu",
):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        vecs.append(emb.cpu())
    return torch.cat(vecs, dim=0).numpy()


def build_class_centroids(
    train_tsv: str,
    all_labels: list[str],
    encoder_name_or_path: str,
    per_class_cap: int = 200,
    seed: int = 0,
    max_length: int = 256,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Use TRAINING DATA ONLY to compute a centroid vector for each label.
    Cap per label samples by per_class_cap to reduce compute and balance classes.
    """
    df = pd.read_csv(train_tsv, sep="\t")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns ['text','label'] in {train_tsv}, got {list(df.columns)}")

    rng = random.Random(seed)

    by_label = defaultdict(list)
    for _, row in df.iterrows():
        lab = str(row["label"])
        if lab in all_labels:
            by_label[lab].append(str(row["text"]))

    sampled_texts = {}
    for lab in all_labels:
        texts = by_label.get(lab, [])
        if not texts:
            raise ValueError(f"No training examples found for label: {lab}")
        if len(texts) > per_class_cap:
            rng.shuffle(texts)
            texts = texts[:per_class_cap]
        sampled_texts[lab] = texts

    tokenizer = AutoTokenizer.from_pretrained(encoder_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(encoder_name_or_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    centroids = {}
    for lab in all_labels:
        vecs = embed_texts(
            sampled_texts[lab],
            tokenizer,
            model,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        c = vecs.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids[lab] = c.astype(np.float32)

    return centroids


# ---------------------------
# Simple k-means (numpy) on label centroids
# ---------------------------

def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    k-means++ init on cosine distance (since X are L2-normalized, cosine sim = dot).
    Distance = 1 - dot. We clamp for numerical safety.
    """
    n, d = X.shape
    centers = np.empty((k, d), dtype=X.dtype)

    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]

    # Numerical safety: clamp dot into [-1, 1]
    dot0 = X @ centers[0].T
    dot0 = np.clip(dot0, -1.0, 1.0)
    closest_dist = 1.0 - dot0  # should be in [0, 2]

    for ci in range(1, k):
        # clamp to non-negative and build probability distribution
        closest_dist = np.maximum(closest_dist, 0.0)
        s = float(closest_dist.sum())

        if not np.isfinite(s) or s <= 1e-12:
            # fallback: if all distances are ~0 (duplicate points), pick random
            idx = rng.integers(0, n)
        else:
            probs = closest_dist / s
            # extra guard: remove negative/NaN just in case
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = np.maximum(probs, 0.0)
            probs = probs / (probs.sum() + 1e-12)
            idx = rng.choice(n, p=probs)

        centers[ci] = X[idx]

        dot_new = X @ centers[ci].T
        dot_new = np.clip(dot_new, -1.0, 1.0)
        dist_to_new = 1.0 - dot_new
        closest_dist = np.minimum(closest_dist, dist_to_new)

    return centers



def run_kmeans(
    X: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-means on cosine similarity.
    - Assignment: pick center with max dot (cosine).
    - Update: mean then re-normalize.
    Returns: (assignments [n], centers [k,d])
    """
    rng = np.random.default_rng(seed)
    n, _ = X.shape
    if k <= 1:
        return np.zeros((n,), dtype=np.int64), np.mean(X, axis=0, keepdims=True)

    k = min(k, n)  # safety
    centers = kmeans_pp_init(X, k, rng)

    prev_assign = None
    for _ in range(max_iter):
        sims = X @ centers.T  # [n,k]
        assign = np.argmax(sims, axis=1)
        if prev_assign is not None and np.array_equal(assign, prev_assign):
            break
        prev_assign = assign

        new_centers = np.zeros_like(centers)
        for ci in range(k):
            members = X[assign == ci]
            if len(members) == 0:
                new_centers[ci] = X[rng.integers(0, n)]
            else:
                c = members.mean(axis=0)
                c = c / (np.linalg.norm(c) + 1e-12)
                new_centers[ci] = c
        centers = new_centers

    return prev_assign, centers


# ---------------------------
# Allocate known labels per cluster for a given KCR
# ---------------------------

def allocate_known_counts_per_cluster(cluster_sizes: list[int], total_known: int) -> list[int]:
    """
    Allocate total_known across clusters proportionally to size, then adjust to match sum exactly.
    Enforce:
      - for clusters with size >= 2: 1 <= known_i <= size-1 (keep both known+unknown in cluster)
      - for size == 1: known_i can be 0 or 1
    """
    C = sum(cluster_sizes)
    raw = [sz * total_known / C for sz in cluster_sizes]
    base = [int(np.floor(x)) for x in raw]
    frac = [x - b for x, b in zip(raw, base)]

    remaining = total_known - sum(base)
    order = np.argsort(frac)[::-1].tolist()
    i = 0
    while remaining > 0 and order:
        base[order[i % len(order)]] += 1
        remaining -= 1
        i += 1

    # enforce constraints
    for i, sz in enumerate(cluster_sizes):
        if sz >= 2:
            base[i] = max(1, min(base[i], sz - 1))
        else:
            base[i] = 1 if base[i] >= 1 else 0

    def can_inc(idx):
        sz = cluster_sizes[idx]
        return base[idx] < (sz - 1 if sz >= 2 else 1)

    def can_dec(idx):
        sz = cluster_sizes[idx]
        return base[idx] > (1 if sz >= 2 else 0)

    while sum(base) < total_known:
        candidates = [i for i in range(len(base)) if can_inc(i)]
        if not candidates:
            break
        j = max(candidates, key=lambda x: (cluster_sizes[x], -base[x]))
        base[j] += 1

    while sum(base) > total_known:
        candidates = [i for i in range(len(base)) if can_dec(i)]
        if not candidates:
            break
        j = min(candidates, key=lambda x: (cluster_sizes[x], base[x]))
        base[j] -= 1

    # last-resort adjust
    if sum(base) != total_known:
        diff = total_known - sum(base)
        idxs = list(range(len(base)))
        if diff > 0:
            for _ in range(diff):
                for i in sorted(idxs, key=lambda x: -cluster_sizes[x]):
                    if base[i] < cluster_sizes[i]:
                        base[i] += 1
                        break
        else:
            for _ in range(-diff):
                for i in sorted(idxs, key=lambda x: cluster_sizes[x]):
                    if base[i] > 0:
                        base[i] -= 1
                        break

    return base


def pick_known_within_cluster(
    labels: list[str],
    label_vecs: dict[str, np.ndarray],
    k: int,
    method: str,
    rng: random.Random,
) -> list[str]:
    """
    Pick k labels from this cluster.
    - closest: pick labels closest to cluster centroid (cosine)
    - random: random pick (seeded)
    """
    if k <= 0:
        return []
    if k >= len(labels):
        return labels[:]

    if method == "random":
        labels = labels[:]
        rng.shuffle(labels)
        return labels[:k]

    V = np.stack([label_vecs[l] for l in labels], axis=0)
    center = V.mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)
    sims = V @ center

    # stable tie-break: tiny noise from rng
    sims2 = sims + (rng.random() * 1e-9)
    order = np.argsort(sims2)[::-1].tolist()
    return [labels[i] for i in order[:k]]


# ---------------------------
# Main
# ---------------------------

def parse_kcrs(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def auto_num_clusters(C: int) -> int:
    """
    Default: k = round(C/4), with safety bounds.
    """
    return max(2, min(C - 1, int(round(C / 4))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="e.g. /ssd/lijinpeng/code/bolt/data")
    ap.add_argument("--dataset", required=True, help="e.g. 20NG")

    ap.add_argument("--out_fold_type", default="sim_fold", help='folder prefix under label/, e.g. "sim_fold" -> sim_fold5')
    ap.add_argument("--fold_num", type=int, default=5)
    ap.add_argument("--parts", type=int, default=5)
    ap.add_argument("--kcrs", default="0.25,0.5,0.75")

    ap.add_argument("--encoder", required=True, help="HF model name or local path")
    ap.add_argument("--per_class_cap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--num_clusters", type=int, default=0,
                    help="k for k-means; if 0, use auto k=round(C/4)")
    ap.add_argument("--max_kmeans_iter", type=int, default=50)
    ap.add_argument("--intra_select", choices=["closest", "random"], default="closest")

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)

    ap.add_argument("--overwrite", action="store_true",
                    help="If set, overwrite existing label_known_*.list in output folder.")

    args = ap.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    all_label_path = os.path.join(dataset_dir, "label", "label.list")
    if not os.path.isfile(all_label_path):
        raise FileNotFoundError(f"label.list not found: {all_label_path}")

    all_labels = pd.read_csv(all_label_path, header=None)[0].astype(str).tolist()
    C = len(all_labels)
    if C < 2:
        raise ValueError(f"Need at least 2 labels, got {C}")

    train_tsv = os.path.join(dataset_dir, "origin_data", "train.tsv")
    if not os.path.isfile(train_tsv):
        raise FileNotFoundError(f"origin_data/train.tsv not found: {train_tsv}")

    # Build label centroids (TRAIN only)
    label_vecs = build_class_centroids(
        train_tsv=train_tsv,
        all_labels=all_labels,
        encoder_name_or_path=args.encoder,
        per_class_cap=args.per_class_cap,
        seed=args.seed,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    X = np.stack([label_vecs[l] for l in all_labels], axis=0)  # [C,H], normalized

    kcrs = parse_kcrs(args.kcrs)

    k = args.num_clusters if args.num_clusters and args.num_clusters > 0 else auto_num_clusters(C)
    if k >= C:
        k = max(2, C - 1)

    out_fold_dir = os.path.join(dataset_dir, "label", f"{args.out_fold_type}{args.fold_num}")
    os.makedirs(out_fold_dir, exist_ok=True)

    print(f"[Info] Dataset: {dataset_dir}")
    print(f"[Info] Labels: C={C}, k-means clusters k={k} (override with --num_clusters)")
    print(f"[Info] Output split folder: {out_fold_dir}")
    print(f"[Info] KCRs: {kcrs} | parts: {args.parts} | intra_select: {args.intra_select}")

    for part_idx in range(args.parts):
        rng = random.Random(args.seed + part_idx * 1000)

        # cluster labels (vary seed per part for diversity)
        assign, _ = run_kmeans(
            X,
            k=k,
            seed=args.seed + part_idx,
            max_iter=args.max_kmeans_iter,
        )

        clusters = defaultdict(list)
        for li, ci in enumerate(assign):
            clusters[int(ci)].append(all_labels[li])

        cluster_ids = sorted(clusters.keys())
        cluster_lists = [clusters[cid] for cid in cluster_ids]
        cluster_sizes = [len(lst) for lst in cluster_lists]

        part_dir = os.path.join(out_fold_dir, f"part{part_idx}")
        os.makedirs(part_dir, exist_ok=True)

        for kcr in kcrs:
            total_known = int(round(C * kcr))
            # avoid degenerate all-known or none-known
            total_known = max(1, min(total_known, C - 1))

            known_counts = allocate_known_counts_per_cluster(cluster_sizes, total_known)

            known_labels = []
            for lst, ki in zip(cluster_lists, known_counts):
                chosen = pick_known_within_cluster(
                    labels=lst,
                    label_vecs=label_vecs,
                    k=ki,
                    method=args.intra_select,
                    rng=rng,
                )
                known_labels.extend(chosen)

            # unique + adjust to exact size
            known_set = list(dict.fromkeys(known_labels))
            if len(known_set) != total_known:
                all_set = set(all_labels)
                kset = set(known_set)
                if len(known_set) < total_known:
                    need = total_known - len(known_set)
                    candidates = list(all_set - kset)
                    rng.shuffle(candidates)
                    known_set.extend(candidates[:need])
                else:
                    drop = len(known_set) - total_known
                    rng.shuffle(known_set)
                    known_set = known_set[drop:]

            out_path = os.path.join(part_dir, f"label_known_{kcr}.list")
            if os.path.exists(out_path) and not args.overwrite:
                raise FileExistsError(
                    f"{out_path} exists. Use --overwrite to overwrite, or change --out_fold_type."
                )

            with open(out_path, "w", encoding="utf-8") as f:
                for lab in known_set:
                    f.write(lab + "\n")

    print("[Done] Wrote sim splits in-place.")
    print(f"Use: fold_type={args.out_fold_type}, fold_num={args.fold_num}, dataset={args.dataset}")
    print(f"Example known-list path: label/{args.out_fold_type}{args.fold_num}/part0/label_known_0.5.list")


if __name__ == "__main__":
    main()
