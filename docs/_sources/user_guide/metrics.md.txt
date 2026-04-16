# Metrics Guide

This page explains the meaning of common metrics in the summary table and how to read them. Different tasks use different metric fields; the following describes the most common conventions.

## 1. Summary Table Field Structure

`summary_{result_file}.csv` typically contains two types of fields:
- Experiment setting fields: `method`, `dataset`, `known_cls_ratio`, `labeled_ratio`, `cluster_num_factor`, `seed` (and possibly fold information and extra hyperparameters)
- Metric fields: vary by task (e.g., `ACC`, `H-Score`, `F1`, etc.)
- `args`: a JSON string recording key run parameters (used for reproducibility and deduplication)

Recommended reading approaches:
- Horizontal comparison: fix `dataset` / ratios / folds, and compare metric differences across methods
- Vertical comparison: fix `method`, and compare stability and trends across ratios / seeds / folds

## 2. Common GCD Metrics (Examples)

- `ACC`  
  Overall clustering/classification accuracy (the exact definition depends on the method implementation, and it typically reflects overall assignment quality).

- `H-Score`  
  A composite metric designed to balance performance on Known classes and Novel classes; commonly used as an overall evaluation score.

- `K-ACC`  
  Accuracy on Known classes.

- `N-ACC`  
  Accuracy on Novel classes.

- `ARI` (Adjusted Rand Index)  
  Measures the agreement between clustering results and ground-truth labels, adjusted for chance agreement.

- `NMI` (Normalized Mutual Information)  
  Measures mutual-information consistency between clusters and ground-truth labels, typically in the range [0, 1].

Reading tips:
- If a method has high `K-ACC` but low `N-ACC`, it usually indicates weak novel-class discovery ability.
- `H-Score` is often more suitable for overall ranking, but you should still interpret it together with the decomposed K/N metrics to understand the reasons.

## 3. Common OpenSet Metrics (Examples)

- `F1`  
  Overall F1 score (the harmonic mean of precision and recall), used as a general-purpose evaluation metric.

- `K-F1`  
  F1 score on Known classes.

- `N-F1`  
  F1 score on Novel / Unknown classes.

Reading tips:
- There is often a trade-off between `K-F1` and `N-F1`.
- Looking only at overall `F1` may hide weaknesses in unknown-class recognition, so it is recommended to check `K-F1` and `N-F1` at the same time.

## 4. Relation to Method Implementations

Different methods may write additional columns in `results.csv` (e.g., loss, runtime, extra hyperparameters).
During aggregation, the framework typically keeps core fields and appends `args` to ensure traceability.
If you need extra metrics to appear in the summary table, it is recommended to standardize the result column names on the method side and ensure they are written to `results/{task}/{method}/results.csv`.
