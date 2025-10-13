import numpy as np
import numpy.linalg as LA


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x

    try:
        return x.numpy()
    except Exception:
        pass

    try:
        return np.asarray(x)
    except Exception:
        pass

    return np.array(x)


def l2_normalize_rows(X, eps=1e-12):
    X = _to_numpy(X).astype(np.float32, copy=False)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def compute_centroids_numpy(X, y):
    X = _to_numpy(X)
    y = _to_numpy(y)
    classes = np.unique(y)
    centroids = []
    for c in classes:
        mask = y == c
        if not np.any(mask):
            centroids.append(np.zeros((X.shape[1],), dtype=np.float32))
        else:
            centroids.append(np.mean(X[mask], axis=0))
    return np.vstack(centroids).astype(np.float32), classes


def pairwise_euclidean(A, B):
    A = _to_numpy(A).astype(np.float32, copy=False)
    B = _to_numpy(B).astype(np.float32, copy=False)

    aa = np.sum(A * A, axis=1, keepdims=True)
    bb = np.sum(B * B, axis=1, keepdims=True).T
    ab = A @ B.T
    d2 = np.maximum(aa + bb - 2.0 * ab, 0.0)
    return np.sqrt(d2, dtype=np.float32)


def find_best_radius_numpy(X_train, y_train, centroids, alpha=1.0, step_size=0.01):
    X = _to_numpy(X_train).astype(np.float32, copy=False)
    y = _to_numpy(y_train)
    C = _to_numpy(centroids).astype(np.float32, copy=False)

    classes = np.unique(y)
    num_classes = len(classes)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    radius = np.zeros(shape=(num_classes,), dtype=np.float32)

    for i, c in enumerate(classes):

        dists_sel = LA.norm(X - C[i], axis=1)

        id_mask = (y == c).astype(np.float32)
        ood_mask = (y != c).astype(np.float32)

        id_cnt = np.sum(id_mask)
        ood_cnt = np.sum(ood_mask)

        if id_cnt == 0:

            radius[i] = 0.0
            continue

        per = ood_cnt / id_cnt

        while radius[i] < 2.0:
            ood_criterion = (dists_sel - radius[i]) * ood_mask
            id_criterion = (radius[i] - dists_sel) * id_mask

            crit = np.mean(ood_criterion) - (np.mean(id_criterion) * per / float(alpha))

            if crit < 0:

                radius[i] -= step_size
                break
            radius[i] += step_size

        if radius[i] < 0:
            radius[i] = 0.0

    return radius, classes


class ADBThreshold:

    def __init__(self, alpha=1.0, step_size=0.01, oos_label=-1):
        self.radius = None
        self.centroids = None
        self.classes = None
        self.oos_label = oos_label
        self.alpha = alpha
        self.step_size = step_size

    def fit(self, X_train, y_train):

        X_train = l2_normalize_rows(X_train)
        y_train = _to_numpy(y_train)

        centroids, classes = compute_centroids_numpy(X_train, y_train)
        centroids = l2_normalize_rows(centroids)

        radius, classes_checked = find_best_radius_numpy(
            X_train, y_train, centroids, alpha=self.alpha, step_size=self.step_size
        )
        assert np.array_equal(classes, classes_checked), "Class ordering mismatch."

        self.centroids = centroids
        self.radius = radius
        self.classes = classes

    def predict(self, X_test):

        X_test = l2_normalize_rows(X_test)

        dmat = pairwise_euclidean(X_test, self.centroids)

        predictions_idx = np.argmin(dmat, axis=1)

        predictions = self.classes[predictions_idx]

        chosen_centroids = self.centroids[predictions_idx]
        chosen_radius = self.radius[predictions_idx]
        d = np.linalg.norm(X_test - chosen_centroids, axis=1)
        predictions = np.where(d < chosen_radius, predictions, self.oos_label)

        return predictions

    def predict_proba(self, X_test):
        raise NotImplementedError(
            "Adaptive Decision Boundary Threshold 仅在 ood_train 中使用。"
        )
