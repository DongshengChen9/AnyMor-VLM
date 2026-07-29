"""
Morphometrics-only semi-supervised K-Means baseline.

This script removes the AutoEncoder and all model training. It obtains the
unique_id of each labelled training region and each test region from the
existing dataset loaders, looks up the corresponding standardized morphology
features in an .npz file, and evaluates seeded semi-supervised K-Means.

Expected NPZ keys:
    ids           : shape (N,)
    X             : shape (N, D)
    feature_names : shape (D,)  (optional but recommended)
"""

import argparse
import json
import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

from config import exp_root, get_dir
from data.get_datasets import get_class_splits, get_datasets
from util.cluster_and_log_utils import log_accs_from_preds_v4
from util.general_utils import init_experiment


class MetadataOnlyTransform:
    """Return a tiny placeholder because morphology features replace images."""

    def __call__(self, _image):
        return torch.zeros(1, dtype=torch.uint8)


def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_class_split_info(args):
    """Load or generate the same class split used by the original baseline."""

    if not args.random_class_split:
        if not os.path.exists(args.class_split_file):
            raise FileNotFoundError(
                f"Class split file does not exist: {args.class_split_file}"
            )

        with open(args.class_split_file, "r", encoding="utf-8") as f:
            split_info = json.load(f)

        args.train_classes = split_info["train_classes"]
        args.unlabeled_classes = split_info["unlabeled_classes"]
        args.train_labelled_classes = split_info["train_labelled_classes"]
        args.train_unlabelled_classes = split_info["train_unlabelled_classes"]
        return args

    rng = np.random.RandomState(args.seed)
    all_classes = np.arange(args.num_classes4test)
    rng.shuffle(all_classes)

    num_train_classes = int(args.num_classes4test / 3 * 2)
    num_train_labelled_classes = int(num_train_classes / 2)

    train_classes = np.sort(all_classes[:num_train_classes])
    novel_classes = np.sort(all_classes[num_train_classes:])

    labelled_classes = np.sort(
        rng.choice(
            train_classes,
            size=num_train_labelled_classes,
            replace=False,
        )
    )
    unlabelled_train_classes = np.sort(
        np.setdiff1d(train_classes, labelled_classes)
    )
    unlabeled_classes = sorted(
        novel_classes.tolist() + unlabelled_train_classes.tolist()
    )

    split_info = {
        "train_classes": train_classes.tolist(),
        "unlabeled_classes": unlabeled_classes,
        "train_labelled_classes": labelled_classes.tolist(),
        "train_unlabelled_classes": unlabelled_train_classes.tolist(),
    }

    os.makedirs("data", exist_ok=True)
    counter = 1
    while True:
        class_split_path = os.path.join(
            "data", f"class_split_info_run{counter}.json"
        )
        if not os.path.exists(class_split_path):
            break
        counter += 1

    with open(class_split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    args.class_split_file = class_split_path
    args.train_classes = split_info["train_classes"]
    args.unlabeled_classes = split_info["unlabeled_classes"]
    args.train_labelled_classes = split_info["train_labelled_classes"]
    args.train_unlabelled_classes = split_info["train_unlabelled_classes"]
    return args


def _canonical_id(value) -> str:
    """Normalize tensor/NumPy/Python IDs to a stable dictionary key."""

    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"unique_id must be scalar, got shape {tuple(value.shape)}")
        value = value.detach().cpu().item()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, str):
        text = value.strip()
        # Make integer-like strings consistent with integer/tensor IDs.
        try:
            numeric = float(text)
            if np.isfinite(numeric) and numeric.is_integer():
                return str(int(numeric))
        except ValueError:
            pass
        return text

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"unique_id cannot be non-finite: {value}")
        if float(value).is_integer():
            return str(int(value))

    return str(value)


def _to_numpy_1d(values, name: str) -> np.ndarray:
    """Convert a collated DataLoader field into a one-dimensional array."""

    if torch.is_tensor(values):
        array = values.detach().cpu().numpy()
    elif isinstance(values, np.ndarray):
        array = values
    else:
        array = np.asarray(values)

    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1:
        # mask_lab is commonly returned as shape (B, 1).
        if array.shape[1:] == (1,):
            array = array[:, 0]
        else:
            array = array.reshape(-1)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def load_morphology_npz(
    npz_path: str,
    nan_policy: str = "error",
) -> Tuple[np.ndarray, Dict[str, int], np.ndarray]:
    """Load morphology features and build a unique_id -> row lookup."""

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Morphology feature file does not exist: {npz_path}")

    feature_names = None
    with np.load(npz_path, allow_pickle=True) as data:
        required = {"ids", "X"}
        missing_keys = required.difference(data.files)
        if missing_keys:
            raise KeyError(
                f"NPZ is missing required key(s): {sorted(missing_keys)}; "
                f"available keys: {data.files}"
            )

        ids = np.asarray(data["ids"]).reshape(-1)
        X = np.asarray(data["X"], dtype=np.float64)
        if "feature_names" in data.files:
            try:
                feature_names = np.asarray(
                    data["feature_names"]
                ).reshape(-1).astype(str)
            except ModuleNotFoundError:
                print(
                    "Warning: feature_names cannot be loaded because of "
                    "NumPy version incompatibility. Default names will be used."
                )
                feature_names = None

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D feature matrix, got shape {X.shape}")

    if feature_names is None:
        feature_names = np.asarray(
            [f"feature_{i}" for i in range(X.shape[1])], dtype=object
        )

    if len(ids) != len(X):
        raise ValueError(f"ids and X have different lengths: {len(ids)} vs {len(X)}")
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            "feature_names length does not match X feature dimension: "
            f"{len(feature_names)} vs {X.shape[1]}"
        )
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"X cannot be empty, got shape {X.shape}")

    nonfinite = ~np.isfinite(X)
    if np.any(nonfinite):
        count = int(nonfinite.sum())
        if nan_policy == "error":
            bad_rows, bad_cols = np.where(nonfinite)
            examples = [
                f"id={ids[r]!r}, feature={feature_names[c]!r}"
                for r, c in zip(bad_rows[:10], bad_cols[:10])
            ]
            raise ValueError(
                f"Morphology matrix contains {count} NaN/Inf values. "
                f"Examples: {examples}. Use --nan_policy mean or zero to impute."
            )
        if nan_policy == "zero":
            X = X.copy()
            X[nonfinite] = 0.0
        elif nan_policy == "mean":
            X = X.copy()
            finite_X = np.where(np.isfinite(X), X, np.nan)
            column_means = np.nanmean(finite_X, axis=0)
            column_means = np.where(np.isfinite(column_means), column_means, 0.0)
            rows, cols = np.where(nonfinite)
            X[rows, cols] = column_means[cols]
        else:
            raise ValueError(f"Unknown nan_policy: {nan_policy}")

    id_to_row: Dict[str, int] = {}
    duplicate_ids: List[str] = []
    for row, raw_id in enumerate(ids):
        key = _canonical_id(raw_id)
        if key in id_to_row:
            duplicate_ids.append(key)
        else:
            id_to_row[key] = row

    if duplicate_ids:
        preview = sorted(set(duplicate_ids))[:20]
        raise ValueError(
            "NPZ ids must be unique after normalization. Duplicate ID examples: "
            f"{preview}"
        )

    return X, id_to_row, feature_names


def lookup_embeddings(
    unique_ids: Sequence,
    X: np.ndarray,
    id_to_row: Dict[str, int],
    missing_id_policy: str,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Look up feature rows while preserving the input sample order."""

    rows: List[int] = []
    keep_mask = np.zeros(len(unique_ids), dtype=bool)
    missing: List[str] = []

    for sample_index, raw_id in enumerate(unique_ids):
        key = _canonical_id(raw_id)
        row = id_to_row.get(key)
        if row is None:
            missing.append(key)
            continue
        rows.append(row)
        keep_mask[sample_index] = True

    if missing and missing_id_policy == "error":
        preview = missing[:20]
        raise KeyError(
            f"{len(missing)} {split_name} unique_id(s) were not found in the NPZ. "
            f"Examples: {preview}"
        )
    if missing_id_policy not in {"error", "skip"}:
        raise ValueError(f"Unknown missing_id_policy: {missing_id_policy}")
    if not rows:
        raise ValueError(
            f"No {split_name} samples could be matched to morphology features"
        )

    return X[np.asarray(rows, dtype=int)], keep_mask, missing


def kmeanspp_pick_centroids(
    X_unlabeled: np.ndarray,
    existing_centroids: np.ndarray,
    n_to_pick: int,
    random_state: int = 0,
) -> np.ndarray:
    """Pick new centroids with K-Means++ while accounting for seen centroids."""

    rng = np.random.RandomState(random_state)
    n_samples, dim = X_unlabeled.shape

    if n_to_pick <= 0:
        return np.zeros((0, dim), dtype=X_unlabeled.dtype)
    if n_samples == 0:
        raise ValueError("Cannot initialize unseen centroids from an empty array")

    if existing_centroids is None or len(existing_centroids) == 0:
        first_idx = rng.randint(0, n_samples)
        centroids = [X_unlabeled[first_idx].copy()]
        n_existing = 0
    else:
        centroids = [c.copy() for c in existing_centroids]
        n_existing = len(centroids)

    target_total = n_existing + n_to_pick

    while len(centroids) < target_total:
        C = np.asarray(centroids)
        d2 = np.sum(
            (X_unlabeled[:, None, :] - C[None, :, :]) ** 2,
            axis=2,
        )
        nearest_d2 = np.min(d2, axis=1)

        if existing_centroids is not None and len(existing_centroids) > 0:
            d2_existing = np.sum(
                (
                    X_unlabeled[:, None, :]
                    - existing_centroids[None, :, :]
                )
                ** 2,
                axis=2,
            )
            min_d2_existing = np.min(d2_existing, axis=1)
            min_allowed_dist = 0.3 * np.median(d2)
            nearest_d2 = nearest_d2.copy()
            nearest_d2[min_d2_existing < min_allowed_dist] = 1e-12

        if not np.all(np.isfinite(nearest_d2)):
            raise ValueError("Non-finite distance encountered during K-Means++")

        distance_sum = float(nearest_d2.sum())
        if distance_sum <= 1e-12:
            idx = rng.randint(0, n_samples)
        else:
            probabilities = nearest_d2 / distance_sum
            idx = rng.choice(n_samples, p=probabilities)
        centroids.append(X_unlabeled[idx].copy())

    return np.asarray(centroids[n_existing:target_total])


def seeded_semi_supervised_kmeans(
    X: np.ndarray,
    labels: np.ndarray,
    mask_lab: np.ndarray,
    seen_classes: Sequence[int],
    K: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Semi-supervised K-Means.

    Labelled points retain their ground-truth class assignments. Unlabelled
    points are repeatedly assigned to their nearest centroid. Seen-class
    centroids are initialized using labelled morphology features; remaining
    centroid slots are initialized from unlabelled data with K-Means++.
    """

    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    mask_lab = np.asarray(mask_lab, dtype=bool)
    seen_classes = np.asarray(seen_classes, dtype=int)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if len(X) != len(labels) or len(X) != len(mask_lab):
        raise ValueError("X, labels, and mask_lab must have the same length")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if len(np.unique(seen_classes)) != len(seen_classes):
        raise ValueError(f"seen_classes contains duplicates: {seen_classes.tolist()}")
    if np.any(seen_classes < 0) or np.any(seen_classes >= K):
        raise ValueError(
            f"seen class ids must be in [0, {K - 1}], "
            f"got {seen_classes.tolist()}"
        )
    if K < len(seen_classes):
        raise ValueError(f"K={K} is smaller than the number of seen classes")

    n, feature_dim = X.shape
    unlabeled_idx = np.where(~mask_lab)[0]
    seen_set = set(seen_classes.tolist())
    unseen_slots = np.asarray(
        [cluster_id for cluster_id in range(K) if cluster_id not in seen_set],
        dtype=int,
    )

    centroids = np.zeros((K, feature_dim), dtype=X.dtype)
    fixed_assign = np.full(n, -1, dtype=int)

    for class_id in seen_classes:
        class_indices = np.where((labels == class_id) & mask_lab)[0]
        if len(class_indices) == 0:
            raise ValueError(
                f"No labelled morphology features found for seen class {class_id}"
            )
        centroids[class_id] = np.mean(X[class_indices], axis=0)
        fixed_assign[class_indices] = class_id

    if len(unseen_slots) > 0:
        if len(unlabeled_idx) == 0:
            if len(seen_classes) == 0:
                raise ValueError("Cannot initialize centroids without any samples")
            for slot in unseen_slots:
                source_class = seen_classes[rng.randint(0, len(seen_classes))]
                centroids[slot] = centroids[source_class]
        else:
            existing_centroids = centroids[seen_classes]
            centroids[unseen_slots] = kmeanspp_pick_centroids(
                X[unlabeled_idx],
                existing_centroids,
                len(unseen_slots),
                random_state=random_state,
            )

    assignments = np.full(n, -1, dtype=int)

    for _iteration in range(max_iter):
        distances = pairwise_distances(X, centroids, metric="euclidean")
        new_assignments = np.argmin(distances, axis=1)
        new_assignments[fixed_assign >= 0] = fixed_assign[fixed_assign >= 0]

        new_centroids = np.zeros_like(centroids)
        for cluster_id in range(K):
            assigned_indices = np.where(new_assignments == cluster_id)[0]
            if len(assigned_indices) == 0:
                # Reinitialize an empty cluster from an unlabelled point where
                # possible, so labelled points are not chosen unnecessarily.
                source_pool = unlabeled_idx if len(unlabeled_idx) > 0 else np.arange(n)
                new_centroids[cluster_id] = X[
                    source_pool[rng.randint(0, len(source_pool))]
                ]
            else:
                new_centroids[cluster_id] = np.mean(
                    X[assigned_indices], axis=0
                )

        max_shift = np.linalg.norm(
            centroids - new_centroids, axis=1
        ).max()
        centroids = new_centroids
        assignments = new_assignments

        if max_shift <= tol:
            break

    return assignments, centroids


def collect_labelled_train_metadata(train_eval_loader: DataLoader):
    """Collect unique_id and labels only for labelled training examples."""

    labelled_ids: List = []
    labelled_labels: List[np.ndarray] = []

    for batch in train_eval_loader:
        if len(batch) != 4:
            raise ValueError(
                "train_dataset must return "
                "(image, class_label, unique_id, mask_lab)"
            )
        _images, class_labels, unique_ids, mask_lab = batch

        labels_np = _to_numpy_1d(class_labels, "class_labels").astype(int)
        ids_np = _to_numpy_1d(unique_ids, "unique_ids")

        if torch.is_tensor(mask_lab):
            mask_np = mask_lab.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask_lab)
        if mask_np.ndim == 0:
            mask_np = mask_np.reshape(1)
        elif mask_np.ndim > 1:
            # Match the original baseline's mask_lab = mask_lab[:, 0].
            mask_np = mask_np[:, 0]
        mask_np = mask_np.astype(bool).reshape(-1)

        if not (len(labels_np) == len(ids_np) == len(mask_np)):
            raise ValueError(
                "Training labels, unique_ids, and mask_lab have different lengths"
            )

        labelled_ids.extend(ids_np[mask_np].tolist())
        labelled_labels.append(labels_np[mask_np])

    if not labelled_labels or sum(len(x) for x in labelled_labels) == 0:
        raise ValueError("No labelled examples were found in train_eval_loader")

    return np.asarray(labelled_ids, dtype=object), np.concatenate(labelled_labels)


def collect_test_metadata(test_loader: DataLoader):
    """Collect test unique_id and ground-truth labels in DataLoader order."""

    all_ids: List = []
    all_labels: List[np.ndarray] = []

    for batch in test_loader:
        if len(batch) != 3:
            raise ValueError(
                "test_dataset must return (image, class_label, unique_id)"
            )
        _images, class_labels, unique_ids = batch

        labels_np = _to_numpy_1d(class_labels, "test class_labels").astype(int)
        ids_np = _to_numpy_1d(unique_ids, "test unique_ids")
        if len(labels_np) != len(ids_np):
            raise ValueError("Test labels and unique_ids have different lengths")

        all_ids.extend(ids_np.tolist())
        all_labels.append(labels_np)

    if not all_labels:
        raise ValueError("test_loader is empty")

    return np.asarray(all_ids, dtype=object), np.concatenate(all_labels)


def final_model_test(train_eval_loader: DataLoader, test_loader: DataLoader, args):
    """
    Read morphology features from NPZ, run semi-supervised K-Means, and log
    clustering accuracy/ARI/NMI using the project's existing evaluator.
    """

    morphology_X, id_to_row, feature_names = load_morphology_npz(
        args.morphology_npz,
        nan_policy=args.nan_policy,
    )

    train_ids, train_labels = collect_labelled_train_metadata(train_eval_loader)
    test_ids, test_labels = collect_test_metadata(test_loader)

    X_lab, train_keep, missing_train = lookup_embeddings(
        train_ids,
        morphology_X,
        id_to_row,
        args.missing_id_policy,
        split_name="labelled-training",
    )
    y_lab = train_labels[train_keep]

    X_test, test_keep, missing_test = lookup_embeddings(
        test_ids,
        morphology_X,
        id_to_row,
        args.missing_id_policy,
        split_name="test",
    )
    y_test = test_labels[test_keep]
    matched_test_ids = test_ids[test_keep]

    if missing_train:
        args.logger.warning(
            "Skipped {} labelled-training samples missing from the NPZ".format(len(missing_train))
        )
    if missing_test:
        args.logger.warning(
            "Skipped {} test samples missing from the NPZ".format(len(missing_test))
        )

    if X_lab.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Train/test morphology dimensions differ: {X_lab.shape[1]} vs "
            f"{X_test.shape[1]}"
        )

    args.logger.info(
        "Loaded morphology features: NPZ rows={}, dimensions={}, labelled train matched={}/{}, test matched={}/{}".format(
        morphology_X.shape[0],
        morphology_X.shape[1],
        len(X_lab),
        len(train_ids),
        len(X_test),
        len(test_ids)
    ))

    X_all = np.vstack([X_lab, X_test])
    n_lab = len(X_lab)
    n_test = len(X_test)

    masked_labels = np.concatenate(
        [y_lab, -np.ones(n_test, dtype=int)]
    )
    is_labelled = np.concatenate(
        [np.ones(n_lab, dtype=bool), np.zeros(n_test, dtype=bool)]
    )

    assignments, centroids = seeded_semi_supervised_kmeans(
        X=X_all,
        labels=masked_labels,
        mask_lab=is_labelled,
        seen_classes=args.train_labelled_classes,
        K=args.mlp_out_dim,
        max_iter=args.kmeans_max_iter,
        tol=args.kmeans_tol,
        random_state=args.seed,
    )
    test_predictions = assignments[n_lab:]

    mask_seen = np.isin(y_test, args.train_labelled_classes).astype(bool)
    mask_new = (~np.isin(y_test, args.train_classes)).astype(bool)

    results = log_accs_from_preds_v4(
        y_true=y_test,
        y_pred=test_predictions,
        mask_seen=mask_seen,
        mask_new=mask_new,
        T=0,
        eval_funcs=args.eval_funcs,
        save_name="Final Morphometrics Test ACC",
        print_output=False,
        args=args,
    )

    (
        all_acc,
        old_acc,
        self_acc,
        new_acc,
        all_ari,
        old_ari,
        self_ari,
        new_ari,
        all_nmi,
        old_nmi,
        self_nmi,
        new_nmi,
    ) = results

    args.logger.info('Final Test Acc: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_acc, old_acc, self_acc, new_acc))
    args.logger.info('Final Test ARI: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_ari, old_ari, self_ari, new_ari))
    args.logger.info('Final Test NMI: All {:.4f} | Old {:.4f} | Self {:.4f} | New {:.4f}'.format(all_nmi, old_nmi, self_nmi, new_nmi))

    if args.predictions_output:
        output_dir = os.path.dirname(args.predictions_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(
            args.predictions_output,
            ids=matched_test_ids,
            y_true=y_test,
            y_pred=test_predictions,
            centroids=centroids,
            feature_names=feature_names,
        )

    return results


# Alias using the naming style of the original AE script.
test_Final_model = final_model_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Morphometrics semi-supervised K-Means baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data/evaluation arguments retained for compatibility with the project.
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--eval_funcs",
        nargs="+",
        default=["v2"],
        help="Evaluation functions used by log_accs_from_preds_v4",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='urbanform', help='options: urbanform, allform')
    parser.add_argument("--prop_train_labels", type=float, default=0.8)
    parser.add_argument("--use_ssb_splits", action="store_true", default=False)
    parser.add_argument("--transform", type=str, default="imagenet")
    parser.add_argument("--exp_root", type=str, default=exp_root)
    parser.add_argument("--exp_name", default="morphometrics_kmeans", type=str)
    parser.add_argument("--exp_id", default="debug", type=str)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument(
        "--num_classes4test",
        type=int,
        default=6,
        help="Total number of clusters/classes K",
    )
    parser.add_argument(
        "--class_split_file",
        type=str,
        default="data/class_split_info_run00.json",
    )
    parser.add_argument(
        "--random_class_split",
        action="store_true",
        default=False,
        help="Randomly generate the labelled/unlabelled/novel class split",
    )

    parser.add_argument(
        "--morphology_npz",
        type=str,
        default="batch_morphology_62_features_standardized.npz",
        help="NPZ containing ids, X, and feature_names",
    )
    parser.add_argument(
        "--missing_id_policy",
        choices=["error", "skip"],
        default="error",
        help="How to handle dataset unique_ids absent from the NPZ",
    )
    parser.add_argument(
        "--nan_policy",
        choices=["error", "mean", "zero"],
        default="error",
        help="How to handle NaN/Inf values in morphology features",
    )
    parser.add_argument("--kmeans_max_iter", type=int, default=300)
    parser.add_argument("--kmeans_tol", type=float, default=1e-4)
    parser.add_argument(
        "--predictions_output",
        type=str,
        default=None,
        help="Optional .npz path for test IDs, labels, predictions, and centroids",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_random_seed(args.seed)

    # Retain the original project's dataset-specific split initialization.
    args = get_class_splits(args)
    init_experiment(args, runner_name=["Morphometrics"], exp_id=args.exp_id)

    # Some dataset helpers expect these attributes even though no image model is used.
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 512
    args.feat_dim = 0
    args.num_mlp_layers = 0

    args = get_class_split_info(args)
    args.mlp_out_dim = args.num_classes4test

    class_names_path = os.path.join(
        get_dir,
        f"dataset_class_name/{args.dataset_name}_name.npy",
    )
    if not os.path.exists(class_names_path):
        print(f"generate class_names_path for {args.dataset_name}")
        from dataset_class_name import gen_classnames

        gen_classnames.gen(args.dataset_name, class_names_path)

    class_names = np.load(class_names_path, allow_pickle=True)
    args.base_names = class_names[args.train_classes]

    print("train_classes:", class_names[args.train_classes])
    print("train_labelled_classes:", class_names[args.train_labelled_classes])
    print("train_unlabelled_classes:", class_names[args.train_unlabelled_classes])
    novel_class_ids = sorted(
        set(args.unlabeled_classes) - set(args.train_unlabelled_classes)
    )
    print("novel_classes:", class_names[novel_class_ids])

    # Images are not embeddings in this baseline. The transform only returns a
    # one-element placeholder so dataset loading remains compatible and cheap.
    metadata_transform = MetadataOnlyTransform()

    (
        train_dataset,
        test_dataset,
        _unlabelled_train_examples_test,
        _datasets,
        _train_examples_test,
        _full_dataset,
    ) = get_datasets(
        args.dataset_name,
        metadata_transform,
        metadata_transform,
        args,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    args.logger.info(
        "Accuracy of Semi-supervised K-Means++ using morphology features:"
    )
    final_model_test(train_eval_loader, test_loader, args)


if __name__ == "__main__":
    main()
