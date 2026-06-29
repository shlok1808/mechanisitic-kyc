"""Shared statistics helpers: bootstrap CIs and macro one-vs-rest AUROC.

Used by S3 (gate, Spearman CI) and S5 (probe AUROC CI). Kept dependency-light (numpy +
scipy + sklearn, all CPU) so the probe-evaluation logic is unit-testable without a GPU.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def spearman_ci(x, y, n_boot=2000, seed=42):
    """Spearman rho with a bootstrap 95% percentile CI (resampling paired observations)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    boot = []
    idx = np.arange(len(x))
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if np.std(x[s]) > 0 and np.std(y[s]) > 0:
            boot.append(spearmanr(x[s], y[s]).correlation)
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))
    return float(rho), float(lo), float(hi)


def macro_ovr_auroc(y_true, y_score, labels):
    """Macro-averaged one-vs-rest AUROC.

    `y_true`: int/str class labels, shape [n]. `y_score`: class probabilities, shape
    [n, n_classes] aligned to `labels` order. Falls back to the binary AUROC when there
    are two classes (sklearn wants the positive-class column there).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, float)
    labels = list(labels)
    if len(labels) == 2:
        pos = labels[1]
        return float(roc_auc_score((y_true == pos).astype(int), y_score[:, 1]))
    return float(roc_auc_score(y_true, y_score, multi_class="ovr",
                               average="macro", labels=labels))


def auroc_ci(y_true, y_score, labels, n_boot=2000, seed=42):
    """Macro-OvR AUROC with a bootstrap 95% CI over the evaluation rows.

    Bootstrap resamples that happen to miss a class (so OvR is undefined) are skipped.
    Returns (point_estimate, lo, hi).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, float)
    labels = list(labels)
    point = macro_ovr_auroc(y_true, y_score, labels)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    boot = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if set(np.unique(y_true[s])) != set(labels):     # need every class present for OvR
            continue
        try:
            boot.append(macro_ovr_auroc(y_true[s], y_score[s], labels))
        except ValueError:
            continue
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))
    return float(point), float(lo), float(hi)
