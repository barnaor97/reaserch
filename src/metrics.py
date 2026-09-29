"""
Evaluation helpers and metric computation for the restoration prioritization
project. Used by Notebooks 05, 06, and 07 to produce comparable result
artefacts across the main model and all robustness experiments.

v3 additions:
    * per_lc_stats — mean predicted scores grouped by land-cover class,
      complementing per_category_stats which groups by recurrence category.
"""

import json
import os
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix, precision_recall_curve)
from torch.utils.data import (BatchSampler, DataLoader, RandomSampler,
                              SequentialSampler, TensorDataset)


# ============================================================
# Loaders
# ============================================================
class _PixelSubset(torch.utils.data.Dataset):
    """A view onto a pixel subset of the full arrays.

    Nothing is copied at construction. Each fetch gathers one batch directly
    out of the parent arrays, so a split costs one batch of memory rather
    than a full slice of X.
    """

    def __init__(self, X, y, mask, targets, indices):
        self.X = X
        self.y = y
        self.mask = mask
        self.severity = targets["severity"]
        self.persistence = targets["persistence"]
        self.recovery_gap = targets["recovery_gap"]
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, batch):
        # The BatchSampler below hands us a whole list of positions at once,
        # so this is a single vectorised gather per batch, not per sample.
        rows = self.indices[np.asarray(batch)]
        return (
            torch.from_numpy(np.ascontiguousarray(self.X[rows])),
            torch.from_numpy(np.ascontiguousarray(self.y[rows])),
            torch.from_numpy(np.ascontiguousarray(self.mask[rows])),
            torch.from_numpy(np.ascontiguousarray(self.severity[rows])),
            torch.from_numpy(np.ascontiguousarray(self.persistence[rows])),
            torch.from_numpy(np.ascontiguousarray(self.recovery_gap[rows])),
        )


def make_loader(X, y, mask, targets, indices, batch_size, shuffle):
    """Build a DataLoader for a subset of pixels.

    Previously this did torch.tensor(X[indices]), which materialised the
    subset twice: once for the fancy index and once for the tensor copy. On
    the full dataset that was ~19 GB on top of X for the training split
    alone, which pushed the machine into swap. Batches are now gathered on
    demand. Batch composition is unchanged: RandomSampler draws from the same
    torch RNG that shuffle=True used.
    """
    ds = _PixelSubset(X, y, mask, targets, indices)
    inner = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    sampler = BatchSampler(inner, batch_size=batch_size, drop_last=False)
    # batch_size=None: the sampler yields whole batches, so DataLoader passes
    # the index list straight to __getitem__ and skips the default collate.
    return DataLoader(ds, sampler=sampler, batch_size=None)


# ============================================================
# Single epoch
# ============================================================
def run_epoch(model, criterion, optimizer, loader, device, train, grad_clip=1.0,
              compute_auc=True):
    """Run one epoch on the supplied loader.

    compute_auc=False skips the per-epoch AUC and the probability/label
    accumulation it needs. On a full training fold that block costs ~13.6 s
    per epoch -- more than the training step itself -- and the training AUC is
    only ever written to history.json: model selection and every plot use the
    validation AUC. Leave it True for validation passes.
    """
    model.train(train)
    sums = {"total": 0, "burned": 0, "auxiliary": 0, "n": 0}
    probs_all, labels_all, mask_all = [], [], []

    for batch in loader:
        X_b, y_b, m_b, sev_b, per_b, rec_b = (b.to(device) for b in batch)
        tgt = {"burned": y_b, "severity": sev_b,
               "persistence": per_b, "recovery_gap": rec_b}

        if train:
            optimizer.zero_grad()
        outputs = model(X_b, mask=m_b)
        losses  = criterion(outputs, tgt, m_b)

        if train:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = X_b.size(0)
        for k in ["total", "burned", "auxiliary"]:
            sums[k] += losses[k].item() * bs
        sums["n"] += bs

        if compute_auc:
            probs_all.append(
                torch.sigmoid(outputs["burned_logits"]).detach().cpu().numpy())
            labels_all.append(y_b.cpu().numpy())
            mask_all.append(m_b.cpu().numpy())

    metrics = {k: sums[k] / sums["n"]
               for k in ["total", "burned", "auxiliary"]}
    if not compute_auc:
        metrics["auc"] = float("nan")
        return metrics
    probs   = np.concatenate(probs_all).reshape(-1)
    labels  = np.concatenate(labels_all).reshape(-1)
    msk     = np.concatenate(mask_all).reshape(-1).astype(bool)
    try:
        metrics["auc"] = roc_auc_score(labels[msk], probs[msk])
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


# ============================================================
# Threshold tuning
# ============================================================
def tune_threshold(model, loader, device, n_thresholds=200):
    """Sweep thresholds on the validation set and return the F1-optimal one."""
    model.eval()
    probs_all, labels_all, mask_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            X_b, y_b, m_b, *_ = (b.to(device) for b in batch)
            out = model(X_b, mask=m_b)
            probs_all.append(torch.sigmoid(out["burned_logits"]).cpu().numpy())
            labels_all.append(y_b.cpu().numpy())
            mask_all.append(m_b.cpu().numpy())

    probs  = np.concatenate(probs_all).reshape(-1)
    labels = np.concatenate(labels_all).reshape(-1)
    msk    = np.concatenate(mask_all).reshape(-1).astype(bool)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    f1s = [f1_score(labels[msk], (probs[msk] >= t).astype(int),
                    zero_division=0) for t in thresholds]
    best_thr = float(thresholds[int(np.argmax(f1s))])
    best_f1  = float(max(f1s))
    return best_thr, best_f1, probs, labels, msk


# ============================================================
# Test evaluation
# ============================================================
def evaluate_test_set(model, loader, device, threshold, targets, idx_test):
    """Run model on test set and compute the full metric panel."""
    model.eval()
    test_probs, test_labels, test_mask = [], [], []
    test_sev, test_per, test_rec = [], [], []

    with torch.no_grad():
        for batch in loader:
            X_b, y_b, m_b, *_ = (b.to(device) for b in batch)
            out = model(X_b, mask=m_b)
            test_probs.append(torch.sigmoid(out["burned_logits"]).cpu().numpy())
            test_labels.append(y_b.cpu().numpy())
            test_mask.append(m_b.cpu().numpy())
            test_sev.append(out["severity"].cpu().numpy())
            test_per.append(out["persistence"].cpu().numpy())
            test_rec.append(out["recovery_gap"].cpu().numpy())

    test_probs  = np.concatenate(test_probs).reshape(-1)
    test_labels = np.concatenate(test_labels).reshape(-1)
    test_mask   = np.concatenate(test_mask).reshape(-1).astype(bool)
    test_sev    = np.concatenate(test_sev)
    test_per    = np.concatenate(test_per)
    test_rec    = np.concatenate(test_rec)

    preds = (test_probs[test_mask] >= threshold).astype(int)
    labels_thr = test_labels[test_mask]

    burned_metrics = {
        "auc":       float(roc_auc_score(labels_thr, test_probs[test_mask])),
        "threshold": float(threshold),
        "precision": float(precision_score(labels_thr, preds, zero_division=0)),
        "recall":    float(recall_score(labels_thr, preds, zero_division=0)),
        "f1":        float(f1_score(labels_thr, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels_thr, preds).tolist(),
    }

    # Correlations: predictions vs analytical targets on test set
    correlations = {}
    for dim_name, predictions in [
        ("severity",     test_sev),
        ("persistence",  test_per),
        ("recovery_gap", test_rec),
    ]:
        true_vals = targets[dim_name][idx_test]
        r_p, p_p = pearsonr(predictions, true_vals)
        r_s, p_s = spearmanr(predictions, true_vals)
        correlations[dim_name] = {
            "pearson":         float(r_p),
            "pearson_pvalue":  float(p_p),
            "spearman":        float(r_s),
            "spearman_pvalue": float(p_s),
        }

    predictions = {
        "burned_probs":  test_probs,
        "burned_labels": test_labels,
        "mask":          test_mask,
        "severity":      test_sev,
        "persistence":   test_per,
        "recovery_gap":  test_rec,
    }

    return burned_metrics, correlations, predictions


# ============================================================
# Per-category breakdown by recurrence
# ============================================================
def per_category_stats(predictions, idx_test, meta_df, pixel_types):
    """Mean predicted scores grouped by pixel-type category."""
    pixel_types_test = meta_df.iloc[idx_test]["pixel_type"].values
    stats = {}
    for ptype in pixel_types:
        sel = pixel_types_test == ptype
        if sel.any():
            stats[ptype] = {
                "n": int(sel.sum()),
                "severity":     float(predictions["severity"][sel].mean()),
                "persistence":  float(predictions["persistence"][sel].mean()),
                "recovery_gap": float(predictions["recovery_gap"][sel].mean()),
            }
    return stats


# ============================================================
# Per-land-cover breakdown
# ============================================================
def per_lc_stats(predictions, idx_test, lc_labels, lc_categories):
    """Mean predicted scores grouped by dominant land-cover class.

    Complements per_category_stats: instead of stratifying by fire
    recurrence, it stratifies by land-cover class. Useful to detect
    systematic biases such as elevated predictions on cropland caused
    by harvest cycles.

    Parameters
    ----------
    predictions : dict
        Output of evaluate_test_set: contains severity, persistence,
        recovery_gap, and burned_probs arrays aligned with idx_test.
    idx_test : np.ndarray
        Indices selecting test-set rows from the full pixel array.
    lc_labels : np.ndarray, shape (N,)
        Dominant LC class code for each pixel in the full array.
    lc_categories : dict
        Mapping from LC integer code to human-readable name
        (typically config.LC_CATEGORIES).
    """
    lc_test = lc_labels[idx_test]
    stats = {}
    for code, name in lc_categories.items():
        sel = lc_test == code
        n = int(sel.sum())
        if n == 0:
            continue
        stats[name] = {
            "code":         int(code),
            "n":            n,
            "burned_prob_mean": float(predictions["burned_probs"].reshape(
                predictions["severity"].shape[0], -1)[sel].mean()),
            "severity":     float(predictions["severity"][sel].mean()),
            "persistence":  float(predictions["persistence"][sel].mean()),
            "recovery_gap": float(predictions["recovery_gap"][sel].mean()),
        }
    return stats


# ============================================================
# Persistence
# ============================================================
def save_experiment_results(exp_dir, exp_name, model_state, history,
                            burned_metrics, correlations, per_category,
                            extra=None):
    """Write a complete experiment record to disk."""
    os.makedirs(exp_dir, exist_ok=True)

    if model_state is not None:
        torch.save(model_state, os.path.join(exp_dir, "model.pt"))

    if history is not None:
        with open(os.path.join(exp_dir, "history.json"), "w") as f:
            json.dump(history, f)

    results = {
        "experiment_name":  exp_name,
        "burned_detection": burned_metrics,
        "correlations":     correlations,
        "per_category":     per_category,
    }
    if extra:
        results.update(extra)

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


# ============================================================
# Pretty print
# ============================================================
def print_summary(exp_name, n_params, burned_metrics, correlations,
                  per_category, pixel_types):
    """Print a clean summary block for an experiment."""
    print("=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 70)
    print(f"\nModel parameters: {n_params:,}")
    print(f"\nBurned detection:")
    print(f"  AUC:        {burned_metrics['auc']:.4f}")
    print(f"  Threshold:  {burned_metrics['threshold']:.3f}")
    print(f"  Precision:  {burned_metrics['precision']:.3f}")
    print(f"  Recall:     {burned_metrics['recall']:.3f}")
    print(f"  F1:         {burned_metrics['f1']:.3f}")

    print(f"\nDimension correlations:")
    print(f"  {'dimension':15s} {'Pearson':>8s} {'Spearman':>9s}")
    for dim, corr in correlations.items():
        print(f"  {dim:15s} {corr['pearson']:>8.3f} {corr['spearman']:>9.3f}")

    print(f"\nPer-category mean predictions:")
    print(f"  {'category':20s} {'n':>4s} {'severity':>10s} "
          f"{'persistence':>11s} {'recovery_gap':>12s}")
    for ptype in pixel_types:
        if ptype in per_category:
            v = per_category[ptype]
            print(f"  {ptype:20s} {v['n']:>4d} "
                  f"{v['severity']:>10.3f} "
                  f"{v['persistence']:>11.3f} "
                  f"{v['recovery_gap']:>12.3f}")
    print()


def print_lc_summary(lc_stats, lc_categories):
    """Print an LC-stratified summary block for an experiment."""
    print(f"\nPer-land-cover mean predictions:")
    print(f"  {'category':22s} {'n':>5s} {'sev':>8s} {'per':>8s} {'rec':>8s}")
    for name, code in [(v, k) for k, v in lc_categories.items()]:
        if name in lc_stats:
            v = lc_stats[name]
            print(f"  {name:22s} {v['n']:>5d} "
                  f"{v['severity']:>8.3f} "
                  f"{v['persistence']:>8.3f} "
                  f"{v['recovery_gap']:>8.3f}")
    print()
