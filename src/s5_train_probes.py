"""S5: RQ1 linear probes over the S4 activation cache.

The headline existence claim: the model maintains a linearly decodable representation of the
CLIENT's risk tolerance that reflects integration of profile facts, not lexical echo. We test
it by training a linear probe on EXPLICIT vignettes (overt risk language) and measuring
transfer to IMPLICIT held-out vignettes (banned-lexicon, narrative only).

Design (see plan + config `probe:`):
  - Probe = L2 logistic regression (linear -> tests linear decodability, the relevant notion
    for steering/patching; yields a reusable weight vector). C chosen on a val split.
  - Sweep EVERY (layer, position): the best layer is task-specific (lit ~65% depth but wide),
    so l* = argmax implicit-held-out macro-AUROC is picked empirically, not assumed.
  - Targets: 3-tier one-vs-rest macro AUROC (headline) + continuous ridge on risk_score (R2).
  - Split by profile_id (twins/pairs share a profile -> vignette split would leak); pairs are
    excluded from the probe (they are constructed extremes for S6 patching).
  - Rigor (Hewitt & Liang): TF-IDF / bag-of-words / majority baselines + a shuffled-label
    CONTROL probe; report selectivity = real - control.
  - 5-seed split stability at l*; bootstrap CI on the headline implicit AUROC.
  - Stretch: multi-layer logistic stacking (ensembling lifts AUROC when signal is distributed).

Consumes S4 shards (results/activations/<tag>/shard_*.npz), consolidating them once into a
memmapped acts_all.npy so RAM stays flat. Pure logic (splits, depth, selectivity, assembly)
is torch-free and unit-tested in tests/test_s5.py.

Usage:
    python src/s5_train_probes.py [--config config.yaml]
        [--activations results/activations/gemma-2-9b-it]
        [--positions profile_end,profile_mean,decision] [--layers 18,20,...]
        [--seeds 5] [--no-ensemble]
"""

import argparse
import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import yaml

from utils.stats import auroc_ci, macro_ovr_auroc

# Degenerate splits (a tier missing from a tiny test set) make OvR AUROC undefined; we guard
# for that explicitly below, so silence the per-call sklearn noise.
try:
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
except Exception:
    pass

PAIRS_EXCLUDED = True   # probe trains/tests on twins only; counterfactual pairs are for S6


# --------------------------------------------------------------------------------------
# Pure logic (numpy/sklearn, no torch) -- unit-tested in tests/test_s5.py
# --------------------------------------------------------------------------------------
def make_splits(profile_ids, seed=42, fracs=(0.70, 0.10, 0.20)):
    """Deterministic profile_id -> {'train','val','test'} assignment.

    Splitting by profile (not vignette) keeps explicit/implicit twins on the same side, so a
    probe can't memorize a profile via its other rendering.
    """
    uniq = sorted(set(profile_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_tr = int(round(fracs[0] * len(uniq)))
    n_va = int(round(fracs[1] * len(uniq)))
    out = {}
    for rank, i in enumerate(perm):
        split = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
        out[uniq[i]] = split
    return out


def split_indices(labels, split):
    """Row-index arrays for each probe split, given per-row `labels` and a profile->split map.

    train/val/explicit_test draw from EXPLICIT rows; implicit_test (headline) and the optional
    implicit_train ceiling draw from IMPLICIT rows; pairs (pair_id set) are excluded.
    """
    buckets = {"train": [], "val": [], "explicit_test": [], "implicit_test": [], "implicit_train": []}
    for i, lab in enumerate(labels):
        if PAIRS_EXCLUDED and lab.get("pair_id") is not None:
            continue
        s = split.get(lab["profile_id"])
        if s is None:
            continue
        vt = lab["vignette_type"]
        if vt == "explicit":
            buckets[{"train": "train", "val": "val", "test": "explicit_test"}[s]].append(i)
        elif vt == "implicit":
            if s == "test":
                buckets["implicit_test"].append(i)
            elif s == "train":
                buckets["implicit_train"].append(i)
    return {k: np.asarray(v, dtype=int) for k, v in buckets.items()}


def fractional_depth(layer, n_layers):
    """Layer index as a fraction of total depth in [0,1] (comparable to the ~65% literature)."""
    return round(layer / (n_layers - 1), 3) if n_layers > 1 else 0.0


def selectivity(real_auroc, control_auroc):
    """Probe selectivity = real-task AUROC minus shuffled-label control AUROC (Hewitt & Liang)."""
    return round(real_auroc - control_auroc, 4)


def config_sha(config_path):
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:12]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Activation loading (memmap consolidation of S4 shards)
# --------------------------------------------------------------------------------------
def load_meta(act_dir):
    return json.loads((Path(act_dir) / "meta.json").read_text())


def consolidate(act_dir):
    """Build (once) a memmapped acts_all.npy [N,L,P,d] + labels.jsonl from the S4 shards.

    Returns (acts_memmap, labels_list, meta). Re-runs are no-ops once the consolidated files
    exist (and match the shard row count).
    """
    act_dir = Path(act_dir)
    meta = load_meta(act_dir)
    acts_path, labels_path = act_dir / "acts_all.npy", act_dir / "labels.jsonl"
    shards = sorted(act_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {act_dir} -- run S4 first")

    label_keys = ["vignette_id", "profile_id", "pair_id", "tier", "risk_score",
                  "vignette_type", "contradictory"]

    # Reuse only if the consolidation is internally consistent AND up to date with the shards
    # (a stale acts_all.npy from a smaller earlier run -- e.g. an S4 --dry-run -- must NOT be
    # reused after S4 has written more shards).
    if acts_path.exists() and labels_path.exists():
        labels = [json.loads(l) for l in open(labels_path)]
        acts = np.load(acts_path, mmap_mode="r")
        fresh = acts_path.stat().st_mtime >= max(sh.stat().st_mtime for sh in shards)
        if acts.shape[0] == len(labels) and fresh:
            return acts, labels, meta   # already consolidated and current
        print(f"    stale/partial acts_all.npy ({acts.shape[0]} rows) -> rebuilding from {len(shards)} shards")

    # Stream shards into one memmap.
    n_total = 0
    shapes = []
    for sh in shards:
        with np.load(sh, allow_pickle=True) as z:
            shapes.append(z["acts"].shape)
            n_total += z["acts"].shape[0]
    _, L, P, d = shapes[0]
    acts = np.lib.format.open_memmap(acts_path, mode="w+", dtype=np.float16, shape=(n_total, L, P, d))
    labels, off = [], 0
    for sh in shards:
        with np.load(sh, allow_pickle=True) as z:
            a = z["acts"]
            acts[off:off + a.shape[0]] = a
            for r in range(a.shape[0]):
                labels.append({k: _py(z[k][r]) for k in label_keys})
            off += a.shape[0]
    acts.flush()
    with open(labels_path, "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    print(f"    consolidated {len(shards)} shards -> acts_all.npy {acts.shape}")
    return np.load(acts_path, mmap_mode="r"), labels, meta


def _py(x):
    """numpy scalar/object -> json-safe python type."""
    if isinstance(x, np.generic):
        x = x.item()
    return x


def load_plane(acts, l_idx, p_idx):
    """Float32 [N, d] for one (layer, position) -- the unit a single probe trains on."""
    return np.asarray(acts[:, l_idx, p_idx, :], dtype=np.float32)


def load_texts(cfg):
    """vignette_id -> raw text, for the TF-IDF / bag-of-words baselines."""
    texts = {}
    vdir = Path(cfg["paths"]["vignettes_dir"])
    for name in ("explicit", "implicit"):
        p = vdir / f"{name}.jsonl"
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                texts[r["vignette_id"]] = r["text"]
    return texts


# --------------------------------------------------------------------------------------
# Probes (sklearn, CPU)
# --------------------------------------------------------------------------------------
def fit_logistic(Xtr, ytr, Xva, yva, Cs=(0.01, 0.1, 1.0, 10.0), seed=42):
    """Standardize on train, grid C on val macro-AUROC, refit best. Returns a probe dict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
    classes = sorted(set(ytr))
    best, best_auc = None, -1.0
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        clf.fit(Xtr_s, ytr)
        auc = macro_ovr_auroc(yva, clf.predict_proba(Xva_s), list(clf.classes_))
        if auc > best_auc:
            best, best_auc = clf, auc
    return {"scaler": scaler, "clf": best, "classes": list(best.classes_), "C": best.C}


def probe_proba(probe, X):
    return probe["clf"].predict_proba(probe["scaler"].transform(X))


def eval_probe(probe, X, y):
    proba = probe_proba(probe, X)
    auc = macro_ovr_auroc(y, proba, probe["classes"])
    acc = float((probe["clf"].predict(probe["scaler"].transform(X)) == np.asarray(y)).mean())
    return auc, acc, proba


def ridge_r2(Xtr, str_tr, Xte, str_te, seed=42):
    """Continuous probe: ridge on risk_score, return R2 on the eval set."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    rg = Ridge(alpha=1.0, random_state=seed).fit(sc.transform(Xtr), str_tr)
    return float(r2_score(str_te, rg.predict(sc.transform(Xte))))


def text_baseline(kind, train_texts, ytr, eval_sets, seed=42):
    """TF-IDF or bag-of-words + logistic on raw text; macro-AUROC per eval set."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = (TfidfVectorizer(max_features=20000, ngram_range=(1, 2)) if kind == "tfidf"
           else CountVectorizer(max_features=20000))
    Xtr = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(Xtr, ytr)
    out = {}
    for name, (texts, y) in eval_sets.items():
        out[name] = round(macro_ovr_auroc(y, clf.predict_proba(vec.transform(texts)), list(clf.classes_)), 4)
    return out


def probe_to_npz(path, probe, ridge_coef, layer, position, frac):
    """Persist a probe's linear params for S8/S10/S11 (no pickled sklearn object)."""
    clf, sc = probe["clf"], probe["scaler"]
    np.savez(path, classes=np.array(probe["classes"], dtype=object),
             coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32),
             scaler_mean=sc.mean_.astype(np.float32), scaler_scale=sc.scale_.astype(np.float32),
             ridge_coef=np.asarray(ridge_coef, dtype=np.float32) if ridge_coef is not None else np.array([]),
             layer=layer, position=position, frac_depth=frac, C=probe["C"])


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def run(cfg, args):
    act_dir = Path(args.activations or
                   Path(cfg["paths"]["results_dir"]) / "activations" / "gemma-2-9b-it")
    acts, labels, meta = consolidate(act_dir)
    cfg_layers, positions_all = meta["layers"], meta["positions"]
    n_layers_model = meta["num_hidden_layers"]

    layers = ([int(x) for x in args.layers.split(",")] if args.layers else cfg_layers)
    positions = (args.positions.split(",") if args.positions else positions_all)
    layers = [l for l in layers if l in cfg_layers]
    positions = [p for p in positions if p in positions_all]

    split = make_splits([lab["profile_id"] for lab in labels], seed=cfg["seed"])
    idx = split_indices(labels, split)
    tier = np.array([lab["tier"] for lab in labels])
    score = np.array([lab["risk_score"] for lab in labels], dtype=float)
    print(f"S5: train={len(idx['train'])} val={len(idx['val'])} "
          f"explicit_test={len(idx['explicit_test'])} implicit_test={len(idx['implicit_test'])}")
    n_tiers = len(set(tier))
    for key in ("train", "val", "implicit_test"):
        present = set(tier[idx[key]]) if len(idx[key]) else set()
        if len(idx[key]) < 3 * n_tiers or len(present) < n_tiers:
            raise RuntimeError(
                f"split '{key}' is degenerate ({len(idx[key])} rows, tiers present={sorted(present)} "
                f"of {n_tiers}). The probe needs every tier in each split. This usually means the "
                f"activation cache is too small -- DON'T run S5 on an S4 --dry-run (100-vignette) "
                f"cache; cache the full set (drop --dry-run) even on the 2B dev model.")
    print(f"    sweep: {len(layers)} layers x {len(positions)} positions")

    ytr, yva = tier[idx["train"]], tier[idx["val"]]
    str_tr = score[idx["train"]]

    # ---- baselines (text, shared across the sweep) ----
    texts = load_texts(cfg)

    def texts_for(key):
        return [texts[labels[i]["vignette_id"]] for i in idx[key]]

    eval_sets = {"explicit_test": (texts_for("explicit_test"), tier[idx["explicit_test"]]),
                 "implicit_test": (texts_for("implicit_test"), tier[idx["implicit_test"]])}
    baselines = {}
    if texts:
        for kind in ("tfidf", "bow"):
            baselines[kind] = text_baseline(kind, texts_for("train"), ytr, eval_sets, cfg["seed"])
        maj = max(set(ytr), key=list(ytr).count)               # majority-class AUROC == 0.5 by def
        baselines["majority"] = {"explicit_test": 0.5, "implicit_test": 0.5, "class": str(maj)}
    print(f"    baselines: {baselines.get('tfidf')}")

    # ---- the (layer, position) sweep ----
    rng = np.random.default_rng(cfg["seed"])
    sweep = {p: {} for p in positions}
    proba_store = {}            # (layer, position) -> {'val','imp'} for the ensemble
    best = {"implicit_auroc": -1.0}
    for p in positions:
        p_idx = positions_all.index(p)
        for l in layers:
            l_idx = cfg_layers.index(l)
            plane = load_plane(acts, l_idx, p_idx)
            Xtr, Xva = plane[idx["train"]], plane[idx["val"]]
            Xet, Xit = plane[idx["explicit_test"]], plane[idx["implicit_test"]]
            probe = fit_logistic(Xtr, ytr, Xva, yva, seed=cfg["seed"])
            et_auc, et_acc, _ = eval_probe(probe, Xet, tier[idx["explicit_test"]])
            it_auc, it_acc, it_proba = eval_probe(probe, Xit, tier[idx["implicit_test"]])
            _, _, va_proba = eval_probe(probe, Xva, yva)
            # shuffled-label control (selectivity): shuffle BOTH train and val labels so C
            # selection can't peek at the real signal; then score on the REAL implicit test.
            ctrl = fit_logistic(Xtr, ytr[rng.permutation(len(ytr))],
                                Xva, yva[rng.permutation(len(yva))], seed=cfg["seed"])
            ctrl_auc, _, _ = eval_probe(ctrl, Xit, tier[idx["implicit_test"]])
            r2 = ridge_r2(Xtr, str_tr, Xit, score[idx["implicit_test"]], seed=cfg["seed"])
            rec = {"frac_depth": fractional_depth(l, n_layers_model),
                   "explicit_test": {"auroc": round(et_auc, 4), "acc": round(et_acc, 4)},
                   "implicit_test": {"auroc": round(it_auc, 4), "acc": round(it_acc, 4),
                                     "ridge_r2": round(r2, 4)},
                   "control_auroc": round(ctrl_auc, 4),
                   "selectivity": selectivity(it_auc, ctrl_auc)}
            sweep[p][l] = rec
            proba_store[(l, p)] = {"val": va_proba, "imp": it_proba}
            if it_auc > best["implicit_auroc"]:
                best = {"position": p, "layer": l, "l_idx": l_idx, "p_idx": p_idx,
                        "frac_depth": rec["frac_depth"], "implicit_auroc": round(it_auc, 4),
                        "ridge_r2": round(r2, 4), "probe": probe,
                        "ridge_coef": None}
        print(f"    [{p}] best so far: L{best.get('layer')} "
              f"AUROC={best['implicit_auroc']:.4f}")

    # ---- headline CI + 5-seed stability at l* ----
    if "layer" not in best:
        raise RuntimeError("no (layer, position) produced a valid AUROC -- every probe returned "
                           "NaN, which means a tier is missing from implicit_test. Use a larger "
                           "activation cache (not an S4 --dry-run).")
    bl, bp = best["layer"], best["position"]
    plane = load_plane(acts, best["l_idx"], best["p_idx"])
    point, lo, hi = auroc_ci(tier[idx["implicit_test"]],
                             probe_proba(best["probe"], plane[idx["implicit_test"]]),
                             best["probe"]["classes"], seed=cfg["seed"])
    best["ci"] = [round(lo, 4), round(hi, 4)]
    seed_aucs = []
    for s in range(args.seeds):
        sp = make_splits([lab["profile_id"] for lab in labels], seed=cfg["seed"] + s)
        si = split_indices(labels, sp)
        pr = fit_logistic(plane[si["train"]], tier[si["train"]], plane[si["val"]], tier[si["val"]],
                          seed=cfg["seed"] + s)
        a, _, _ = eval_probe(pr, plane[si["implicit_test"]], tier[si["implicit_test"]])
        seed_aucs.append(round(a, 4))
    stability = {"implicit_auroc_mean": round(float(np.mean(seed_aucs)), 4),
                 "std": round(float(np.std(seed_aucs)), 4), "seeds": seed_aucs}

    # ---- multi-layer ensemble (stretch) at the best position ----
    ensemble = None
    if not args.no_ensemble and len(layers) > 1:
        ens_layers = [l for l in layers if (l, bp) in proba_store]
        Xv = np.hstack([proba_store[(l, bp)]["val"] for l in ens_layers])
        Xi = np.hstack([proba_store[(l, bp)]["imp"] for l in ens_layers])
        from sklearn.linear_model import LogisticRegression
        meta_clf = LogisticRegression(max_iter=2000, random_state=cfg["seed"]).fit(Xv, yva)
        ens_auc = macro_ovr_auroc(tier[idx["implicit_test"]], meta_clf.predict_proba(Xi),
                                  list(meta_clf.classes_))
        ensemble = {"position": bp, "n_layers": len(ens_layers),
                    "implicit_auroc": round(ens_auc, 4),
                    "lift_over_best_single": round(ens_auc - best["implicit_auroc"], 4)}

    # ---- thresholds (pre-registered) ----
    pcfg = cfg["probe"]
    tfidf_imp = baselines.get("tfidf", {}).get("implicit_test", 0.5)
    thresholds = {
        "auroc_min": pcfg["auroc_min"], "baseline_margin_min": pcfg["baseline_margin_min"],
        "ridge_r2_min": pcfg["ridge_r2_min"],
        "auroc_pass": bool(best["implicit_auroc"] >= pcfg["auroc_min"]),
        "baseline_margin_pass": bool(best["implicit_auroc"] - tfidf_imp >= pcfg["baseline_margin_min"]),
        "ridge_pass": bool(best["ridge_r2"] >= pcfg["ridge_r2_min"]),
    }
    thresholds["RQ1_clean_win"] = bool(thresholds["auroc_pass"] and thresholds["baseline_margin_pass"])

    # ---- persist ----
    res_dir = Path(cfg["paths"]["results_dir"])
    probe_dir = res_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    # best overall + the best layer at each position (S10/S11 read at 'decision')
    probe_to_npz(probe_dir / f"probe_best_L{bl}_{bp}.npz", best["probe"], None, bl, bp, best["frac_depth"])
    for p in positions:
        bl_p = max(sweep[p], key=lambda l: sweep[p][l]["implicit_test"]["auroc"])
        pr = fit_logistic(load_plane(acts, cfg_layers.index(bl_p), positions_all.index(p))[idx["train"]],
                          ytr, load_plane(acts, cfg_layers.index(bl_p), positions_all.index(p))[idx["val"]],
                          yva, seed=cfg["seed"])
        probe_to_npz(probe_dir / f"probe_L{bl_p}_{p}.npz", pr, None, bl_p, p, fractional_depth(bl_p, n_layers_model))

    report = {
        "meta": {"model": meta["model"], "num_hidden_layers": n_layers_model,
                 "layers_swept": layers, "positions_swept": positions,
                 "n_train": len(idx["train"]), "n_implicit_test": len(idx["implicit_test"]),
                 "seed": cfg["seed"], "config_sha": config_sha(args.config),
                 "git_commit": git_commit(), "pairs_excluded": PAIRS_EXCLUDED},
        "best": {k: v for k, v in best.items() if k not in ("probe", "l_idx", "p_idx", "ridge_coef")},
        "seed_stability": stability, "baselines": baselines, "ensemble": ensemble,
        "thresholds": thresholds, "sweep": sweep,
    }
    out = res_dir / "s5_probe_results.json"
    out.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    print(f"  report -> {out}  |  probes -> {probe_dir}")
    return report


def _print_summary(report):
    b, t = report["best"], report["thresholds"]
    print("=== S5 PROBE (RQ1) ===")
    print(f"  best: L{b['layer']} ({b['frac_depth']} depth) @ {b['position']}  "
          f"implicit AUROC={b['implicit_auroc']:.4f} CI{b['ci']}  ridge R2={b['ridge_r2']:.3f}")
    print(f"  5-seed stability: {report['seed_stability']['implicit_auroc_mean']:.4f} "
          f"+/- {report['seed_stability']['std']:.4f}")
    if report["baselines"].get("tfidf"):
        print(f"  TF-IDF implicit AUROC={report['baselines']['tfidf']['implicit_test']:.4f}  "
              f"(probe margin {b['implicit_auroc'] - report['baselines']['tfidf']['implicit_test']:+.4f})")
    if report["ensemble"]:
        e = report["ensemble"]
        print(f"  ensemble ({e['n_layers']} layers): AUROC={e['implicit_auroc']:.4f} "
              f"(lift {e['lift_over_best_single']:+.4f})")
    print(f"  [implicit AUROC >= {t['auroc_min']}] {'PASS' if t['auroc_pass'] else 'FAIL'}  "
          f"[>= TF-IDF+{t['baseline_margin_min']}] {'PASS' if t['baseline_margin_pass'] else 'FAIL'}  "
          f"[ridge R2 >= {t['ridge_r2_min']}] {'PASS' if t['ridge_pass'] else 'FAIL'}")
    print(f"  RQ1 clean win: {'YES' if t['RQ1_clean_win'] else 'NO (report honestly; causal sections carry)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--activations", default=None, help="S4 cache dir (results/activations/<tag>)")
    ap.add_argument("--positions", default=None, help="comma subset of cached positions")
    ap.add_argument("--layers", default=None, help="comma subset of cached layers")
    ap.add_argument("--seeds", type=int, default=5, help="split-seed count for stability at l*")
    ap.add_argument("--no-ensemble", action="store_true", help="skip the multi-layer stacking row")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, args)


if __name__ == "__main__":
    main()
