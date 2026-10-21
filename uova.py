#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWARE - "Which task is the controller working on?" (Engage #2 hackathon)
Pipeline di learning-to-rank per massimizzare il top-3 hit rate.
"""
import argparse
import os
import re
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# --------------------------------------------------------------------------------------
# Costanti
# --------------------------------------------------------------------------------------
TYPES = ["entry_conflict_resolution", "exit_conflict_resolution", "conflict_resolution", "entry_coordination",
         "exit_coordination", "assume_aircraft", "transfer_aircraft", "non_conformance_resolution",
         "qos_improvement", "entry_conditions", "exit_conditions", "zone_conflict", "return_to_route",
         "aircraft_request"]
TYPE_RE = re.compile(r"^(?:" + "|".join(TYPES) + r")_(.*)$")

# colonne che NON sono feature (identificatori / etichetta)
ID_COLS = {"sample_id", "row_index", "session", "atco", "scenario", "timestamp", "task_name", "target_index"}

CONFLICT_IDX = [0, 1, 6]

BASE_PARAMS = dict(learning_rate=0.04, num_leaves=24, max_depth=6, min_data_in_leaf=80, feature_fraction=0.5,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=20.0, max_bin=127, verbosity=-1,
                   force_col_wise=True)
OBJECTIVES = {
    "lambdarank": dict(objective="lambdarank", lambdarank_truncation_level=10, label_gain=[0, 1]),
    "xendcg": dict(objective="rank_xendcg"),
    "binary": dict(objective="binary"),
}
WINDOWS = [(0, 0), (1, 1), (2, 2), (3, 3), (5, 5), (2, 0), (4, 0), (8, 0)]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --------------------------------------------------------------------------------------
# Dati e feature
# --------------------------------------------------------------------------------------
def read_csv(path):
    df = pd.read_csv(path, low_memory=False)
    return df.astype({c: np.float32 for c in df.columns[df.dtypes == np.float64]})

def get_rel_cols(df):
    """Trova dinamicamente tutte le colonne numeriche utili per i rank relativi."""
    exclude = ID_COLS.union({"target_index", "row_index"})
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def add_features(df, row_pos=False):
    """Aggiunge feature relative allo snapshot in modo robusto."""
    df = df.sort_values(["sample_id", "row_index"], kind="stable").reset_index(drop=True)
    
    # 1. Identifica la colonna dell'aereo (callsign)
    if "aircraft" in df.columns:
        df["_cs"] = df["aircraft"]
    elif "callsign" in df.columns:
        df["_cs"] = df["callsign"]
    else:
        cs_all = df["task_name"].str.extract(TYPE_RE)[0]
        df["_cs"] = cs_all.str.split("_").str[0]
        
    # 2. Crea un ID univoco per il task (Aereo + Tipo) per lo smoothing
    task_col = "task_type" if "task_type" in df.columns else "task_name"
    df["_task_id"] = df["_cs"].astype(str) + "_" + df[task_col].astype(str)

    if "time_to_sector" in df.columns:
        df["abs_tts"] = df["time_to_sector"].abs()
    else:
        df["abs_tts"] = 0.0

    g = df.groupby("sample_id", sort=False)
    new = {}
    
    # 3. Feature relative dinamiche (Rank, Distanza dal min, Z-score)
    rel_cols = get_rel_cols(df)
    for c in rel_cols:
        grp = g[c]
        new[f"{c}__rk"] = grp.rank(pct=True, na_option='keep')
        new[f"{c}__dmin"] = df[c] - grp.transform("min")
        std = grp.transform("std")
        new[f"{c}__z"] = (df[c] - grp.transform("mean")) / (std.fillna(0) + 1e-6)
        
    # 4. Rank relativi dentro lo stesso tipo di task
    gt = df.groupby(["sample_id", task_col], sort=False)
    for c in ["time_to_sector", "x_task_age", "x_state_age", "a_since_flight", "k_timeToExitPoint"]:
        if c in df.columns:
            new[f"{c}__rkT"] = gt[c].rank(pct=True, na_option='keep')

    if row_pos and "seq_len" in df.columns:
        new["row_pos"] = df["row_index"] / df["seq_len"]
        
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # 5. Feature conflitto (l'aereo è coinvolto in un conflitto?)
    if task_col == "task_type":
        conf_types = [t for t in TYPES if "conflict" in t]
        conf = df.loc[df[task_col].isin(conf_types), ["sample_id", "_cs"]]
    else:
        conf = df.loc[df[task_col].str.contains("conflict", na=False), ["sample_id", "_cs"]]
        
    if len(conf):
        conf = conf.drop_duplicates()
        conf["ac_in_conflict"] = np.int8(1)
        n0 = len(df)
        df = df.merge(conf, on=["sample_id", "_cs"], how="left")
        assert len(df) == n0
        df["ac_in_conflict"] = df["ac_in_conflict"].fillna(0).astype(np.int8)
    else:
        df["ac_in_conflict"] = np.int8(0)

    return df.astype({c: np.float32 for c in df.columns[df.dtypes == np.float64]})

def add_blocks(df, gap=5.0):
    """Blocchi di snapshot contigui per lo smoothing temporale."""
    s = df.groupby("sample_id", sort=False).agg(session=("session", "first"), ts=("timestamp", "first")).reset_index()
    s["sec"] = pd.to_timedelta(s["ts"]).dt.total_seconds()
    s = s.sort_values(["session", "sec", "sample_id"], kind="stable")
    dt = s.groupby("session", sort=False)["sec"].diff()
    s["blk"] = (dt.isna() | (dt > gap) | (dt < 0)).cumsum()
    out = df.copy()
    out["_blk"] = out["sample_id"].map(s.set_index("sample_id")["blk"]).astype(np.int64)
    return out

def feature_cols(df):
    return [c for c in df.columns if c not in ID_COLS and not c.startswith("_")]

# --------------------------------------------------------------------------------------
# Metrica, normalizzazione, smoothing, submission
# --------------------------------------------------------------------------------------
def top3_hit(sid, y, s):
    d = pd.DataFrame({"sid": sid, "s": s, "y": y})
    d["r"] = d.groupby("sid", sort=False)["s"].rank(ascending=False, method="first")
    return float((d.loc[d["y"] == 1, "r"] <= 3).mean())

def zscore(sid, s):
    d = pd.DataFrame({"sid": sid, "s": s})
    g = d.groupby("sid", sort=False)["s"]
    return ((d["s"] - g.transform("mean")) / (g.transform("std").fillna(0.0) + 1e-9)).values

def smooth(blk, task_id, sid, s, back, fwd):
    """Media mobile raggruppata per ID univoco del task (Aereo + Tipo)."""
    if back == 0 and fwd == 0:
        return s
    gid = pd.DataFrame({"b": blk, "n": task_id}).groupby(["b", "n"], sort=False).ngroup().values
    order = np.lexsort((sid, gid))
    g_s, v = gid[order], s[order]
    n = len(v)
    idx = np.arange(n)
    first = np.r_[True, g_s[1:] != g_s[:-1]]
    last = np.r_[g_s[1:] != g_s[:-1], True]
    start_of = np.maximum.accumulate(np.where(first, idx, 0))
    end_of = np.minimum.accumulate(np.where(last, idx, n)[::-1])[::-1]
    lo = np.maximum(idx - back, start_of)
    hi = np.minimum(idx + fwd, end_of)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    out = np.empty(n)
    out[order] = (cs[hi + 1] - cs[lo]) / (hi - lo + 1)
    return out

def make_submission(df, score):
    t = pd.DataFrame({"sample_id": df["sample_id"].values, "row_index": df["row_index"].values, "s": score})
    t = t.sort_values(["sample_id", "s"], ascending=[True, False], kind="stable")
    top = t.groupby("sample_id", sort=True).head(3)
    return (top.groupby("sample_id", sort=True)["row_index"].apply(lambda r: " ".join(map(str, r)))
            .rename("prediction").reset_index())

def official_score(df_labelled, score, data_dir):
    for p in (os.path.dirname(os.path.abspath(__file__)), data_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from data import scorer as scorer
    except Exception:
        try:
            from data import scorer
        except Exception:
            return None
    sub = make_submission(df_labelled, score)
    sol = (df_labelled.loc[df_labelled["target_index"] == 1, ["sample_id", "row_index"]]
           .rename(columns={"row_index": "target"}))
    return scorer.score(sol, sub, "sample_id")

# --------------------------------------------------------------------------------------
# Modelli
# --------------------------------------------------------------------------------------
def group_sizes(sid):
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    return np.diff(np.r_[starts, len(sid)])

def train_model(name, X, y, sid, rounds, seed, lr):
    params = dict(BASE_PARAMS, **OBJECTIVES[name], seed=seed, learning_rate=lr)
    grp = group_sizes(sid) if name != "binary" else None
    
    # Trova dinamicamente le colonne categoriche
    cat_feats = [c for c in X.columns if X[c].dtype == 'object' or c == 'task_index']
    
    ds = lgb.Dataset(X, label=y, group=grp, categorical_feature=cat_feats)
    return lgb.train(params, ds, num_boost_round=rounds)

def cv_oof(name, X, y, sid, atco, n_folds, k_list, lr, seed=0):
    oof = {k: np.zeros(len(X)) for k in k_list}
    for tr, va in GroupKFold(n_splits=n_folds).split(X, groups=atco):
        bst = train_model(name, X.iloc[tr], y[tr], sid[tr], max(k_list), seed, lr)
        Xva = X.iloc[va]
        for k in k_list:
            oof[k][va] = bst.predict(Xva, num_iteration=k)
    return oof

def pick_window(df, sid, y, ens, mode):
    base = top3_hit(sid, y, ens)
    if mode == "off":
        return (0, 0), base
    log("smoothing temporale sulle predizioni OOF (indietro, avanti) -> top-3:")
    best, best_sc = (0, 0), base
    for back, fwd in WINDOWS:
        if mode == "causal" and fwd > 0:
            continue
        # Usa _task_id per non mischiare aerei diversi
        sc = top3_hit(sid, y, smooth(df["_blk"].values, df["_task_id"].values, sid, ens, back, fwd))
        log(f"    ({back},{fwd}){' causale' if fwd == 0 and back > 0 else '        '}  {sc:.4f}")
        if sc > best_sc + 0.002:
            best, best_sc = (back, fwd), sc
    return best, best_sc

# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="directory dei dati")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--quick", action="store_true", help="prova rapida")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", default="lambdarank,xendcg,binary")
    ap.add_argument("--smooth", choices=["auto", "causal", "off"], default="auto")
    ap.add_argument("--row-pos", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    if args.quick:
        lr, k_list, seeds, folds = 0.08, [40, 80, 150], 1, min(args.folds, 3)
    else:
        lr, k_list, seeds, folds = BASE_PARAMS["learning_rate"], [75, 150, 250, 400, 600], args.seeds, args.folds

    d = args.data_dir
    log("carico e preparo i dati ...")
    train = add_blocks(add_features(read_csv(f"{d}/train.csv"), args.row_pos))
    test = add_blocks(add_features(read_csv(f"{d}/test.csv"), args.row_pos))
    vpath = f"{d}/validation.csv"
    val = add_blocks(add_features(read_csv(vpath), args.row_pos)) if os.path.exists(vpath) else None
    
    feats = feature_cols(train)
    missing = [c for c in feats if c not in test.columns]
    if missing:
        sys.exit(f"colonne di train assenti in test: {missing}")
        
    log(f"{train['sample_id'].nunique()} campioni train, {len(feats)} feature, "
        f"{train['atco'].nunique()} controllori, {len(models)} modelli")
        
    X = train[feats]
    y = train["target_index"].values.astype(int)
    sid = train["sample_id"].values
    folds = min(folds, train["atco"].nunique())
    if folds < 2:
        sys.exit("servono almeno 2 controllori distinti in train.csv per la CV per gruppi")

    # 1) CV leave-controller-out ------------------------------------------------------
    best_k, oof_z = {}, []
    log(f"CV per controllore ({folds} fold) - baseline 'task piu' vicino al settore': "
        f"{top3_hit(sid, y, -train['abs_tts'].values):.4f}")
    for m in models:
        t0 = time.time()
        oof = cv_oof(m, X, y, sid, train["atco"].values, folds, k_list, lr)
        sc = {k: top3_hit(sid, y, oof[k]) for k in k_list}
        best_k[m] = max(sc, key=sc.get)
        oof_z.append(zscore(sid, oof[best_k[m]]))
        log(f"  {m:10s} top-3 per n. round {{{', '.join(f'{k}: {v:.4f}' for k, v in sc.items())}}}  "
            f"-> uso {best_k[m]}  ({time.time() - t0:.0f}s)")
    ens_oof = np.mean(oof_z, axis=0)
    log(f"  ENSEMBLE OOF: {top3_hit(sid, y, ens_oof):.4f}")
    window, w_sc = pick_window(train, sid, y, ens_oof, args.smooth)
    log(f"finestra scelta {window}  -> OOF {w_sc:.4f}")

    # 2) stima onesta su validation.csv ---------------------------------------------------
    if val is not None:
        Xv, sv, yv = val[feats], val["sample_id"].values, val["target_index"].values.astype(int)
        zs = []
        for m in models:
            bst = train_model(m, X, y, sid, best_k[m], 0, lr)
            p = bst.predict(Xv)
            zs.append(zscore(sv, p))
            log(f"  validation {m:10s} {top3_hit(sv, yv, p):.4f}")
        ens_v = np.mean(zs, axis=0)
        # Usa _task_id per lo smoothing
        ens_vs = smooth(val["_blk"].values, val["_task_id"].values, sv, ens_v, *window)
        log(f"  validation ENSEMBLE {top3_hit(sv, yv, ens_v):.4f} | con smoothing {window}: "
            f"{top3_hit(sv, yv, ens_vs):.4f}")
        off = official_score(val, ens_vs, d)
        if off is not None:
            log(f"  >>> PUNTEGGIO UFFICIALE (scorer.py) sulla validation: {off:.4f} <<<")

    # 3) modello finale su train + validation ---------------------------------------------
    full = pd.concat([train, val], ignore_index=True) if val is not None else train
    Xf, yf, sf = full[feats], full["target_index"].values.astype(int), full["sample_id"].values
    Xt, st = test[feats], test["sample_id"].values
    zs, imp = [], None
    for m in models:
        preds = []
        for s in range(seeds):
            bst = train_model(m, Xf, yf, sf, int(best_k[m] * 1.1), 1000 + s, lr)
            preds.append(bst.predict(Xt))
            if imp is None:
                imp = pd.Series(bst.feature_importance("gain"), index=feats)
        zs.append(zscore(st, np.mean(preds, axis=0)))
        log(f"  finale {m}: {seeds} seed addestrati")
        
    # Usa _task_id per lo smoothing finale
    ens_t = smooth(test["_blk"].values, test["_task_id"].values, st, np.mean(zs, axis=0), *window)

    sub = make_submission(test, ens_t)
    
    # Allineamento sicuro con sample_submission.csv se presente
    ref = f"{d}/sample_submission.csv"
    if os.path.exists(ref):
        ref_ids = pd.read_csv(ref)["sample_id"]
        sub = sub.set_index("sample_id").loc[ref_ids].reset_index()
        
    sub.to_csv(args.out, index=False)
    imp.sort_values(ascending=False).to_csv(os.path.splitext(args.out)[0] + "_feature_importance.csv")
    log(f"scritto {args.out} ({len(sub)} righe). Top-10 feature per gain:")
    print((imp / imp.sum()).sort_values(ascending=False).head(10).round(3).to_string())

if __name__ == "__main__":
    main()