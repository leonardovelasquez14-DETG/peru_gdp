#!/usr/bin/env python3
"""
Etapa 3 + Etapa 5 (parte clasica): Modelos de referencia en esquema expansivo causal.

Implementa y evalua ocho especificaciones de referencia estrictamente en tiempo real
sobre el objetivo principal Y_t^(11) = y_t^(v11) - y_t^(v0) (n = 350 limpios, con
n_oos = 250 predicciones entre 2003m2 y 2024m11 y bloque congelado final 2019m12-2024m11,
n = 60):
  1. Zero              : no-cambio (Y_hat = 0)
  2. BiasCorr_Mean     : sesgo medio historico expansivo causal
  3. BiasCorr_Roll60   : media movil causal de las ultimas 60 revisiones completadas
  4. AR_BIC            : autorregresivo sobre rezagos observables completados Y_lag11..13 (BIC)
  5. ARX_BIC           : AR_BIC + y0 y armonicos estacionales sin1, cos1 (BIC)
  6. Ridge             : regresion Ridge sobre los 20 predictores observables (benchmark principal)
  7. DFM_Factor_Ridge  : 2 factores PCA sectoriales + rezagos/estacionalidad con Ridge
  8. GBDT_NumPy        : arboles impulsados por gradiente en NumPy puro (XGBoost no instalado)

Genera artefactos CSV/JSON/SHA-256 en output/etapa3_clasicos/ y figuras en
main_tex/images/ (copia exacta en output/etapa3_clasicos/).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rtd_eval_utils as u  # noqa: E402

ANALYSIS_DIR = ROOT_DIR / "output" / "analysis_output"
OUT_DIR = ROOT_DIR / "output" / "etapa3_clasicos"
TEX_IMG_DIR = ROOT_DIR / "main_tex" / "images"

EVAL_START_IDX = 100  # 2003m2 (v0 = 2003-04, 90 obs completadas de entrenamiento)
FROZEN_START_PERIOD = "2019m12"  # Ultimos 60 meses limpios (2019m12..2024m11)

RIDGE_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0)
DFM_RIDGE_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)
GBDT_DEPTH_GRID = (1, 2)
GBDT_LR_GRID = (0.03, 0.06)
GBDT_NTREES_GRID = (25, 50)

AR_LAGS = ("Y_lag11", "Y_lag12", "Y_lag13")
ARX_EXOG = ("y0", "sin1", "cos1")
SECTOR_COLS = [f"sec_{s}" for s in u.FULL_SECTORS]
DFM_CORE_NONSEC = ("y0", "Y_lag11", "Y_lag12", "Y_obs_mean6", "sin1", "cos1")

MODEL_ORDER = [
    "Zero",
    "BiasCorr_Mean",
    "BiasCorr_Roll60",
    "AR_BIC",
    "ARX_BIC",
    "Ridge",
    "DFM_Factor_Ridge",
    "GBDT_NumPy",
]

MODEL_LABELS_ES = {
    "Zero": r"Referencia nula ($\widehat{Y}=0$)",
    "BiasCorr_Mean": "Corrección clásica (media expansiva)",
    "BiasCorr_Roll60": "Corrección clásica (media móvil 60m)",
    "AR_BIC": "AR(p) observable (BIC)",
    "ARX_BIC": "ARX(p) observable (BIC)",
    "Ridge": "Ridge (benchmark clásico)",
    "DFM_Factor_Ridge": "Factor DFM-PCA + Ridge",
    "GBDT_NumPy": "GBDT (NumPy, árboles acotados)",
}


def fit_ols_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> float:
    """Ajusta MCO con intercepto mediante minimos cuadrados numericamente estables."""
    n = x_tr.shape[0]
    a_tr = np.column_stack([np.ones(n), x_tr]) if x_tr.shape[1] > 0 else np.ones((n, 1))
    beta, _, _, _ = np.linalg.lstsq(a_tr, y_tr, rcond=None)
    a_te = np.concatenate(([1.0], x_te)) if x_te.size > 0 else np.array([1.0])
    return float(np.dot(a_te, beta))


def select_ar_order_bic_and_predict(
    df: pd.DataFrame,
    tr_idx: np.ndarray,
    eval_idx: int,
    max_p: int = 3,
    exog_cols: Sequence[str] = (),
) -> Tuple[float, int]:
    """
    Selecciona orden AR p in {1..max_p} por BIC sobre submuestra completa comun
    de la ventana causal e imputa con mediana causal si el punto test tiene NaN.
    """
    all_lag_cols = list(AR_LAGS[:max_p])
    req_cols = all_lag_cols + list(exog_cols)
    sub = df.loc[tr_idx, req_cols + ["Y_H11"]]
    complete_mask = sub.notna().all(axis=1).to_numpy()

    y_full = df.loc[tr_idx, "Y_H11"].to_numpy(dtype=float)
    if complete_mask.sum() < max_p + 12:
        return float(np.mean(y_full)), 0

    best_bic = np.inf
    best_p = 1
    sub_complete = sub.iloc[complete_mask]
    y_c = sub_complete["Y_H11"].to_numpy(dtype=float)
    n_c = len(y_c)

    for p in range(1, max_p + 1):
        cols_p = list(AR_LAGS[:p]) + list(exog_cols)
        x_c = sub_complete[cols_p].to_numpy(dtype=float)
        design = np.column_stack([np.ones(n_c), x_c])
        beta, _, _, _ = np.linalg.lstsq(design, y_c, rcond=None)
        rss = float(np.sum((y_c - design @ beta) ** 2))
        sigma2 = max(rss / n_c, 1e-12)
        k_params = design.shape[1] + 1
        bic = n_c * np.log(sigma2) + k_params * np.log(n_c)
        if bic < best_bic:
            best_bic = bic
            best_p = p

    cols_best = list(AR_LAGS[:best_p]) + list(exog_cols)
    fit_df = df.loc[tr_idx, cols_best + ["Y_H11"]].dropna()
    x_tr_best = fit_df[cols_best].to_numpy(dtype=float)
    y_tr_best = fit_df["Y_H11"].to_numpy(dtype=float)
    x_te_raw = df.loc[eval_idx, cols_best].to_numpy(dtype=float)

    # Si algun rezago puntual en test es NaN (p. ej. tras cruce de año base), usar mediana causal
    if np.isnan(x_te_raw).any():
        medians = np.median(x_tr_best, axis=0)
        x_te_clean = np.where(np.isnan(x_te_raw), medians, x_te_raw)
    else:
        x_te_clean = x_te_raw

    return fit_ols_predict(x_tr_best, y_tr_best, x_te_clean), best_p


def fit_ridge_closed_form(
    x_tr_sc: np.ndarray, y_tr: np.ndarray, x_te_sc: np.ndarray, lam: float
) -> np.ndarray:
    """Ajusta Ridge centrado (sin penalizar intercepto) sobre matriz ya estandarizada."""
    y_mean = float(np.mean(y_tr))
    yc = y_tr - y_mean
    p = x_tr_sc.shape[1]
    gram = x_tr_sc.T @ x_tr_sc + lam * np.eye(p)
    rhs = x_tr_sc.T @ yc
    beta = np.linalg.solve(gram, rhs)
    return y_mean + (x_te_sc @ beta)


def build_dfm_features(
    x_sec_tr_raw: np.ndarray,
    x_nonsec_tr_raw: np.ndarray,
    x_sec_te_raw: np.ndarray,
    x_nonsec_te_raw: np.ndarray,
    n_factors: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae n_factors PCA estaticos estrictamente dentro del entrenamiento y combina."""
    sec_tr_sc, sec_te_sc = u.fit_impute_and_scale(x_sec_tr_raw, x_sec_te_raw)
    _, _, vt = np.linalg.svd(sec_tr_sc, full_matrices=False)
    k = min(n_factors, vt.shape[0])
    f_tr = sec_tr_sc @ vt[:k].T
    f_te = sec_te_sc @ vt[:k].T

    raw_tr = np.hstack([f_tr, x_nonsec_tr_raw])
    raw_te = np.hstack([f_te, x_nonsec_te_raw])
    return u.fit_impute_and_scale(raw_tr, raw_te)


def get_inner_val_pairs(
    df: pd.DataFrame, tr_idx: np.ndarray, stride: int = 1
) -> List[Tuple[np.ndarray, int]]:
    """
    Construye pares (inner_train_idx, val_idx) estrictamente causales al final de tr_idx.
    """
    n_tr = len(tr_idx)
    n_val = min(36, max(18, n_tr // 4))
    val_candidates = tr_idx[-n_val:]
    if stride > 1:
        val_candidates = val_candidates[::stride]

    pairs: List[Tuple[np.ndarray, int]] = []
    v_h11_tr = df.loc[tr_idx, "v_H11_date"].to_numpy()
    for val_i in val_candidates:
        v0_val = df.loc[val_i, "v0_date"].to_datetime64()
        mask = (tr_idx < val_i) & (v_h11_tr <= v0_val)
        inner_tr = tr_idx[mask]
        if len(inner_tr) >= 24:
            pairs.append((inner_tr, int(val_i)))
    return pairs


def predict_ridge_causal(
    df: pd.DataFrame,
    x_all: np.ndarray,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    eval_idx: int,
    lam_grid: Sequence[float] = RIDGE_GRID,
) -> Tuple[float, float]:
    """Selecciona lambda por validacion temporal causal interna y predice Ridge."""
    pairs = get_inner_val_pairs(df, tr_idx, stride=1)
    best_lam = 25.0
    if pairs:
        sq_errs = np.zeros(len(lam_grid), dtype=float)
        for inner_tr, val_i in pairs:
            x_i_tr, x_i_val = u.fit_impute_and_scale(x_all[inner_tr], x_all[[val_i]])
            y_i_tr = y_all[inner_tr]
            y_i_val = y_all[val_i]
            # SVD una sola vez por punto interno para evaluar toda la grilla lambda
            y_mean = float(np.mean(y_i_tr))
            yc = y_i_tr - y_mean
            u_mat, s_vals, vt = np.linalg.svd(x_i_tr, full_matrices=False)
            uty = u_mat.T @ yc
            v_x = vt @ x_i_val[0]
            for j, lam in enumerate(lam_grid):
                coef_f = s_vals / (s_vals ** 2 + lam)
                pred_val = y_mean + float(np.dot(v_x, coef_f * uty))
                sq_errs[j] += (y_i_val - pred_val) ** 2
        best_lam = float(lam_grid[int(np.argmin(sq_errs))])

    x_tr_sc, x_te_sc = u.fit_impute_and_scale(x_all[tr_idx], x_all[[eval_idx]])
    pred = float(fit_ridge_closed_form(x_tr_sc, y_all[tr_idx], x_te_sc, best_lam)[0])
    return pred, best_lam


def predict_dfm_causal(
    df: pd.DataFrame,
    x_sec_all: np.ndarray,
    x_nonsec_all: np.ndarray,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    eval_idx: int,
    lam_grid: Sequence[float] = DFM_RIDGE_GRID,
) -> Tuple[float, float]:
    """Factor DFM-PCA + Ridge con extraccion de factores y lambda causales."""
    pairs = get_inner_val_pairs(df, tr_idx, stride=1)
    best_lam = 10.0
    if pairs:
        sq_errs = np.zeros(len(lam_grid), dtype=float)
        for inner_tr, val_i in pairs:
            x_i_tr, x_i_val = build_dfm_features(
                x_sec_all[inner_tr],
                x_nonsec_all[inner_tr],
                x_sec_all[[val_i]],
                x_nonsec_all[[val_i]],
            )
            y_i_tr = y_all[inner_tr]
            y_i_val = y_all[val_i]
            y_mean = float(np.mean(y_i_tr))
            yc = y_i_tr - y_mean
            u_mat, s_vals, vt = np.linalg.svd(x_i_tr, full_matrices=False)
            uty = u_mat.T @ yc
            v_x = vt @ x_i_val[0]
            for j, lam in enumerate(lam_grid):
                coef_f = s_vals / (s_vals ** 2 + lam)
                pred_val = y_mean + float(np.dot(v_x, coef_f * uty))
                sq_errs[j] += (y_i_val - pred_val) ** 2
        best_lam = float(lam_grid[int(np.argmin(sq_errs))])

    x_tr_dfm, x_te_dfm = build_dfm_features(
        x_sec_all[tr_idx],
        x_nonsec_all[tr_idx],
        x_sec_all[[eval_idx]],
        x_nonsec_all[[eval_idx]],
    )
    pred = float(fit_ridge_closed_form(x_tr_dfm, y_all[tr_idx], x_te_dfm, best_lam)[0])
    return pred, best_lam


def _best_vectorized_split(
    x_node: np.ndarray, r_node: np.ndarray, min_leaf: int = 8
) -> Tuple[int, float, np.ndarray] | None:
    """Encuentra el mejor corte L2 en todas las columnas simultaneamente usando NumPy."""
    m = x_node.shape[0]
    if m < 2 * min_leaf:
        return None

    order = np.argsort(x_node, axis=0, kind="mergesort")
    xs = np.take_along_axis(x_node, order, axis=0)
    rs = r_node[order]

    total_sum = float(np.sum(r_node))
    parent_score = (total_sum ** 2) / m
    left_sum = np.cumsum(rs[:-1, :], axis=0)
    right_sum = total_sum - left_sum
    left_n = np.arange(1, m, dtype=float)[:, None]
    right_n = m - left_n

    valid = (xs[:-1, :] < xs[1:, :] - 1e-12) & (left_n >= min_leaf) & (right_n >= min_leaf)
    if not np.any(valid):
        return None

    scores = np.where(valid, (left_sum ** 2) / left_n + (right_sum ** 2) / right_n, -np.inf)
    flat_idx = int(np.argmax(scores))
    best_score = float(scores.flat[flat_idx])
    if best_score <= parent_score + 1e-10:
        return None

    row_i, col_j = np.unravel_index(flat_idx, scores.shape)
    thr = 0.5 * float(xs[row_i, col_j] + xs[row_i + 1, col_j])
    left_mask = x_node[:, col_j] <= thr
    return int(col_j), thr, left_mask


def _fit_cart_tree(
    x: np.ndarray, r: np.ndarray, max_depth: int, min_leaf: int = 8
) -> Dict[str, object]:
    """Ajusta un arbol de regresion CART de profundidad maxima acotada."""
    if max_depth <= 0 or x.shape[0] < 2 * min_leaf:
        return {"leaf": True, "val": float(np.mean(r))}

    split = _best_vectorized_split(x, r, min_leaf=min_leaf)
    if split is None:
        return {"leaf": True, "val": float(np.mean(r))}

    col, thr, left_mask = split
    return {
        "leaf": False,
        "col": col,
        "thr": thr,
        "left": _fit_cart_tree(x[left_mask], r[left_mask], max_depth - 1, min_leaf),
        "right": _fit_cart_tree(x[~left_mask], r[~left_mask], max_depth - 1, min_leaf),
    }


def _predict_cart_tree(tree: Dict[str, object], x: np.ndarray) -> np.ndarray:
    """Evalua vectorialmente un arbol CART sobre la matriz x."""
    if tree["leaf"]:
        return np.full(x.shape[0], float(tree["val"]), dtype=float)
    mask = x[:, int(tree["col"])] <= float(tree["thr"])
    out = np.empty(x.shape[0], dtype=float)
    if np.any(mask):
        out[mask] = _predict_cart_tree(tree["left"], x[mask])  # type: ignore[arg-type]
    if np.any(~mask):
        out[~mask] = _predict_cart_tree(tree["right"], x[~mask])  # type: ignore[arg-type]
    return out


def fit_gbdt_staged_predictions(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    max_depth: int,
    learning_rate: float,
    checkpoints: Sequence[int] = GBDT_NTREES_GRID,
    min_leaf: int = 8,
) -> Dict[int, float]:
    """Entrena GBDT con arboles NumPy y devuelve predicciones en varios cortes n_trees."""
    max_trees = max(checkpoints)
    ckpt_set = set(checkpoints)
    f0 = float(np.mean(y_tr))
    pred_tr = np.full(len(y_tr), f0, dtype=float)
    pred_te = np.full(x_te.shape[0], f0, dtype=float)
    out: Dict[int, float] = {}

    for m in range(1, max_trees + 1):
        residuals = y_tr - pred_tr
        tree = _fit_cart_tree(x_tr, residuals, max_depth=max_depth, min_leaf=min_leaf)
        pred_tr += learning_rate * _predict_cart_tree(tree, x_tr)
        pred_te += learning_rate * _predict_cart_tree(tree, x_te)
        if m in ckpt_set:
            out[m] = float(pred_te[0])
    return out


def predict_gbdt_causal(
    df: pd.DataFrame,
    x_all: np.ndarray,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    eval_idx: int,
    cached_cfg: Tuple[int, float, int] | None = None,
    retune_hp: bool = True,
) -> Tuple[float, Tuple[int, float, int]]:
    """
    Predice con GBDT en NumPy puro. Selecciona hiperparametros sobre puntos internos
    causales (stride=3 -> ~12 puntos por ventana cuando retune_hp=True).
    """
    best_cfg = cached_cfg if cached_cfg is not None else (1, 0.03, 25)
    if retune_hp or cached_cfg is None:
        pairs = get_inner_val_pairs(df, tr_idx, stride=3)
        if pairs:
            cfgs = [
                (d, lr, nt)
                for d in GBDT_DEPTH_GRID
                for lr in GBDT_LR_GRID
                for nt in GBDT_NTREES_GRID
            ]
            sq_errs = {cfg: 0.0 for cfg in cfgs}
            for inner_tr, val_i in pairs:
                x_i_tr, x_i_val = u.fit_impute_and_scale(x_all[inner_tr], x_all[[val_i]])
                y_i_tr = y_all[inner_tr]
                y_i_val = y_all[val_i]
                for d in GBDT_DEPTH_GRID:
                    for lr in GBDT_LR_GRID:
                        staged = fit_gbdt_staged_predictions(
                            x_i_tr, y_i_tr, x_i_val, max_depth=d, learning_rate=lr
                        )
                        for nt in GBDT_NTREES_GRID:
                            sq_errs[(d, lr, nt)] += (y_i_val - staged[nt]) ** 2
            best_cfg = min(cfgs, key=lambda c: sq_errs[c])

    x_tr_sc, x_te_sc = u.fit_impute_and_scale(x_all[tr_idx], x_all[[eval_idx]])
    d_opt, lr_opt, nt_opt = best_cfg
    pred_map = fit_gbdt_staged_predictions(
        x_tr_sc,
        y_all[tr_idx],
        x_te_sc,
        max_depth=d_opt,
        learning_rate=lr_opt,
        checkpoints=(nt_opt,),
    )
    return pred_map[nt_opt], best_cfg


def assign_subregimes(df_oos: pd.DataFrame) -> pd.DataFrame:
    """Asigna mascaras de subperiodos economicos y particion congelada sobre OOS."""
    out = df_oos.copy()
    dt = out["target_date"]
    out["regimen_hist"] = np.where(
        dt <= pd.Timestamp("2013-02-01"),
        "pre_base_2003m2_2013m2",
        np.where(
            dt < pd.Timestamp("2019-12-01"),
            "post_base_pre_covid_2014m1_2019m11",
            "congelado_covid_post_2019m12_2024m11",
        ),
    )
    out["bloque_eval"] = np.where(
        dt < pd.Timestamp("2019-12-01"),
        "pre_congelado_2003m2_2019m11",
        "congelado_2019m12_2024m11",
    )
    return out


def run_expanding_backtest(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el backtest expansivo causal completo para los 8 modelos clasicos."""
    x_all = df[u.FEATURE_COLS].to_numpy(dtype=float)
    x_sec_all = df[SECTOR_COLS].to_numpy(dtype=float)
    x_nonsec_all = df[list(DFM_CORE_NONSEC)].to_numpy(dtype=float)
    y_all = df["Y_H11"].to_numpy(dtype=float)

    pred_rows: List[Dict[str, object]] = []
    hp_rows: List[Dict[str, object]] = []

    gbdt_cfg_cache: Tuple[int, float, int] | None = None
    t0 = time.perf_counter()

    for step, eval_idx in enumerate(range(EVAL_START_IDX, len(df))):
        tr_idx = u.get_causal_train_indices(df, eval_idx)
        y_tr = y_all[tr_idx]

        # Asegurar causalidad estricta sin fuga futura
        assert np.all(df.loc[tr_idx, "v_H11_date"] <= df.loc[eval_idx, "v0_date"])

        p_zero = 0.0
        p_mean = float(np.mean(y_tr))
        p_roll60 = float(np.mean(y_tr[-min(60, len(y_tr)) :]))
        p_ar, p_ar_order = select_ar_order_bic_and_predict(df, tr_idx, eval_idx, max_p=3)
        p_arx, p_arx_order = select_ar_order_bic_and_predict(
            df, tr_idx, eval_idx, max_p=3, exog_cols=ARX_EXOG
        )
        p_ridge, lam_ridge = predict_ridge_causal(df, x_all, y_all, tr_idx, eval_idx)
        p_dfm, lam_dfm = predict_dfm_causal(
            df, x_sec_all, x_nonsec_all, y_all, tr_idx, eval_idx
        )
        # Re-seleccionar hiperparametros GBDT cada trimestre en ventana (refit completo cada mes)
        p_gbdt, gbdt_cfg_cache = predict_gbdt_causal(
            df,
            x_all,
            y_all,
            tr_idx,
            eval_idx,
            cached_cfg=gbdt_cfg_cache,
            retune_hp=(step % 3 == 0),
        )

        row_base = df.loc[eval_idx]
        pred_rows.append(
            {
                "eval_idx": eval_idx,
                "target_period": row_base["target_period"],
                "target_date": row_base["target_date"],
                "v0_date": row_base["v0_date"],
                "v_H11_date": row_base["v_H11_date"],
                "n_train": int(len(tr_idx)),
                "Y_H11": float(row_base["Y_H11"]),
                "y0": float(row_base["y0"]),
                "y_H11": float(row_base["y_H11"]),
                "pred_Zero": p_zero,
                "pred_BiasCorr_Mean": p_mean,
                "pred_BiasCorr_Roll60": p_roll60,
                "pred_AR_BIC": p_ar,
                "pred_ARX_BIC": p_arx,
                "pred_Ridge": p_ridge,
                "pred_DFM_Factor_Ridge": p_dfm,
                "pred_GBDT_NumPy": p_gbdt,
            }
        )
        hp_rows.append(
            {
                "eval_idx": eval_idx,
                "target_period": row_base["target_period"],
                "n_train": int(len(tr_idx)),
                "ar_p_bic": p_ar_order,
                "arx_p_bic": p_arx_order,
                "ridge_lambda": lam_ridge,
                "dfm_lambda": lam_dfm,
                "gbdt_depth": gbdt_cfg_cache[0],
                "gbdt_lr": gbdt_cfg_cache[1],
                "gbdt_ntrees": gbdt_cfg_cache[2],
            }
        )

    elapsed = time.perf_counter() - t0
    print(f"[Etapa 3+5] Backtest causal completado ({len(pred_rows)} origenes) en {elapsed:.2f}s")
    return assign_subregimes(pd.DataFrame(pred_rows)), pd.DataFrame(hp_rows)


def evaluate_tables(
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calcula metricas globales, por subperiodos, comparaciones DM/bootstrap y Clark-West."""
    y = pred_df["Y_H11"].to_numpy(dtype=float)

    global_rows: List[Dict[str, object]] = []
    for m in MODEL_ORDER:
        yhat = pred_df[f"pred_{m}"].to_numpy(dtype=float)
        met = u.compute_metrics(y, yhat)
        met["modelo"] = m
        met["label_es"] = MODEL_LABELS_ES[m]
        global_rows.append(met)
    global_df = pd.DataFrame(global_rows)[
        ["modelo", "label_es", "n", "rmse", "mae", "med_ae", "sesgo_medio", "r2_oos"]
    ]

    # Subperiodos exhaustivos + bloque pre-congelado
    slices: List[Tuple[str, pd.Series]] = [
        ("pre_base_2003m2_2013m2", pred_df["regimen_hist"] == "pre_base_2003m2_2013m2"),
        (
            "post_base_pre_covid_2014m1_2019m11",
            pred_df["regimen_hist"] == "post_base_pre_covid_2014m1_2019m11",
        ),
        ("pre_congelado_2003m2_2019m11", pred_df["bloque_eval"] == "pre_congelado_2003m2_2019m11"),
        ("congelado_2019m12_2024m11", pred_df["bloque_eval"] == "congelado_2019m12_2024m11"),
    ]
    sub_rows: List[Dict[str, object]] = []
    for sname, mask in slices:
        sub = pred_df[mask]
        ys = sub["Y_H11"].to_numpy(dtype=float)
        for m in MODEL_ORDER:
            yhat_s = sub[f"pred_{m}"].to_numpy(dtype=float)
            met = u.compute_metrics(ys, yhat_s)
            met["subperiodo"] = sname
            met["modelo"] = m
            sub_rows.append(met)
    sub_df = pd.DataFrame(sub_rows)[
        ["subperiodo", "modelo", "n", "rmse", "mae", "med_ae", "sesgo_medio", "r2_oos"]
    ]

    # Comparaciones DM-HLN y bootstrap vs Ridge y vs BiasCorr_Mean en global y bloques
    comp_scopes = [
        ("global_2003m2_2024m11", pred_df),
        ("pre_congelado_2003m2_2019m11", pred_df[pred_df["bloque_eval"] == "pre_congelado_2003m2_2019m11"]),
        ("congelado_2019m12_2024m11", pred_df[pred_df["bloque_eval"] == "congelado_2019m12_2024m11"]),
    ]
    comp_rows: List[Dict[str, object]] = []
    for scope_name, df_s in comp_scopes:
        ys = df_s["Y_H11"].to_numpy(dtype=float)
        e_ridge = ys - df_s["pred_Ridge"].to_numpy(dtype=float)
        e_zero = ys - df_s["pred_Zero"].to_numpy(dtype=float)
        e_bias = ys - df_s["pred_BiasCorr_Mean"].to_numpy(dtype=float)

        for m in MODEL_ORDER:
            e_m = ys - df_s[f"pred_{m}"].to_numpy(dtype=float)
            d_sq_r, dm_sq_r, p_sq_r = u.diebold_mariano_test(e_m, e_ridge, loss="sq")
            _, dm_abs_r, p_abs_r = u.diebold_mariano_test(e_m, e_ridge, loss="abs")
            diff_r, ci_lo_r, ci_hi_r = u.block_bootstrap_rmse_diff(e_m, e_ridge, block_len=12, n_boot=2000, seed=42)

            _, dm_sq_b, p_sq_b = u.diebold_mariano_test(e_m, e_bias, loss="sq")
            diff_b, ci_lo_b, ci_hi_b = u.block_bootstrap_rmse_diff(e_m, e_bias, block_len=12, n_boot=2000, seed=42)

            _, dm_sq_z, p_sq_z = u.diebold_mariano_test(e_m, e_zero, loss="sq")
            diff_z, ci_lo_z, ci_hi_z = u.block_bootstrap_rmse_diff(e_m, e_zero, block_len=12, n_boot=2000, seed=42)

            comp_rows.append(
                {
                    "alcance": scope_name,
                    "n": int(len(ys)),
                    "modelo": m,
                    "d_mse_vs_ridge": d_sq_r,
                    "dm_hln_sq_vs_ridge": dm_sq_r,
                    "p_dm_sq_vs_ridge": p_sq_r,
                    "dm_hln_abs_vs_ridge": dm_abs_r,
                    "p_dm_abs_vs_ridge": p_abs_r,
                    "rmse_diff_vs_ridge": diff_r,
                    "ci95_low_vs_ridge": ci_lo_r,
                    "ci95_high_vs_ridge": ci_hi_r,
                    "dm_hln_sq_vs_biasmean": dm_sq_b,
                    "p_dm_sq_vs_biasmean": p_sq_b,
                    "rmse_diff_vs_biasmean": diff_b,
                    "ci95_low_vs_biasmean": ci_lo_b,
                    "ci95_high_vs_biasmean": ci_hi_b,
                    "dm_hln_sq_vs_zero": dm_sq_z,
                    "p_dm_sq_vs_zero": p_sq_z,
                    "rmse_diff_vs_zero": diff_z,
                    "ci95_low_vs_zero": ci_lo_z,
                    "ci95_high_vs_zero": ci_hi_z,
                }
            )
    comp_df = pd.DataFrame(comp_rows)

    # Pruebas Clark-West sobre pares anidados en muestra global
    nested_pairs = [
        ("Zero", "BiasCorr_Mean"),
        ("BiasCorr_Mean", "AR_BIC"),
        ("AR_BIC", "ARX_BIC"),
        ("BiasCorr_Mean", "Ridge"),
        ("ARX_BIC", "Ridge"),
        ("BiasCorr_Mean", "DFM_Factor_Ridge"),
    ]
    cw_rows: List[Dict[str, object]] = []
    for small_m, large_m in nested_pairs:
        yhat_s = pred_df[f"pred_{small_m}"].to_numpy(dtype=float)
        yhat_l = pred_df[f"pred_{large_m}"].to_numpy(dtype=float)
        cw_stat, p_cw = u.clark_west_test(y - yhat_s, yhat_s, y - yhat_l, yhat_l)
        cw_rows.append(
            {
                "alcance": "global_2003m2_2024m11",
                "modelo_restringido": small_m,
                "modelo_ampliado": large_m,
                "cw_stat": cw_stat,
                "p_value_one_sided": p_cw,
            }
        )
    cw_df = pd.DataFrame(cw_rows)
    return global_df, sub_df, comp_df, cw_df


def make_figures(pred_df: pd.DataFrame, sub_df: pd.DataFrame) -> List[Path]:
    """Genera dos figuras de diagnostico OOS para LaTeX y salida de auditoria."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    TEX_IMG_DIR.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    # Figura 1: Serie OOS realizada vs predicciones representativas y error acumulado
    fig, axes = plt.subplots(
        2, 1, figsize=(11.2, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]}
    )
    dates = pred_df["target_date"]
    y = pred_df["Y_H11"]

    ax0 = axes[0]
    ax0.axvspan(
        pd.Timestamp("2019-12-01"),
        pd.Timestamp("2024-11-30"),
        color="#FDEBD0",
        alpha=0.55,
        label="Bloque congelado final (2019m12–2024m11, n=60)",
    )
    ax0.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax0.plot(dates, y, color="#1B365D", lw=1.25, label=r"Realizada $Y_t^{(11)}$", zorder=3)
    ax0.plot(
        dates,
        pred_df["pred_BiasCorr_Mean"],
        color="#6C757D",
        lw=1.35,
        ls="--",
        label="Corrección clásica (media expansiva)",
        zorder=4,
    )
    ax0.plot(
        dates,
        pred_df["pred_Ridge"],
        color="#C84B31",
        lw=1.55,
        label="Ridge (benchmark clásico, 20 variables)",
        zorder=5,
    )
    ax0.plot(
        dates,
        pred_df["pred_GBDT_NumPy"],
        color="#2E8B57",
        lw=1.25,
        alpha=0.9,
        label="GBDT (NumPy, árboles acotados)",
        zorder=4,
    )
    ax0.set_ylabel("Revisión (p.p.)")
    ax0.set_title(
        r"Pronósticos causales fuera de muestra de $Y_t^{(11)}$ (esquema expansivo, 2003m2–2024m11)",
        pad=8,
    )
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.9)

    ax1 = axes[1]
    ax1.axvspan(pd.Timestamp("2019-12-01"), pd.Timestamp("2024-11-30"), color="#FDEBD0", alpha=0.55)
    ax1.axhline(0.0, color="#444444", lw=0.9, ls="--", label="Igual pérdida acumulada que Ridge")
    e_ridge2 = (y - pred_df["pred_Ridge"]) ** 2
    comp_lines = [
        ("Zero", r"Nula ($\widehat{Y}=0$) − Ridge", "#1B365D", "-"),
        ("BiasCorr_Mean", "Media expansiva − Ridge", "#6C757D", "--"),
        ("DFM_Factor_Ridge", "DFM-PCA + Ridge − Ridge", "#D4A017", "-."),
        ("GBDT_NumPy", "GBDT (NumPy) − Ridge", "#2E8B57", "-"),
    ]
    for m, lbl, col, ls in comp_lines:
        cum_diff = np.cumsum((y - pred_df[f"pred_{m}"]) ** 2 - e_ridge2)
        ax1.plot(dates, cum_diff, color=col, lw=1.45, ls=ls, label=lbl)

    ax1.set_ylabel(r"$\Delta$ SE acumulado (vs. Ridge)")
    ax1.set_xlabel("Mes objetivo")
    ax1.set_ylim(-23.0, 50.0)
    ax1.set_title(
        "Diferencia acumulada de error cuadrático respecto a Ridge (valores > 0 favorecen a Ridge)",
        pad=6,
    )
    ax1.grid(axis="y", alpha=0.25)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(loc="upper left", ncol=3, frameon=True, facecolor="white", framealpha=0.94, fontsize=8.0)

    fig.tight_layout()
    for d in (OUT_DIR, TEX_IMG_DIR):
        p = d / "etapa3_oos_predicciones.png"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        generated.append(p)
    plt.close(fig)

    # Figura 2: RMSE fuera de muestra por regímenes y comparación global
    reg_order = [
        ("pre_base_2003m2_2013m2", "Pre-cambio de base\n(2003m2–2013m2, n=119)"),
        ("post_base_pre_covid_2014m1_2019m11", "Post-base pre-COVID\n(2014m1–2019m11, n=71)"),
        ("congelado_2019m12_2024m11", "Bloque congelado\n(2019m12–2024m11, n=60)"),
    ]
    models_plot = [
        ("Zero", "Nula (0)", "#8E9AAF"),
        ("BiasCorr_Mean", "Media exp.", "#6C757D"),
        ("BiasCorr_Roll60", "Móvil 60m", "#495057"),
        ("AR_BIC", "AR (BIC)", "#457B9D"),
        ("ARX_BIC", "ARX (BIC)", "#1B365D"),
        ("Ridge", "Ridge", "#C84B31"),
        ("DFM_Factor_Ridge", "DFM+Ridge", "#D4A017"),
        ("GBDT_NumPy", "GBDT (NumPy)", "#2E8B57"),
    ]

    fig2, ax2 = plt.subplots(figsize=(11.4, 5.2))
    x_pos = np.arange(len(reg_order), dtype=float)
    width = 0.102
    offsets = (np.arange(len(models_plot)) - (len(models_plot) - 1) / 2.0) * width

    for idx_m, (m_key, m_lbl, col) in enumerate(models_plot):
        vals = []
        for reg_key, _ in reg_order:
            row = sub_df[(sub_df["subperiodo"] == reg_key) & (sub_df["modelo"] == m_key)]
            vals.append(float(row["rmse"].iloc[0]))
        bars = ax2.bar(x_pos + offsets[idx_m], vals, width=width * 0.92, color=col, label=m_lbl)
        for b in bars:
            h = b.get_height()
            ax2.text(
                b.get_x() + b.get_width() / 2.0,
                h + 0.025,
                f"{h:.2f}".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=90,
            )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([lbl for _, lbl in reg_order])
    ax2.set_ylim(0.0, float(sub_df["rmse"].max()) * 1.28)
    ax2.set_ylabel("RMSE fuera de muestra (p.p.)")
    ax2.set_title(
        "RMSE fuera de muestra por régimen económico y bloque congelado (Etapas 3 y 5)",
        pad=10,
    )
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper left", ncol=4, frameon=True, facecolor="white", framealpha=0.92)
    fig2.tight_layout()

    for d in (OUT_DIR, TEX_IMG_DIR):
        p = d / "etapa3_rmse_regimenes.png"
        fig2.savefig(p, dpi=220, bbox_inches="tight")
        generated.append(p)
    plt.close(fig2)

    return generated


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_only = "--plot-only" in sys.argv[1:]
    csv_paths = {
        "predicciones_oos": OUT_DIR / "etapa3_predicciones_oos.csv",
        "metricas_globales": OUT_DIR / "etapa3_metricas_globales.csv",
        "metricas_subperiodos": OUT_DIR / "etapa3_metricas_subperiodos.csv",
        "comparacion_estadistica": OUT_DIR / "etapa3_comparacion_estadistica.csv",
        "clark_west": OUT_DIR / "etapa3_clark_west.csv",
        "hiperparametros_hist": OUT_DIR / "etapa3_hiperparametros_hist.csv",
    }

    if plot_only:
        pred_df = pd.read_csv(
            csv_paths["predicciones_oos"],
            parse_dates=["target_date", "v0_date", "v_H11_date"],
        )
        sub_df = pd.read_csv(csv_paths["metricas_subperiodos"])
        global_df = pd.read_csv(csv_paths["metricas_globales"])
    else:
        df = u.build_realtime_feature_table(ANALYSIS_DIR)
        pred_df, hp_df = run_expanding_backtest(df)
        global_df, sub_df, comp_df, cw_df = evaluate_tables(pred_df)
        pred_df.to_csv(csv_paths["predicciones_oos"], index=False)
        global_df.to_csv(csv_paths["metricas_globales"], index=False)
        sub_df.to_csv(csv_paths["metricas_subperiodos"], index=False)
        comp_df.to_csv(csv_paths["comparacion_estadistica"], index=False)
        cw_df.to_csv(csv_paths["clark_west"], index=False)
        hp_df.to_csv(csv_paths["hiperparametros_hist"], index=False)

        summary = {
            "n_clean_total": int(len(df)),
            "eval_start_idx": EVAL_START_IDX,
            "oos_start_period": str(pred_df["target_period"].iloc[0]),
            "oos_end_period": str(pred_df["target_period"].iloc[-1]),
            "n_oos": int(len(pred_df)),
            "n_pre_congelado": int((pred_df["bloque_eval"] == "pre_congelado_2003m2_2019m11").sum()),
            "n_congelado": int((pred_df["bloque_eval"] == "congelado_2019m12_2024m11").sum()),
            "gbdt_implementation_note": (
                "GBDT_NumPy es una implementacion pura en NumPy de gradient boosted regression trees "
                "porque el paquete C++ xgboost no esta instalado en este entorno."
            ),
            "global_metrics": global_df.to_dict(orient="records"),
            "clark_west": cw_df.to_dict(orient="records"),
        }
        (OUT_DIR / "etapa3_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    figs = make_figures(pred_df, sub_df)
    summary_path = OUT_DIR / "etapa3_summary.json"
    hash_targets = list(csv_paths.values()) + [summary_path] + [
        p for p in figs if p.parent == OUT_DIR
    ]
    chk_rows = [
        {"archivo": p.name, "sha256": u.sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(hash_targets, key=lambda x: x.name)
    ]
    pd.DataFrame(chk_rows).to_csv(OUT_DIR / "etapa3_checksums_sha256.csv", index=False)

    print("\n=== Metricas globales OOS (2003m2-2024m11, n=250) ===")
    print(global_df.drop(columns=["label_es"]).to_string(index=False))
    print("\n=== Bloque congelado (2019m12-2024m11, n=60) ===")
    print(sub_df[sub_df["subperiodo"] == "congelado_2019m12_2024m11"].to_string(index=False))


if __name__ == "__main__":
    main()
