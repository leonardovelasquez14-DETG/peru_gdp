#!/usr/bin/env python3
"""
Utilidades compartidas para validación en tiempo real (Etapa 5) e inferencia (Etapa 8).

Garantiza que todos los modelos (Etapas 3, 4 y 6) utilicen exactamente:
  - la misma matriz de características observables en v0(t),
  - el mismo esquema expansivo causal v_H(s) <= v0(t),
  - las mismas funciones de pérdida, Diebold-Mariano (HLN), Clark-West y
    bootstrap por bloques.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PRIMARY_H = 11
FULL_SECTORS = ("agriculture", "commerce", "construction", "fishing", "manufacturing", "mining")
CORE_5_SECTORS = ("agriculture", "commerce", "construction", "fishing", "manufacturing")
FEATURE_COLS = [
    "y0",
    "dy0_1m",
    "y0_ma3",
    "sec_agriculture",
    "sec_commerce",
    "sec_construction",
    "sec_fishing",
    "sec_manufacturing",
    "sec_mining",
    "Y_lag11",
    "Y_lag12",
    "Y_lag13",
    "Y_obs_mean6",
    "rev_short_1",
    "rev_short_2",
    "rev_short_3",
    "sin1",
    "cos1",
    "sin2",
    "cos2",
]


def sha256_file(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_realtime_feature_table(analysis_dir: Path) -> pd.DataFrame:
    """
    Construye la tabla causal de características observables en el origen v0(t)
    para las 350 observaciones limpias de Y_H11 (1993-10 a 2024-11).
    """
    target_df = pd.read_csv(
        analysis_dir / "muestra_objetivo_gdp.csv",
        parse_dates=["target_date", "v0_date", "v_last_date", "v_H11_date"],
    )
    vlong = pd.read_csv(
        analysis_dir / "vintages_long.csv",
        parse_dates=["vintage_date", "target_date"],
    )

    # Mapa rápido (industry, target_date, vintage_date) -> (value, is_sentinel)
    lookup = vlong.set_index(["industry", "target_date", "vintage_date"])
    val_dict = lookup["value"].to_dict()
    sent_dict = lookup["is_base_year_sentinel"].to_dict()

    t_idx = target_df.set_index("target_date")
    clean = (
        target_df[target_df["estado_H11"] == "incluido_limpio"]
        .sort_values("target_date")
        .reset_index(drop=True)
    )

    rows: List[Dict[str, object]] = []
    for _, row in clean.iterrows():
        dt = row["target_date"]
        v0_t = row["v0_date"]

        # 1. Dinámica de corto plazo del PBI publicada dentro del mismo vintage v0(t)
        v_prev1 = val_dict.get(("gdp", dt - pd.DateOffset(months=1), v0_t), np.nan)
        v_prev2 = val_dict.get(("gdp", dt - pd.DateOffset(months=2), v0_t), np.nan)
        ma3_vals = [x for x in (row["y0"], v_prev1, v_prev2) if pd.notna(x)]

        rec: Dict[str, object] = {
            "target_period": row["target_period"],
            "target_date": dt,
            "v0_date": v0_t,
            "v_H11_date": row["v_H11_date"],
            "y0": float(row["y0"]),
            "y_H11": float(row["y_H11"]),
            "Y_H11": float(row["Y_H11"]),
            "dy0_1m": float(row["y0"] - v_prev1) if pd.notna(v_prev1) else np.nan,
            "y0_ma3": float(np.mean(ma3_vals)) if ma3_vals else float(row["y0"]),
        }

        # 2. Primeras estimaciones sectoriales publicadas en v0(t)
        for sec in FULL_SECTORS:
            rec[f"sec_{sec}"] = val_dict.get((sec, dt, v0_t), np.nan)

        # 3. Rezagos completados de Y_H11 estrictamente observables en v0(t)
        for k in (11, 12, 13):
            dt_k = dt - pd.DateOffset(months=k)
            val_k = np.nan
            if dt_k in t_idx.index:
                rk = t_idx.loc[dt_k]
                if (
                    rk["estado_H11"] == "incluido_limpio"
                    and pd.notna(rk["v_H11_date"])
                    and pd.Timestamp(rk["v_H11_date"]) <= v0_t
                ):
                    val_k = float(rk["Y_H11"])
            rec[f"Y_lag{k}"] = val_k

        # Promedio de las últimas 6 revisiones limpias ya completadas antes o en v0(t)
        completed_past = clean[(clean["v_H11_date"] <= v0_t) & (clean["target_date"] < dt)]
        rec["Y_obs_mean6"] = (
            float(completed_past["Y_H11"].tail(6).mean()) if len(completed_past) > 0 else np.nan
        )

        # 4. Revisiones recientes de corto plazo visibles hasta v0(t) sin cruce de base
        for s in (1, 2, 3):
            dt_s = dt - pd.DateOffset(months=s)
            rev_s = np.nan
            if dt_s in t_idx.index:
                rs = t_idx.loc[dt_s]
                v_now = val_dict.get(("gdp", dt_s, v0_t), np.nan)
                s_now = sent_dict.get(("gdp", dt_s, v0_t), True)
                if (
                    rs["lag0_months"] <= 3
                    and (not bool(rs["y0_sentinel"]))
                    and pd.notna(v_now)
                    and (not bool(s_now))
                ):
                    rev_s = float(v_now - rs["y0"])
            rec[f"rev_short_{s}"] = rev_s

        # 5. Estacionalidad armónica del mes objetivo
        m = dt.month
        rec["month"] = int(m)
        rec["sin1"] = float(np.sin(2.0 * np.pi * m / 12.0))
        rec["cos1"] = float(np.cos(2.0 * np.pi * m / 12.0))
        rec["sin2"] = float(np.sin(4.0 * np.pi * m / 12.0))
        rec["cos2"] = float(np.cos(4.0 * np.pi * m / 12.0))

        rows.append(rec)

    return pd.DataFrame(rows)


def get_causal_train_indices(df: pd.DataFrame, eval_idx: int) -> np.ndarray:
    """Devuelve índices s < eval_idx cuya revisión ya finalizó antes o en v0(eval_idx)."""
    v0_now = df.loc[eval_idx, "v0_date"]
    mask = (df.index < eval_idx) & (df["v_H11_date"] <= v0_now)
    return np.flatnonzero(mask.to_numpy())


def fit_impute_and_scale(
    x_train: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Imputa con mediana de entrenamiento y estandariza usando solo la ventana de entrenamiento."""
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    x_tr = np.where(np.isnan(x_train), medians, x_train)
    x_te = np.where(np.isnan(x_test), medians, x_test)
    means = np.mean(x_tr, axis=0)
    stds = np.std(x_tr, axis=0, ddof=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return (x_tr - means) / stds, (x_te - means) / stds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula RMSE, MAE, sesgo medio, mediana del error absoluto y R2 OOS."""
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    med_ae = float(np.median(np.abs(err)))
    bias = float(np.mean(err))
    denom = float(np.mean((y_true - np.mean(y_true)) ** 2))
    r2_oos = float(1.0 - mse / denom) if denom > 0 else np.nan
    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "med_ae": med_ae,
        "sesgo_medio": bias,
        "r2_oos": r2_oos,
    }


def newey_west_variance(d: np.ndarray, maxlags: int) -> float:
    """Varianza de largo plazo Newey-West con kernel de Bartlett."""
    d_c = d - np.mean(d)
    n = len(d_c)
    gamma0 = float(np.dot(d_c, d_c) / n)
    var = gamma0
    for lag in range(1, min(maxlags, n - 1) + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        cov = float(np.dot(d_c[lag:], d_c[:-lag]) / n)
        var += 2.0 * weight * cov
    return max(var, 1e-12)


def diebold_mariano_test(
    e_model: np.ndarray, e_bench: np.ndarray, loss: str = "sq", horizon: int = PRIMARY_H
) -> Tuple[float, float, float]:
    """
    Prueba Diebold-Mariano con corrección Harvey-Leybourne-Newbold (1997).
    d_t = L(e_model) - L(e_bench): valores negativos indican menor pérdida del modelo.
    Retorna (mean_diff, dm_hln_stat, p_value_two_sided).
    """
    if loss == "sq":
        d = (e_model ** 2) - (e_bench ** 2)
    elif loss == "abs":
        d = np.abs(e_model) - np.abs(e_bench)
    else:
        raise ValueError(f"Tipo de pérdida no soportado: {loss}")

    n = len(d)
    d_bar = float(np.mean(d))
    l_nw = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    l_use = max(l_nw, min(horizon, n // 4))
    lrv = newey_west_variance(d, maxlags=l_use)
    se = np.sqrt(lrv / n)
    dm = d_bar / se if se > 0 else 0.0
    # Ajuste de muestra pequeña Harvey-Leybourne-Newbold para horizonte h
    h_eff = max(1, min(horizon, n // 4))
    hln_factor = np.sqrt((n + 1.0 - 2.0 * h_eff + h_eff * (h_eff - 1.0) / n) / n)
    dm_hln = float(dm * hln_factor)
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1)))
    return d_bar, dm_hln, p_val


def clark_west_test(
    e_small: np.ndarray, yhat_small: np.ndarray, e_large: np.ndarray, yhat_large: np.ndarray, horizon: int = PRIMARY_H
) -> Tuple[float, float]:
    """
    Prueba de Clark-West (2007) para modelos anidados (H0: igual MSPE bajo parsimonioso).
    Retorna (cw_stat, p_value_one_sided_large_better).
    """
    f_t = (e_small ** 2) - ((e_large ** 2) - ((yhat_small - yhat_large) ** 2))
    n = len(f_t)
    l_use = max(1, min(horizon, n // 4))
    lrv = newey_west_variance(f_t, maxlags=l_use)
    se = np.sqrt(lrv / n)
    cw_stat = float(np.mean(f_t) / se) if se > 0 else 0.0
    p_val = float(1.0 - stats.norm.cdf(cw_stat))
    return cw_stat, p_val


def block_bootstrap_rmse_diff(
    e_model: np.ndarray,
    e_bench: np.ndarray,
    block_len: int = 12,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap circular por bloques para IC 95% de (RMSE_model - RMSE_bench).
    Valores negativos indican menor RMSE del modelo.
    """
    rng = np.random.default_rng(seed)
    n = len(e_model)
    sq_m = e_model ** 2
    sq_b = e_bench ** 2
    obs_diff = float(np.sqrt(np.mean(sq_m)) - np.sqrt(np.mean(sq_b)))

    n_blocks = int(np.ceil(n / block_len))
    starts = np.arange(n)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        idx = np.concatenate([(s + np.arange(block_len)) % n for s in chosen])[:n]
        diffs[b] = np.sqrt(np.mean(sq_m[idx])) - np.sqrt(np.mean(sq_b[idx]))

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return obs_diff, float(ci_low), float(ci_high)
