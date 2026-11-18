#!/usr/bin/env python3
"""
Etapa 6 (+ Anexo H potencia): Experimento sintetico calibrado con incertidumbre bootstrap.

Calibra un DGP macroeconomico de revisiones (ec. r_t = mu + phi*r_{t-12} + gamma'f_t + eps_t)
mediante bootstrap circular por bloques (L=12, B_cal=300) sobre la muestra limpia del PBI
peruano (n=350, H=11), propagando la incertidumbre parametrica hacia 12 escenarios
preespecificados a lo largo de:
  - tamano muestral T in {350, 700, 1400},
  - persistencia phi in {0.10, calibrada, 0.65},
  - razon senal-ruido y colas pesadas vs. gaussiano,
  - no linealidad acotada con memoria y choques extremos,
  - dimensionalidad efectiva d_eff in {2, 3, 8}.

Evalua causalmente sobre las ultimas N_OOS = 80 observaciones (sin fuga futura) los modelos:
BiasCorr_Mean, Ridge, RFF_20, ESN_20 y QRC_Ising5. Calcula mejora relativa Delta(theta),
error Monte Carlo, region favorable A con delta_min = 2.0%, y analisis de potencia MDE.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import etapa4_reservorios_qrc as e4  # noqa: E402
import rtd_eval_utils as u  # noqa: E402

ANALYSIS_DIR = ROOT_DIR / "output" / "analysis_output"
ETAPA4_DIR = ROOT_DIR / "output" / "etapa4_reservorios"
OUT_DIR = ROOT_DIR / "output" / "etapa6_sintetico"
TEX_IMG_DIR = ROOT_DIR / "main_tex" / "images"

N_REPS = 20
N_OOS_SYN = 80
P_OBS = 12
RIDGE_GRID_SYN = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0)
DELTA_MIN_PCT = 2.0
RES_DIM = 20
ALPHA_FIXED = 0.60
LEAK_FIXED = 0.50


@dataclass(frozen=True)
class ScenarioSpec:
    code: str
    eje: str
    descripcion: str
    t_total: int
    d_eff: int
    phi_override: float | None
    snr_mult: float
    heavy_tails: bool
    shock_prob: float
    nonlinear_mode: bool


SCENARIOS: Tuple[ScenarioSpec, ...] = (
    ScenarioSpec("S1_Calibrado_T350", "Tamano muestral", "Calibrado Peru (t5+ARCH, T=350)", 350, 3, None, 1.0, True, 0.0, False),
    ScenarioSpec("S2_MuestraMed_T700", "Tamano muestral", "Calibrado Peru (T=700)", 700, 3, None, 1.0, True, 0.0, False),
    ScenarioSpec("S3_MuestraLarga_T1400", "Tamano muestral", "Calibrado Peru (T=1400)", 1400, 3, None, 1.0, True, 0.0, False),
    ScenarioSpec("S4_PersBaja_T350", "Persistencia", "Persistencia baja (phi=0,10, T=350)", 350, 3, 0.10, 1.0, True, 0.0, False),
    ScenarioSpec("S5_PersAlta_T350", "Persistencia", "Persistencia alta (phi=0,65, T=350)", 350, 3, 0.65, 1.0, True, 0.0, False),
    ScenarioSpec("S6_SNRBaja_T350", "Senal-ruido", "SNR baja (0,5x gamma, T=350)", 350, 3, None, 0.5, True, 0.0, False),
    ScenarioSpec("S7_SNRAlta_T350", "Senal-ruido", "SNR alta (2,5x gamma, T=350)", 350, 3, None, 2.5, True, 0.0, False),
    ScenarioSpec("S8_Gaussiano_T350", "Innovaciones", "Gaussiano puro sin ARCH (T=350)", 350, 3, None, 1.0, False, 0.0, False),
    ScenarioSpec("S9_NoLineal_T350", "No linealidad + choques", "No lineal saturante + choques (T=350)", 350, 3, None, 1.8, True, 0.025, True),
    ScenarioSpec("S10_NoLineal_T1000", "No linealidad + choques", "No lineal saturante + choques (T=1000)", 1000, 3, None, 1.8, True, 0.025, True),
    ScenarioSpec("S11_DimBaja_d2", "Dimensionalidad", "Baja dim. efectiva (d_eff=2, p=12)", 350, 2, None, 1.0, True, 0.0, False),
    ScenarioSpec("S12_DimAlta_d8", "Dimensionalidad", "Alta dim. efectiva (d_eff=8, p=12)", 350, 8, None, 1.0, True, 0.0, False),
)


def calibrate_peru_bootstrap_params(n_boot: int = 300, seed: int = 123) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Estima la distribucion bootstrap por bloques circulares (L=12) de los parametros
    empiricos del PBI peruano para propagar incertidumbre de calibracion.
    """
    df = u.build_realtime_feature_table(ANALYSIS_DIR)
    y = df["Y_H11"].to_numpy(dtype=float)
    cols = ["y0", "dy0_1m", "sec_agriculture", "sec_commerce", "sec_construction", "sec_fishing", "sec_manufacturing"]
    x_raw = df[cols].to_numpy(dtype=float)
    y_lag12 = df["Y_lag12"].to_numpy(dtype=float)
    mask = np.isfinite(y_lag12) & np.all(np.isfinite(x_raw), axis=1)

    xc = x_raw[mask]
    xc_sc = (xc - xc.mean(axis=0)) / xc.std(axis=0)
    _, _, vt = np.linalg.svd(xc_sc, full_matrices=False)
    f_lat = xc_sc @ vt[:3].T
    f_sc = (f_lat - f_lat.mean(axis=0)) / f_lat.std(axis=0)
    yc = y[mask]
    lagc = y_lag12[mask]

    n = len(yc)
    block_len = 12
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(seed)

    boot_rows: List[Dict[str, float]] = []
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block_len)) % n for s in starts])[:n]
        yb = yc[idx]
        design = np.column_stack([np.ones(n), lagc[idx], f_sc[idx]])
        beta, _, _, _ = np.linalg.lstsq(design, yb, rcond=None)
        res = yb - design @ beta
        boot_rows.append(
            {
                "rep_boot": float(b),
                "mu": float(beta[0]),
                "phi12": float(np.clip(beta[1], 0.02, 0.75)),
                "gamma_norm": float(np.linalg.norm(beta[2:])),
                "sigma_eps": float(np.std(res, ddof=design.shape[1])),
            }
        )

    boot_df = pd.DataFrame(boot_rows)
    point_design = np.column_stack([np.ones(n), lagc, f_sc])
    beta_pt, _, _, _ = np.linalg.lstsq(point_design, yc, rcond=None)
    res_pt = yc - point_design @ beta_pt
    point_summary = {
        "n_calibration": int(n),
        "mu_point": float(beta_pt[0]),
        "phi12_point": float(beta_pt[1]),
        "gamma_norm_point": float(np.linalg.norm(beta_pt[2:])),
        "sigma_eps_point": float(np.std(res_pt, ddof=point_design.shape[1])),
        "mu_boot_sd": float(boot_df["mu"].std()),
        "phi12_boot_sd": float(boot_df["phi12"].std()),
        "gamma_norm_boot_sd": float(boot_df["gamma_norm"].std()),
        "sigma_eps_boot_sd": float(boot_df["sigma_eps"].std()),
    }
    return boot_df, point_summary


def generate_synthetic_series(
    spec: ScenarioSpec, cal_row: Dict[str, float], rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera una realizacion sintetica (X_obs, Y) con burn-in previo, factores latentes AR(1),
    observables colineales en p=12 columnas, colas t_5 + ARCH opcional y no linealidad.
    """
    burn = 60
    t_len = spec.t_total + burn
    d_eff = spec.d_eff

    # Factores latentes AR(1) con posible presencia de choques extremos tipo COVID
    f = np.zeros((t_len, d_eff), dtype=float)
    phi_f = 0.55
    sig_f = np.sqrt(1.0 - phi_f ** 2)
    for t in range(1, t_len):
        f[t] = phi_f * f[t - 1] + sig_f * rng.normal(0.0, 1.0, size=d_eff)
        if spec.shock_prob > 0.0 and rng.uniform() < spec.shock_prob:
            sign = 1.0 if rng.uniform() < 0.5 else -1.0
            f[t, 0] += sign * rng.uniform(5.0, 7.5)

    # Mapa fijo lineal a p=12 indicadores observados + ruido de medicion moderado
    loading = rng.normal(0.0, 1.0, size=(P_OBS, d_eff))
    loading /= np.linalg.norm(loading, axis=1, keepdims=True)
    x_obs = f @ loading.T + 0.35 * rng.normal(0.0, 1.0, size=(t_len, P_OBS))

    # Construir predictores observables del paso t: columnas del momento t
    mu = float(cal_row["mu"])
    phi12 = float(spec.phi_override if spec.phi_override is not None else cal_row["phi12"])
    sigma_base = float(cal_row["sigma_eps"])
    g_norm = float(cal_row["gamma_norm"]) * spec.snr_mult
    g_dir = rng.normal(0.0, 1.0, size=d_eff)
    g_dir = (g_dir / np.linalg.norm(g_dir)) * g_norm

    y = np.zeros(t_len, dtype=float)
    eps = np.zeros(t_len, dtype=float)
    for t in range(12, t_len):
        if spec.heavy_tails:
            # Innovacion t5 escalada a varianza unitaria * efecto ARCH(1) suave
            z_t = rng.standard_t(df=5) / np.sqrt(5.0 / 3.0)
            vol = sigma_base * np.sqrt(0.75 + 0.25 * min(9.0, (eps[t - 1] / sigma_base) ** 2))
            eps[t] = vol * z_t
        else:
            eps[t] = sigma_base * rng.normal(0.0, 1.0)

        if spec.nonlinear_mode:
            f0 = f[t]
            f1 = f[t - 1]
            # Combinacion no lineal acotada + interaccion con memoria
            nl_sig = g_norm * (
                0.70 * np.tanh(1.1 * f0[0])
                + 0.65 * np.cos(1.3 * f0[1]) * np.tanh(f1[0])
                + 0.55 * np.tanh(f0[2] * f1[1])
            )
            y[t] = mu + phi12 * np.tanh(y[t - 12]) + nl_sig + eps[t]
        else:
            y[t] = mu + phi12 * y[t - 12] + float(np.dot(f[t], g_dir)) + eps[t]

    # Anadir rezago observable t-12 para formar matriz observable en t (p=12 + lag12 = 13 cols)
    y_lag12 = np.roll(y, 12)
    x_full = np.column_stack([x_obs, y_lag12])[burn:]
    y_out = y[burn:]
    return x_full, y_out


def causal_expanding_standardize(x: np.ndarray, burn_min: int = 40) -> np.ndarray:
    """
    Estandariza x_t usando EXCLUSIVAMENTE x_0..x_{t-1} para t >= burn_min,
    garantizando cero fuga temporal dentro del paso unico por reservorio.
    """
    n, p = x.shape
    out = np.empty((n, p), dtype=float)
    m0 = np.mean(x[:burn_min], axis=0)
    s0 = np.maximum(np.std(x[:burn_min], axis=0), 1e-6)
    out[:burn_min] = (x[:burn_min] - m0) / s0

    sum_x = np.sum(x[:burn_min], axis=0)
    sum_x2 = np.sum(x[:burn_min] ** 2, axis=0)
    for t in range(burn_min, n):
        mean_t = sum_x / t
        var_t = np.maximum(sum_x2 / t - mean_t ** 2, 1e-8)
        out[t] = (x[t] - mean_t) / np.sqrt(var_t)
        sum_x += x[t]
        sum_x2 += x[t] ** 2
    return out


def select_lambda_causal_fast(
    z_tr: np.ndarray, y_tr: np.ndarray, n_val: int = 30
) -> float:
    """Selecciona penalizacion Ridge sobre las ultimas n_val observaciones causales."""
    n = len(y_tr)
    start_v = max(24, n - n_val)
    sq_errs = np.zeros(len(RIDGE_GRID_SYN), dtype=float)

    for v in range(start_v, n):
        zi, zv = u.fit_impute_and_scale(z_tr[:v], z_tr[[v]])
        yi = y_tr[:v]
        ym = float(np.mean(yi))
        u_mat, s_vals, vt = np.linalg.svd(zi, full_matrices=False)
        uty = u_mat.T @ (yi - ym)
        vx = vt @ zv[0]
        for j, lam in enumerate(RIDGE_GRID_SYN):
            coef = s_vals / (s_vals ** 2 + lam)
            pred_v = ym + float(np.dot(vx, coef * uty))
            sq_errs[j] += (y_tr[v] - pred_v) ** 2

    return float(RIDGE_GRID_SYN[int(np.argmin(sq_errs))])


def eval_expanding_feature_model(
    z_seq: np.ndarray, y: np.ndarray, t_train_init: int
) -> np.ndarray:
    """Evalua regresion Ridge expansiva causal sobre las ultimas N_OOS_SYN filas."""
    preds = np.empty(len(y) - t_train_init, dtype=float)
    lam_opt = 25.0
    for idx, t in enumerate(range(t_train_init, len(y))):
        if idx % 10 == 0:
            lam_opt = select_lambda_causal_fast(z_seq[:t], y[:t], n_val=30)
        z_tr_sc, z_te_sc = u.fit_impute_and_scale(z_seq[:t], z_seq[[t]])
        y_tr = y[:t]
        ym = float(np.mean(y_tr))
        p = z_tr_sc.shape[1]
        beta = np.linalg.solve(z_tr_sc.T @ z_tr_sc + lam_opt * np.eye(p), z_tr_sc.T @ (y_tr - ym))
        preds[idx] = float(ym + np.dot(z_te_sc[0], beta))
    return preds


def run_synthetic_grid(boot_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Corre los 12 escenarios x 20 replicas bootstrap-calibradas."""
    _, diag_masks, x_flip_pairs = e4.build_pauli_observables(e4.N_QUBITS)
    n_feat = P_OBS + 1

    rep_rows: List[Dict[str, object]] = []
    t_start = time.perf_counter()

    for spec in SCENARIOS:
        t_s0 = time.perf_counter()
        t_train_init = spec.t_total - N_OOS_SYN
        for rep in range(N_REPS):
            seed = 1000 + rep * 37
            rng = np.random.default_rng(seed + sum(ord(c) for c in spec.code))
            cal_idx = int(rng.integers(0, len(boot_df)))
            cal_row = boot_df.iloc[cal_idx].to_dict()

            x_raw, y = generate_synthetic_series(spec, cal_row, rng)
            x_causal_sc = causal_expanding_standardize(x_raw, burn_min=40)

            # Pad 13 features a 20 canales para compatibilidad exacta con extract_qrc_features(2*Nq=10)
            u_gate = e4.build_ising_unitary(e4.N_QUBITS, seed=seed)
            w_enc = e4.build_normalized_projection(2 * e4.N_QUBITS, n_feat, seed=seed)
            w_in_e, w_res_e = e4.build_esn_weights(RES_DIM, n_feat, seed=seed)
            w_rff = e4.build_normalized_projection(RES_DIM, n_feat, seed=seed)
            b_rff = np.random.default_rng(seed + 2000).uniform(0.0, 2.0 * np.pi, size=RES_DIM)

            z_qrc, _ = e4.extract_qrc_features(
                x_causal_sc, w_enc, u_gate, diag_masks, x_flip_pairs, ALPHA_FIXED, LEAK_FIXED
            )
            z_esn = e4.extract_esn_features(x_causal_sc, w_in_e, w_res_e, ALPHA_FIXED, LEAK_FIXED)
            z_rff = e4.extract_rff_features(x_causal_sc, w_rff, b_rff, ALPHA_FIXED)

            y_test = y[t_train_init:]
            pred_bias = np.array([float(np.mean(y[:t])) for t in range(t_train_init, len(y))])
            pred_ridge = eval_expanding_feature_model(x_raw, y, t_train_init)
            pred_rff = eval_expanding_feature_model(z_rff, y, t_train_init)
            pred_esn = eval_expanding_feature_model(z_esn, y, t_train_init)
            pred_qrc = eval_expanding_feature_model(z_qrc, y, t_train_init)

            rmses = {
                "BiasCorr_Mean": float(np.sqrt(np.mean((y_test - pred_bias) ** 2))),
                "Ridge": float(np.sqrt(np.mean((y_test - pred_ridge) ** 2))),
                "RFF_20": float(np.sqrt(np.mean((y_test - pred_rff) ** 2))),
                "ESN_20": float(np.sqrt(np.mean((y_test - pred_esn) ** 2))),
                "QRC_Ising5": float(np.sqrt(np.mean((y_test - pred_qrc) ** 2))),
            }
            rep_rows.append(
                {
                    "escenario": spec.code,
                    "eje": spec.eje,
                    "descripcion": spec.descripcion,
                    "T_total": spec.t_total,
                    "n_oos": N_OOS_SYN,
                    "d_eff": spec.d_eff,
                    "rep": rep,
                    "rmse_BiasCorr": rmses["BiasCorr_Mean"],
                    "rmse_Ridge": rmses["Ridge"],
                    "rmse_RFF_20": rmses["RFF_20"],
                    "rmse_ESN_20": rmses["ESN_20"],
                    "rmse_QRC_Ising5": rmses["QRC_Ising5"],
                    "delta_pct_vs_Ridge": 100.0 * (rmses["Ridge"] - rmses["QRC_Ising5"]) / rmses["Ridge"],
                    "delta_pct_vs_ESN20": 100.0 * (rmses["ESN_20"] - rmses["QRC_Ising5"]) / rmses["ESN_20"],
                    "delta_pct_vs_RFF20": 100.0 * (rmses["RFF_20"] - rmses["QRC_Ising5"]) / rmses["RFF_20"],
                }
            )
        print(f"[Etapa 6] {spec.code} ({N_REPS} reps) terminado en {time.perf_counter()-t_s0:.2f}s")

    rep_df = pd.DataFrame(rep_rows)

    # Resumen por escenario con error Monte Carlo e intervalo 95% y region favorable A
    sum_rows: List[Dict[str, object]] = []
    for spec in SCENARIOS:
        sub = rep_df[rep_df["escenario"] == spec.code]
        r_m = float(sub["rmse_Ridge"].mean())
        rff_m = float(sub["rmse_RFF_20"].mean())
        esn_m = float(sub["rmse_ESN_20"].mean())
        qrc_m = float(sub["rmse_QRC_Ising5"].mean())

        def _delta_stats(col: str) -> Tuple[float, float, float, float]:
            vals = sub[col].to_numpy(dtype=float)
            mean_v = float(np.mean(vals))
            se_v = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            t_crit = float(stats.t.ppf(0.975, df=len(vals) - 1))
            return mean_v, se_v, mean_v - t_crit * se_v, mean_v + t_crit * se_v

        d_r, se_r, lo_r, hi_r = _delta_stats("delta_pct_vs_Ridge")
        d_e, se_e, lo_e, hi_e = _delta_stats("delta_pct_vs_ESN20")
        d_f, se_f, lo_f, hi_f = _delta_stats("delta_pct_vs_RFF20")

        sum_rows.append(
            {
                "escenario": spec.code,
                "eje": spec.eje,
                "descripcion": spec.descripcion,
                "T_total": spec.t_total,
                "rmse_BiasCorr": float(sub["rmse_BiasCorr"].mean()),
                "rmse_Ridge": r_m,
                "rmse_RFF_20": rff_m,
                "rmse_ESN_20": esn_m,
                "rmse_QRC_Ising5": qrc_m,
                "delta_vs_Ridge_pct": d_r,
                "mc_se_Ridge_pct": se_r,
                "ci95_lo_Ridge": lo_r,
                "ci95_hi_Ridge": hi_r,
                "favorable_vs_Ridge": bool((d_r >= DELTA_MIN_PCT) and (lo_r > 0.0)),
                "delta_vs_ESN20_pct": d_e,
                "mc_se_ESN20_pct": se_e,
                "ci95_lo_ESN20": lo_e,
                "ci95_hi_ESN20": hi_e,
                "favorable_vs_ESN20": bool((d_e >= DELTA_MIN_PCT) and (lo_e > 0.0)),
                "delta_vs_RFF20_pct": d_f,
                "mc_se_RFF20_pct": se_f,
                "ci95_lo_RFF20": lo_f,
                "ci95_hi_RFF20": hi_f,
                "favorable_vs_RFF20": bool((d_f >= DELTA_MIN_PCT) and (lo_f > 0.0)),
            }
        )
    print(f"[Etapa 6] Grid completo ejecutado en {time.perf_counter()-t_start:.2f}s")
    return rep_df, pd.DataFrame(sum_rows)


def compute_empirical_power_mde() -> pd.DataFrame:
    """
    Anexo H: estima el efecto minimo detectable (MDE al 80% de potencia, alfa=0,05)
    y curvas de potencia empirica mediante bootstrap circular por bloques sobre los
    diferenciales de perdida observados en el PBI peruano (Etapas 3 y 4).
    """
    e4_pred = pd.read_csv(ETAPA4_DIR / "etapa4_predicciones_oos.csv")
    y = e4_pred["Y_H11"].to_numpy(dtype=float)

    pairs = [
        ("QRC_Ising5", "Ridge", "global_n250"),
        ("QRC_Ising5", "BiasCorr_Mean", "global_n250"),
        ("QRC_Ising5", "ESN_20", "global_n250"),
        ("QRC_Ising5", "RFF_20", "global_n250"),
        ("RFF_20", "Ridge", "global_n250"),
        ("QRC_Ising5", "ESN_20", "pre_congelado_n190"),
        ("QRC_Ising5", "Ridge", "pre_congelado_n190"),
        ("QRC_Ising5", "ESN_20", "congelado_n60"),
    ]

    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(42)
    effects_pct = (2.0, 5.0, 8.0, 12.0)

    for m_a, m_b, scope in pairs:
        if scope == "pre_congelado_n190":
            sub = e4_pred[e4_pred["bloque_eval"] == "pre_congelado_2003m2_2019m11"]
        elif scope == "congelado_n60":
            sub = e4_pred[e4_pred["bloque_eval"] == "congelado_2019m12_2024m11"]
        else:
            sub = e4_pred

        ys = sub["Y_H11"].to_numpy(dtype=float)
        ea = ys - sub[f"pred_{m_a}"].to_numpy(dtype=float)
        eb = ys - sub[f"pred_{m_b}"].to_numpy(dtype=float)
        rmse_b = float(np.sqrt(np.mean(eb ** 2)))
        n = len(ys)

        d = (ea ** 2) - (eb ** 2)
        l_nw = max(1, min(u.PRIMARY_H, n // 4))
        lrv = u.newey_west_variance(d, maxlags=l_nw)
        se_d_nw = float(np.sqrt(lrv / n))
        # Factor analitico 80% potencia bilateral alfa=0.05: (1.96 + 0.842) = 2.802
        mde_mse = 2.8016 * se_d_nw
        # Convertir reduccion MSE = mde_mse a reduccion porcentual de RMSE respecto a rmse_b
        if mde_mse < rmse_b ** 2:
            mde_rmse_pct = 100.0 * (1.0 - np.sqrt(max(1e-12, 1.0 - mde_mse / (rmse_b ** 2))))
        else:
            mde_rmse_pct = 100.0

        # Simulacion bootstrap de potencia centrada bajo H1 de mejora relativa delta%
        d_centered = d - np.mean(d)
        block_len = min(12, n)
        n_blocks = int(np.ceil(n / block_len))
        b_sim = 1000
        pow_dict: Dict[float, float] = {}
        for eff in effects_pct:
            target_mse_gain = (rmse_b ** 2) * (1.0 - (1.0 - eff / 100.0) ** 2)
            rejects = 0
            for _ in range(b_sim):
                starts = rng.integers(0, n, size=n_blocks)
                idx = np.concatenate([(s + np.arange(block_len)) % n for s in starts])[:n]
                db = d_centered[idx] - target_mse_gain
                se_b = np.sqrt(u.newey_west_variance(db, maxlags=l_nw) / n)
                t_b = np.mean(db) / se_b if se_b > 0 else 0.0
                if abs(t_b) > 1.96:
                    rejects += 1
            pow_dict[eff] = float(rejects / b_sim)

        rows.append(
            {
                "alcance": scope,
                "n": int(n),
                "modelo_a": m_a,
                "modelo_b": m_b,
                "rmse_b": rmse_b,
                "se_nw_mse_diff": se_d_nw,
                "mde_80_rmse_pct": float(mde_rmse_pct),
                "power_2pct": pow_dict[2.0],
                "power_5pct": pow_dict[5.0],
                "power_8pct": pow_dict[8.0],
                "power_12pct": pow_dict[12.0],
            }
        )
    return pd.DataFrame(rows)


def make_figures(sum_df: pd.DataFrame, mde_df: pd.DataFrame) -> List[Path]:
    """Genera graficos de region favorable Delta(theta) y analisis de potencia MDE."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.8,
            "axes.labelsize": 9.8,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    TEX_IMG_DIR.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    # Figura 1: Forest plot de mejora relativa Delta(theta) con IC 95% MC
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    y_pos = np.arange(len(sum_df))[::-1]
    eje_map = {"Tamano muestral": "Tamaño muestral", "Senal-ruido": "Señal-ruido"}
    labels = [
        f"{r['escenario'].split('_', 1)[1]} ({eje_map.get(r['eje'], r['eje'])})"
        for _, r in sum_df.iterrows()
    ]

    ax.axvspan(DELTA_MIN_PCT, max(35.0, float(sum_df["ci95_hi_Ridge"].max()) + 4.0), color="#E8F5E9", alpha=0.65,
               label=rf"Región favorable ($\Delta \geq +{DELTA_MIN_PCT:.0f}\%$)")
    ax.axvline(0.0, color="#444444", lw=1.0, ls="--", label=r"Empate ($\Delta = 0\%$)")
    ax.axvline(DELTA_MIN_PCT, color="#2E8B57", lw=1.1, ls=":")

    offsets = (0.16, 0.0, -0.16)
    series = (
        ("vs. Ridge (lineal)", "delta_vs_Ridge_pct", "ci95_lo_Ridge", "ci95_hi_Ridge", "#C84B31", "o"),
        ("vs. ESN_20 (reservorio clásico)", "delta_vs_ESN20_pct", "ci95_lo_ESN20", "ci95_hi_ESN20", "#1B365D", "s"),
        ("vs. RFF_20 (acotado sin memoria)", "delta_vs_RFF20_pct", "ci95_lo_RFF20", "ci95_hi_RFF20", "#7B2CBF", "^"),
    )
    for idx_s, (lbl, col_m, col_lo, col_hi, col, mk) in enumerate(series):
        means = sum_df[col_m].to_numpy()
        lo = means - sum_df[col_lo].to_numpy()
        hi = sum_df[col_hi].to_numpy() - means
        ax.errorbar(
            means,
            y_pos + offsets[idx_s],
            xerr=[np.maximum(0.0, lo), np.maximum(0.0, hi)],
            fmt=mk,
            color=col,
            ecolor=col,
            elinewidth=1.4,
            capsize=2.8,
            markersize=5.2,
            label=lbl,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Mejora relativa de QRC $\Delta(\theta) = 100\,(\mathrm{RMSE}_B - \mathrm{RMSE}_{\mathrm{QRC}})/\mathrm{RMSE}_B$ (%)")
    ax.set_title("Experimento sintético calibrado: desempeño relativo de QRC e intervalos 95% Monte Carlo", pad=10)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94)
    fig.tight_layout()

    for d in (OUT_DIR, TEX_IMG_DIR):
        p = d / "etapa6_delta_escenarios.png"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        generated.append(p)
    plt.close(fig)

    # Figura 2: Potencia estadistica empirica y MDE en la muestra peruana (Anexo H)
    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.1, 1.0]})
    g_mde = mde_df[mde_df["alcance"] == "global_n250"].copy()
    x_idx = np.arange(len(g_mde))
    pair_lbls = [f"{r['modelo_a']}\nvs. {r['modelo_b']}" for _, r in g_mde.iterrows()]
    bars = ax_l.bar(x_idx, g_mde["mde_80_rmse_pct"], color=["#C84B31", "#6C757D", "#1B365D", "#7B2CBF", "#2E8B57"], width=0.56)
    ax_l.axhline(DELTA_MIN_PCT, color="#2E8B57", ls="--", lw=1.2, label=rf"Umbral $\delta_{{\min}} = {DELTA_MIN_PCT:.1f}\%$".replace(".", ","))
    for b in bars:
        h = b.get_height()
        ax_l.text(b.get_x() + b.get_width() / 2.0, h + 0.35, f"{h:.1f}%".replace(".", ","), ha="center", va="bottom", fontsize=8.5)
    ax_l.set_xticks(x_idx)
    ax_l.set_xticklabels(pair_lbls, fontsize=8.3)
    ax_l.set_ylim(0, float(g_mde["mde_80_rmse_pct"].max()) * 1.22)
    ax_l.set_ylabel("MDE en reducción de RMSE (%)")
    ax_l.set_title("Efecto mínimo detectable (80% potencia, n=250)", pad=8)
    ax_l.grid(axis="y", alpha=0.25)
    ax_l.legend(loc="upper center", frameon=True, facecolor="white")

    # Panel derecho: curvas de potencia para contrastes globales clave
    effs = [2.0, 5.0, 8.0, 12.0]
    styles = [
        ("QRC_Ising5", "Ridge", "QRC vs. Ridge", "#C84B31", "-"),
        ("QRC_Ising5", "BiasCorr_Mean", "QRC vs. Media exp.", "#6C757D", "--"),
        ("QRC_Ising5", "ESN_20", "QRC vs. ESN_20", "#1B365D", "-."),
        ("QRC_Ising5", "RFF_20", "QRC vs. RFF_20", "#7B2CBF", "-"),
    ]
    for ma, mb, lbl, col, ls in styles:
        row = g_mde[(g_mde["modelo_a"] == ma) & (g_mde["modelo_b"] == mb)].iloc[0]
        pows = [100.0 * float(row[f"power_{int(e)}pct"]) for e in effs]
        ax_r.plot(effs, pows, marker="o", lw=1.6, ls=ls, color=col, label=lbl)
    ax_r.axhline(80.0, color="#222222", ls=":", lw=1.1, label="Potencia objetivo 80%")
    ax_r.set_xlabel("Reducción verdadera simulada de RMSE (%)")
    ax_r.set_ylabel("Potencia estadística (%)")
    ax_r.set_ylim(0, 108)
    ax_r.set_title("Curvas de potencia bootstrap en muestra real (n=250)", pad=8)
    ax_r.grid(alpha=0.25)
    ax_r.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.94, fontsize=7.8)

    fig2.tight_layout()
    for d in (OUT_DIR, TEX_IMG_DIR):
        p = d / "etapa6_potencia_mde.png"
        fig2.savefig(p, dpi=220, bbox_inches="tight")
        generated.append(p)
    plt.close(fig2)
    return generated


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_only = "--plot-only" in sys.argv[1:]
    csv_paths = {
        "calibracion_boot": OUT_DIR / "etapa6_calibracion_bootstrap.csv",
        "replicas_mc": OUT_DIR / "etapa6_replicas_montecarlo.csv",
        "resumen_escenarios": OUT_DIR / "etapa6_resumen_escenarios.csv",
        "potencia_mde": OUT_DIR / "etapa6_potencia_mde.csv",
    }
    summary_path = OUT_DIR / "etapa6_summary.json"

    if plot_only:
        sum_df = pd.read_csv(csv_paths["resumen_escenarios"])
        mde_df = pd.read_csv(csv_paths["potencia_mde"])
    else:
        boot_df, cal_summary = calibrate_peru_bootstrap_params(n_boot=300, seed=123)
        rep_df, sum_df = run_synthetic_grid(boot_df)
        mde_df = compute_empirical_power_mde()

        boot_df.to_csv(csv_paths["calibracion_boot"], index=False)
        rep_df.to_csv(csv_paths["replicas_mc"], index=False)
        sum_df.to_csv(csv_paths["resumen_escenarios"], index=False)
        mde_df.to_csv(csv_paths["potencia_mde"], index=False)

        payload = {
            "calibration_summary": cal_summary,
            "delta_min_pct": DELTA_MIN_PCT,
            "n_reps_per_scenario": N_REPS,
            "n_oos_per_rep": N_OOS_SYN,
            "resumen_escenarios": sum_df.to_dict(orient="records"),
            "potencia_mde": mde_df.to_dict(orient="records"),
        }
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    figs = make_figures(sum_df, mde_df)
    hash_targets = list(csv_paths.values()) + [summary_path] + [
        p for p in figs if p.parent == OUT_DIR
    ]
    chk_rows = [
        {"archivo": p.name, "sha256": u.sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(hash_targets, key=lambda x: x.name)
    ]
    pd.DataFrame(chk_rows).to_csv(OUT_DIR / "etapa6_checksums_sha256.csv", index=False)

    print("\n=== Resumen Etapa 6: Delta_QRC (%) por escenario ===")
    cols_show = [
        "escenario", "T_total", "rmse_Ridge", "rmse_RFF_20", "rmse_ESN_20", "rmse_QRC_Ising5",
        "delta_vs_Ridge_pct", "favorable_vs_Ridge", "delta_vs_ESN20_pct", "delta_vs_RFF20_pct",
    ]
    print(sum_df[cols_show].to_string(index=False))
    print("\n=== Anexo H: MDE y Potencia Empirica ===")
    print(mde_df.to_string(index=False))


if __name__ == "__main__":
    main()
