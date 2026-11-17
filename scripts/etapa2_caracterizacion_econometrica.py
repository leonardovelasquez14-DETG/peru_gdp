#!/usr/bin/env python3
"""
Etapa 2: Caracterización econométrica de las revisiones del PBI peruano.

Lee los artefactos auditados de la Etapa 1 en output/analysis_output/:
  - muestra_objetivo_gdp.csv
  - vintages_long.csv

Produce en output/etapa2_econometria/:
  - etapa2_descriptivos_horizontes.csv
  - etapa2_acf_pacf_y11.csv
  - etapa2_mincer_zarnowitz.csv
  - etapa2_regimenes_y11.csv
  - etapa2_estacionalidad_y11.csv
  - etapa2_asociacion_tiempo_real.csv
  - etapa2_sectores_h11.csv
  - etapa2_checksums_sha256.csv
  - etapa2_summary.json
  - figures/etapa2_acf_pacf.png (+ copia a main_tex/images/)
  - figures/etapa2_regimenes_evolucion.png (+ copia a main_tex/images/)
  - figures/etapa2_sectores_sesgo.png (+ copia a main_tex/images/)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, linear_reset
from statsmodels.tsa.stattools import pacf

PIPELINE_VERSION = "1.0.0"
PRIMARY_H = 11
ROOT_DIR = Path(__file__).resolve().parent.parent
E1_DIR = ROOT_DIR / "output" / "analysis_output"
OUT_DIR = ROOT_DIR / "output" / "etapa2_econometria"
FIG_OUT_DIR = OUT_DIR / "figures"
TEX_IMG_DIR = ROOT_DIR / "main_tex" / "images"


def sha256_file(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def newey_west_lags(n_obs: int) -> int:
    """Regla automática de Newey-West / Andrews: floor(4 * (T/100)^(2/9))."""
    return max(1, int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


def fit_mean_hac(values: np.ndarray, maxlags: int) -> Tuple[float, float, float]:
    """Estima la media y su error estándar y p-valor HAC (Newey-West)."""
    x = np.ones((len(values), 1))
    res = sm.OLS(values, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.params[0]), float(res.bse[0]), float(res.pvalues[0])


def get_clean_sample(target_df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Extrae la submuestra limpia (con v0 y sin cruce de año base) para horizonte h."""
    if f"estado_H{h}" in target_df.columns:
        mask = target_df[f"estado_H{h}"] == "incluido_limpio"
    else:
        mask = (
            (target_df["lag0_months"] <= 3)
            & target_df[f"Y_H{h}"].notna()
            & (~target_df[f"crosses_base_year_H{h}"])
        )
    return target_df.loc[mask].sort_values("target_date").copy()


def build_horizon_diagnostics(target_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula momentos, pruebas de normalidad y pruebas HAC de sesgo para H=3..12."""
    rows: List[Dict[str, object]] = []
    for h in (3, 6, 9, 10, 11, 12):
        for kind in ("limpia", "bruta"):
            if kind == "limpia":
                sub = get_clean_sample(target_df, h)
            else:
                sub = target_df.loc[
                    (target_df["lag0_months"] <= 3) & target_df[f"Y_H{h}"].notna()
                ].sort_values("target_date")
            y = sub[f"Y_H{h}"].to_numpy(dtype=float)
            n = len(y)
            l_nw = newey_west_lags(n)
            mean_nw, se_nw, p_nw = fit_mean_hac(y, l_nw)
            _, se_ov, p_ov = fit_mean_hac(y, h)
            skew = float(stats.skew(y, bias=False))
            ex_kurt = float(stats.kurtosis(y, fisher=True, bias=False))
            jb_stat, jb_p = stats.jarque_bera(y)
            sw_stat, sw_p = stats.shapiro(y)
            rows.append(
                {
                    "horizonte_H": h,
                    "muestra": kind,
                    "n": n,
                    "L_nw": l_nw,
                    "L_solapamiento": h,
                    "media": round(mean_nw, 4),
                    "se_nw": round(se_nw, 4),
                    "p_nw": float(p_nw),
                    "se_solap": round(se_ov, 4),
                    "p_solap": float(p_ov),
                    "de": round(float(np.std(y, ddof=1)), 4),
                    "mediana": round(float(np.median(y)), 4),
                    "p25": round(float(np.percentile(y, 25)), 4),
                    "p75": round(float(np.percentile(y, 75)), 4),
                    "min": round(float(np.min(y)), 4),
                    "max": round(float(np.max(y)), 4),
                    "asimetria": round(skew, 4),
                    "curtosis_exceso": round(ex_kurt, 4),
                    "jb_stat": round(float(jb_stat), 3),
                    "jb_p": float(jb_p),
                    "shapiro_w": round(float(sw_stat), 4),
                    "shapiro_p": float(sw_p),
                }
            )
    return pd.DataFrame(rows)


def build_acf_pacf(target_df: pd.DataFrame) -> pd.DataFrame:
    """Autocorrelaciones alineadas por calendario, PACF y Ljung-Box para Y_H11."""
    s_cal = target_df.set_index("target_date")["Y_H11"].where(
        target_df.set_index("target_date")["estado_H11"] == "incluido_limpio"
    )
    clean_seq = target_df.loc[target_df["estado_H11"] == "incluido_limpio", "Y_H11"].to_numpy(dtype=float)
    lb = acorr_ljungbox(clean_seq, lags=list(range(1, 19)), return_df=True)
    pacf_vals = pacf(clean_seq, nlags=18, method="ywm")

    rows: List[Dict[str, object]] = []
    for k in range(1, 19):
        pair = pd.concat([s_cal, s_cal.shift(k)], axis=1).dropna()
        r_k, p_k = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        n_k = len(pair)
        rows.append(
            {
                "rezago_k": k,
                "n_pares_calendario": int(n_k),
                "acf_calendario": round(float(r_k), 4),
                "acf_p": float(p_k),
                "ic95_banda": round(1.96 / np.sqrt(n_k), 4),
                "pacf_secuencial": round(float(pacf_vals[k]), 4),
                "ljung_box_q": round(float(lb.loc[k, "lb_stat"]), 3),
                "ljung_box_p": float(lb.loc[k, "lb_pvalue"]),
                "observable_en_v0": bool(k >= PRIMARY_H),
            }
        )
    return pd.DataFrame(rows)


def fit_mincer_zarnowitz_block(sub: pd.DataFrame, h: int, label: str) -> Dict[str, object]:
    """Estima regresión Mincer-Zarnowitz con errores HAC, Wald, ARCH y RESET."""
    y0 = sub["y0"].to_numpy(dtype=float)
    y_h = sub[f"y_H{h}"].to_numpy(dtype=float)
    n = len(y_h)
    l_nw = newey_west_lags(n)
    x_mat = sm.add_constant(y0)
    ols_plain = sm.OLS(y_h, x_mat).fit()
    mz_nw = sm.OLS(y_h, x_mat).fit(cov_type="HAC", cov_kwds={"maxlags": l_nw})
    mz_ov = sm.OLS(y_h, x_mat).fit(cov_type="HAC", cov_kwds={"maxlags": h})

    alpha = float(mz_nw.params[0])
    beta = float(mz_nw.params[1])
    se_a_nw, se_b_nw = float(mz_nw.bse[0]), float(mz_nw.bse[1])
    se_a_ov, se_b_ov = float(mz_ov.bse[0]), float(mz_ov.bse[1])

    p_b1_nw = float(2.0 * (1.0 - stats.t.cdf(abs((beta - 1.0) / se_b_nw), df=n - 2)))
    p_b1_ov = float(2.0 * (1.0 - stats.t.cdf(abs((beta - 1.0) / se_b_ov), df=n - 2)))

    r_mat = (np.eye(2), np.array([0.0, 1.0]))
    wald_nw = mz_nw.wald_test(r_mat, use_f=True, scalar=True)
    wald_ov = mz_ov.wald_test(r_mat, use_f=True, scalar=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        arch1 = het_arch(ols_plain.resid, nlags=1)
        arch4 = het_arch(ols_plain.resid, nlags=4)
    reset_res = linear_reset(ols_plain, power=3, use_f=True)

    return {
        "especificacion": label,
        "horizonte_H": h,
        "n": n,
        "L_nw": l_nw,
        "alpha": round(alpha, 4),
        "se_alpha_nw": round(se_a_nw, 4),
        "p_alpha_nw": float(mz_nw.pvalues[0]),
        "se_alpha_ov": round(se_a_ov, 4),
        "beta": round(beta, 4),
        "se_beta_nw": round(se_b_nw, 4),
        "p_beta1_nw": p_b1_nw,
        "se_beta_ov": round(se_b_ov, 4),
        "p_beta1_ov": p_b1_ov,
        "wald_f_nw": round(float(wald_nw.fvalue), 3),
        "wald_p_nw": float(wald_nw.pvalue),
        "wald_f_ov": round(float(wald_ov.fvalue), 3),
        "wald_p_ov": float(wald_ov.pvalue),
        "r2": round(float(ols_plain.rsquared), 4),
        "arch1_p": float(arch1[1]),
        "arch4_p": float(arch4[1]),
        "reset_f": round(float(reset_res.fvalue), 3),
        "reset_p": float(reset_res.pvalue),
    }


def build_mincer_zarnowitz_table(target_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta Mincer-Zarnowitz por horizonte y por subrégimen de año base / COVID."""
    mz_rows: List[Dict[str, object]] = []
    for h in (11, 12, 6, 3):
        sub = get_clean_sample(target_df, h)
        mz_rows.append(fit_mincer_zarnowitz_block(sub, h, f"Muestra limpia H={h}"))

    sub11 = get_clean_sample(target_df, PRIMARY_H)
    regimes = [
        ("1993-10 a 1999-04 (Base 1990/94)", sub11["target_date"] <= "1999-04-01"),
        ("2000-06 a 2012-12 (Base 1994)", (sub11["target_date"] >= "2000-06-01") & (sub11["target_date"] <= "2012-12-01")),
        ("2014-02 a 2019-12 (Base 2007 pre-COVID)", (sub11["target_date"] >= "2014-02-01") & (sub11["target_date"] <= "2019-12-01")),
        ("2020-01 a 2024-11 (COVID y post-COVID)", sub11["target_date"] >= "2020-01-01"),
    ]
    reg_rows: List[Dict[str, object]] = []
    for name, mask in regimes:
        rsub = sub11.loc[mask].copy()
        y = rsub["Y_H11"].to_numpy(dtype=float)
        l_nw = newey_west_lags(len(y))
        mean_r, se_r, p_r = fit_mean_hac(y, l_nw)
        mz_info = fit_mincer_zarnowitz_block(rsub, PRIMARY_H, name)
        mz_rows.append(mz_info)
        reg_rows.append(
            {
                "regimen": name,
                "n": len(y),
                "L_nw": l_nw,
                "media": round(mean_r, 4),
                "se_nw": round(se_r, 4),
                "p_nw": float(p_r),
                "de": round(float(np.std(y, ddof=1)), 4),
                "mediana": round(float(np.median(y)), 4),
                "min": round(float(np.min(y)), 4),
                "max": round(float(np.max(y)), 4),
                "mz_beta": mz_info["beta"],
                "wald_p_nw": mz_info["wald_p_nw"],
                "reset_p": mz_info["reset_p"],
            }
        )

    return pd.DataFrame(mz_rows), pd.DataFrame(reg_rows)


def build_seasonality_and_rt_models(target_df: pd.DataFrame, vlong: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estacionalidad calendario y regresión diagnóstica con información observable en v0."""
    sub11 = get_clean_sample(target_df, PRIMARY_H)
    month_names = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
    }
    seas_rows: List[Dict[str, object]] = []
    for m_num in range(1, 13):
        vals = sub11.loc[sub11["target_date"].dt.month == m_num, "Y_H11"].to_numpy(dtype=float)
        mean_m, se_m, p_m = fit_mean_hac(vals, maxlags=2)
        seas_rows.append(
            {
                "mes": m_num,
                "nombre_mes": month_names[m_num],
                "n": len(vals),
                "media": round(mean_m, 4),
                "se_nw": round(se_m, 4),
                "p_nw": float(p_m),
                "de": round(float(np.std(vals, ddof=1)), 4),
                "mediana": round(float(np.median(vals)), 4),
            }
        )
    seas_df = pd.DataFrame(seas_rows)

    gdp_v = vlong[vlong["industry"] == "gdp"].set_index(["target_date", "vintage_date"])
    val_map = gdp_v["value"].to_dict()
    sent_map = gdp_v["is_base_year_sentinel"].to_dict()
    t_idx = target_df.set_index("target_date")

    rt_rows: List[Dict[str, object]] = []
    for dt, r in t_idx.iterrows():
        if r["estado_H11"] != "incluido_limpio":
            continue
        v0_t = r["v0_date"]
        rec: Dict[str, object] = {
            "target_date": dt,
            "Y_H11": r["Y_H11"],
            "y0": r["y0"],
            "month": dt.month,
        }
        for k in (11, 12):
            dt_k = dt - pd.DateOffset(months=k)
            if dt_k in t_idx.index:
                rk = t_idx.loc[dt_k]
                if (
                    rk["estado_H11"] == "incluido_limpio"
                    and pd.notna(rk["v_H11_date"])
                    and pd.Timestamp(rk["v_H11_date"]) <= v0_t
                ):
                    rec[f"Y_lag{k}"] = rk["Y_H11"]
        for s in (1, 2):
            dt_s = dt - pd.DateOffset(months=s)
            if dt_s in t_idx.index:
                rs = t_idx.loc[dt_s]
                v_now = val_map.get((dt_s, v0_t), np.nan)
                s_now = sent_map.get((dt_s, v0_t), True)
                if rs["lag0_months"] <= 3 and (not rs["y0_sentinel"]) and pd.notna(v_now) and (not s_now):
                    rec[f"rev_short_{s}"] = float(v_now - rs["y0"])
        rt_rows.append(rec)

    df_rt = pd.DataFrame(rt_rows).dropna().reset_index(drop=True)
    l_nw = newey_west_lags(len(df_rt))
    dummies = pd.get_dummies(df_rt["month"], prefix="mes", drop_first=True, dtype=float)

    specs: List[Tuple[str, pd.DataFrame]] = [
        ("Solo primera publicacion (y0)", sm.add_constant(df_rt[["y0"]])),
        ("Rezagos observables (Y_{t-11}, Y_{t-12})", sm.add_constant(df_rt[["Y_lag11", "Y_lag12"]])),
        ("Revisiones recientes (s=1, 2)", sm.add_constant(df_rt[["rev_short_1", "rev_short_2"]])),
        ("Estacionalidad mensual (11 dummies)", sm.add_constant(dummies)),
        (
            "Modelo conjunto observable en v0",
            sm.add_constant(pd.concat([df_rt[["y0", "Y_lag11", "Y_lag12", "rev_short_1", "rev_short_2"]], dummies], axis=1)),
        ),
    ]

    rt_results: List[Dict[str, object]] = []
    for name, x_df in specs:
        fit = sm.OLS(df_rt["Y_H11"], x_df).fit(cov_type="HAC", cov_kwds={"maxlags": l_nw})
        r_joint = np.eye(len(fit.params))[1:]
        wald = fit.wald_test(r_joint, use_f=True, scalar=True)
        rt_results.append(
            {
                "modelo": name,
                "n": len(df_rt),
                "num_regresores": len(fit.params) - 1,
                "r2": round(float(fit.rsquared), 4),
                "r2_ajustado": round(float(fit.rsquared_adj), 4),
                "wald_f_nw": round(float(wald.fvalue), 3),
                "wald_p_nw": float(wald.pvalue),
                "aic": round(float(fit.aic), 2),
                "bic": round(float(fit.bic), 2),
                "coef_y0": round(float(fit.params["y0"]), 4) if "y0" in fit.params else np.nan,
                "p_y0": float(fit.pvalues["y0"]) if "y0" in fit.pvalues else np.nan,
                "coef_lag12": round(float(fit.params["Y_lag12"]), 4) if "Y_lag12" in fit.params else np.nan,
                "p_lag12": float(fit.pvalues["Y_lag12"]) if "Y_lag12" in fit.pvalues else np.nan,
                "coef_rev_short_1": round(float(fit.params["rev_short_1"]), 4) if "rev_short_1" in fit.params else np.nan,
                "p_rev_short_1": float(fit.pvalues["rev_short_1"]) if "rev_short_1" in fit.pvalues else np.nan,
            }
        )
    return seas_df, pd.DataFrame(rt_results)


def build_sectoral_diagnostics(vlong: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos y pruebas HAC para los 9 sectores a horizonte H=11 limpio."""
    rows: List[Dict[str, object]] = []
    for ind, grp in vlong.groupby("industry"):
        grp = grp.sort_values(["target_date", "vintage_date"])
        vals: List[float] = []
        for t_date, tg in grp.groupby("target_date"):
            r0 = tg.iloc[0]
            lag0 = (r0["vintage_date"].year - t_date.year) * 12 + (r0["vintage_date"].month - t_date.month)
            if lag0 > 3:
                continue
            after = tg[tg["vintage_date"] >= r0["vintage_date"] + pd.DateOffset(months=PRIMARY_H)]
            if len(after) == 0:
                continue
            rh = after.iloc[0]
            if bool(r0["is_base_year_sentinel"]) or bool(rh["is_base_year_sentinel"]):
                continue
            vals.append(float(rh["value"] - r0["value"]))
        arr = np.asarray(vals, dtype=float)
        l_nw = newey_west_lags(len(arr))
        mean_i, se_i, p_i = fit_mean_hac(arr, l_nw)
        rows.append(
            {
                "industry": ind,
                "n": len(arr),
                "L_nw": l_nw,
                "media": round(mean_i, 4),
                "se_nw": round(se_i, 4),
                "p_nw": float(p_i),
                "ic95_inf": round(mean_i - 1.96 * se_i, 4),
                "ic95_sup": round(mean_i + 1.96 * se_i, 4),
                "de": round(float(np.std(arr, ddof=1)), 4),
                "mediana": round(float(np.median(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "asimetria": round(float(stats.skew(arr, bias=False)), 4),
                "curtosis_exceso": round(float(stats.kurtosis(arr, fisher=True, bias=False)), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("media", ascending=False).reset_index(drop=True)


def generate_figures(
    target_df: pd.DataFrame, acf_df: pd.DataFrame, seas_df: pd.DataFrame, sec_df: pd.DataFrame
) -> None:
    """Genera las 3 figuras de Etapa 2 y las copia en main_tex/images/."""
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_IMG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": "#444444", "axes.linewidth": 0.8})

    # Figura 1: ACF calendario + Estacionalidad mensual
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0), dpi=180)
    k = acf_df["rezago_k"].to_numpy()
    obs_mask = acf_df["observable_en_v0"].to_numpy()
    ax1.bar(k[~obs_mask], acf_df.loc[~obs_mask, "acf_calendario"], width=0.65, color="#A6A6A6", label="No observable en $v_0$ ($k < 11$)")
    ax1.bar(k[obs_mask], acf_df.loc[obs_mask, "acf_calendario"], width=0.65, color="#1F4E79", label="Observable en $v_0$ ($k \\geq 11$)")
    ax1.plot(k, acf_df["ic95_banda"], "--", color="#B22222", linewidth=1.0, label="Banda 95% ($\\pm 1{,}96/\\sqrt{n_k}$)")
    ax1.plot(k, -acf_df["ic95_banda"], "--", color="#B22222", linewidth=1.0)
    ax1.axhline(0.0, color="#333333", linewidth=0.8)
    ax1.axvline(10.5, color="#E07B00", linestyle=":", linewidth=1.3)
    ax1.set_xticks([1, 3, 6, 9, 11, 12, 15, 18])
    ax1.set_ylim(-0.16, 0.58)
    ax1.set_xlabel("Rezago calendario $k$ (meses)")
    ax1.set_ylabel("Autocorrelación $\\hat{\\rho}_k$")
    ax1.set_title("Autocorrelación de $Y_t^{(11)}$ y frontera observable")
    ax1.legend(frameon=False, fontsize=7.8, loc="upper right")
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    ax2.errorbar(
        seas_df["mes"],
        seas_df["media"],
        yerr=1.96 * seas_df["se_nw"],
        fmt="o-",
        color="#1F4E79",
        ecolor="#7FA3C4",
        capsize=3.5,
        linewidth=1.4,
    )
    ax2.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax2.set_xlabel("Mes calendario de referencia ($t$)")
    ax2.set_ylabel("Revisión media $Y_t^{(11)}$ (p.p. $\\pm$ IC 95% HAC)")
    ax2.set_title("Estacionalidad mensual de las revisiones")
    ax2.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIG_OUT_DIR / "etapa2_acf_pacf.png", dpi=180)
    plt.close(fig)

    # Figura 2: Evolución temporal y regímenes (1993-2024)
    sub11 = get_clean_sample(target_df, PRIMARY_H)
    roll_mean = sub11["Y_H11"].rolling(24, min_periods=12, center=True).mean()
    roll_sd = sub11["Y_H11"].rolling(24, min_periods=12, center=True).std()

    fig, ax = plt.subplots(figsize=(9.2, 4.1), dpi=180)
    ax.axvspan(pd.Timestamp("1999-05-01"), pd.Timestamp("2000-05-01"), color="#B22222", alpha=0.10, label="Cruces de base excluidos (2000m7 / 2014m3)")
    ax.axvspan(pd.Timestamp("2013-01-01"), pd.Timestamp("2014-01-01"), color="#B22222", alpha=0.10)
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2024-11-01"), color="#E07B00", alpha=0.08, label="Período COVID y post-COVID (2020--2024)")
    ax.plot(sub11["target_date"], sub11["Y_H11"], color="#7FA3C4", linewidth=0.9, alpha=0.85, label="$Y_t^{(11)}$ limpio ($n=350$)")
    ax.plot(sub11["target_date"], roll_mean, color="#1F4E79", linewidth=1.8, label="Media móvil (24 meses)")
    ax.fill_between(sub11["target_date"], roll_mean - roll_sd, roll_mean + roll_sd, color="#1F4E79", alpha=0.15, label="Banda $\\pm 1$ DE móvil")
    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=0.9)
    ax.set_ylabel("Revisión $Y_t^{(11)}$ (p.p.)")
    ax2_title = "Dinámica temporal, agrupamiento de volatilidad y regímenes ($H=11$ limpio)"
    ax.set_title(ax2_title)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right", ncol=2)
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIG_OUT_DIR / "etapa2_regimenes_evolucion.png", dpi=180)
    plt.close(fig)

    # Figura 3: Sesgo medio HAC por sector (H=11 limpio)
    sec_plot = sec_df.sort_values("media", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=180)
    y_pos = np.arange(len(sec_plot))
    colors = ["#B22222" if ind == "gdp" else ("#1F4E79" if p < 0.05 else "#888888") for ind, p in zip(sec_plot["industry"], sec_plot["p_nw"])]
    for idx, row_s in sec_plot.iterrows():
        ax.errorbar(
            row_s["media"],
            idx,
            xerr=1.96 * row_s["se_nw"],
            fmt="o",
            color=colors[idx],
            ecolor=colors[idx],
            capsize=4,
            markersize=6,
            linewidth=1.5,
        )
    ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['industry']} (n={int(r['n'])})" for _, r in sec_plot.iterrows()])
    ax.set_xlabel("Revisión media $Y_{i,t}^{(11)}$ (p.p., IC 95% HAC)")
    ax.set_title("Sesgo medio de revisión por sector económico ($H=11$ limpio)")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIG_OUT_DIR / "etapa2_sectores_sesgo.png", dpi=180)
    plt.close(fig)

    for png in FIG_OUT_DIR.glob("*.png"):
        shutil.copy2(png, TEX_IMG_DIR / png.name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_df = pd.read_csv(
        E1_DIR / "muestra_objetivo_gdp.csv",
        parse_dates=["target_date", "v0_date", "v_last_date", "v_H3_date", "v_H6_date", "v_H11_date", "v_H12_date"],
    )
    vlong = pd.read_csv(E1_DIR / "vintages_long.csv", parse_dates=["vintage_date", "target_date"])

    horizon_diag_df = build_horizon_diagnostics(target_df)
    acf_df = build_acf_pacf(target_df)
    mz_df, reg_df = build_mincer_zarnowitz_table(target_df)
    seas_df, rt_df = build_seasonality_and_rt_models(target_df, vlong)
    sec_df = build_sectoral_diagnostics(vlong)

    outputs = {
        "etapa2_descriptivos_horizontes.csv": horizon_diag_df,
        "etapa2_acf_pacf_y11.csv": acf_df,
        "etapa2_mincer_zarnowitz.csv": mz_df,
        "etapa2_regimenes_y11.csv": reg_df,
        "etapa2_estacionalidad_y11.csv": seas_df,
        "etapa2_asociacion_tiempo_real.csv": rt_df,
        "etapa2_sectores_h11.csv": sec_df,
    }
    for fname, df_out in outputs.items():
        df_out.to_csv(OUT_DIR / fname, index=False)

    generate_figures(target_df, acf_df, seas_df, sec_df)

    checksums = [
        {"tipo": "tabla_econometrica", "archivo": p.name, "tamano_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(OUT_DIR.glob("*.csv"))
    ] + [
        {"tipo": "figura", "archivo": f"figures/{p.name}", "tamano_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(FIG_OUT_DIR.glob("*.png"))
    ]
    pd.DataFrame(checksums).to_csv(OUT_DIR / "etapa2_checksums_sha256.csv", index=False)

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizons": horizon_diag_df.to_dict(orient="records"),
        "mincer_zarnowitz": mz_df.to_dict(orient="records"),
        "regimes": reg_df.to_dict(orient="records"),
        "rt_models": rt_df.to_dict(orient="records"),
        "sectors": sec_df.to_dict(orient="records"),
    }
    (OUT_DIR / "etapa2_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("Etapa 2 completada.\n")
    print("=== Pruebas de sesgo y normalidad (limpias) ===")
    print(horizon_diag_df[horizon_diag_df["muestra"] == "limpia"][["horizonte_H", "n", "media", "se_nw", "p_nw", "se_solap", "p_solap", "asimetria", "curtosis_exceso", "jb_p"]].to_string(index=False))
    print("\n=== Mincer-Zarnowitz ===")
    print(mz_df[["especificacion", "n", "alpha", "se_alpha_nw", "beta", "se_beta_nw", "wald_f_nw", "wald_p_nw", "r2", "arch1_p", "reset_p"]].to_string(index=False))
    print("\n=== Modelos observables en v0 ===")
    print(rt_df[["modelo", "n", "r2", "r2_ajustado", "wald_f_nw", "wald_p_nw"]].to_string(index=False))


if __name__ == "__main__":
    main()
