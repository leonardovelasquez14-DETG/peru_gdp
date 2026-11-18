#!/usr/bin/env python3
"""
Etapa 8 (+ cierre de Etapa 7 y Anexo J/L): Inferencia integrada, correccion por
multiplicidad (Holm-Bonferroni / BH-FDR), Model Confidence Set (MCS) y clasificacion
formal preespecificada sobre las 12 arquitecturas (Etapas 3 y 4).

Salidas en output/etapa8_inferencia/:
  - etapa8_familia_confirmatoria_holm.csv
  - etapa8_model_confidence_set.csv
  - etapa8_verificacion_etapa7_gate.json
  - etapa8_reproducibilidad_global.csv
  - etapa8_summary.json
  - etapa8_checksums_sha256.csv
  - etapa8_mcs_inferencia.png (copia en main_tex/images/)
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rtd_eval_utils as u  # noqa: E402

ETAPA1_DIR = ROOT_DIR / "output" / "analysis_output"
ETAPA2_DIR = ROOT_DIR / "output" / "etapa2_econometria"
ETAPA3_DIR = ROOT_DIR / "output" / "etapa3_clasicos"
ETAPA4_DIR = ROOT_DIR / "output" / "etapa4_reservorios"
ETAPA6_DIR = ROOT_DIR / "output" / "etapa6_sintetico"
OUT_DIR = ROOT_DIR / "output" / "etapa8_inferencia"
TEX_IMG_DIR = ROOT_DIR / "main_tex" / "images"

ALL_MODELS = [
    "Zero",
    "BiasCorr_Mean",
    "BiasCorr_Roll60",
    "AR_BIC",
    "ARX_BIC",
    "Ridge",
    "DFM_Factor_Ridge",
    "GBDT_NumPy",
    "RFF_20",
    "ESN_20",
    "ESN_64",
    "QRC_Ising5",
]


def load_unified_predictions() -> pd.DataFrame:
    """Combina predicciones causales exactas de Etapa 3 y Etapa 4 sobre n=250."""
    e3 = pd.read_csv(ETAPA3_DIR / "etapa3_predicciones_oos.csv")
    e4 = pd.read_csv(ETAPA4_DIR / "etapa4_predicciones_oos.csv")
    assert e3["eval_idx"].tolist() == e4["eval_idx"].tolist()
    assert e3["target_period"].tolist() == e4["target_period"].tolist()
    assert np.allclose(e3["Y_H11"].to_numpy(), e4["Y_H11"].to_numpy(), atol=1e-12)

    df = e3.copy()
    for m in ("RFF_20", "ESN_20", "ESN_64", "QRC_Ising5"):
        df[f"pred_{m}"] = e4[f"pred_{m}"]
    return df


def holm_bonferroni(p_vals: np.ndarray) -> np.ndarray:
    """Ajuste paso descendente Holm-Bonferroni que controla FWER."""
    m = len(p_vals)
    order = np.argsort(p_vals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for k, idx in enumerate(order):
        val = min(1.0, (m - k) * float(p_vals[idx]))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def benjamini_hochberg(p_vals: np.ndarray) -> np.ndarray:
    """Ajuste Benjamini-Hochberg que controla FDR."""
    m = len(p_vals)
    order = np.argsort(p_vals)[::-1]
    adj = np.empty(m, dtype=float)
    running_min = 1.0
    for rank_rev, idx in enumerate(order):
        rank = m - rank_rev
        val = min(1.0, (m / rank) * float(p_vals[idx]))
        running_min = min(running_min, val)
        adj[idx] = running_min
    return adj


def classify_thesis_decision(rmse_diff: float, ci_lo: float, ci_hi: float, p_holm: float) -> str:
    """Clasifica contraste segun las categorias preespecificadas de la Seccion 9.5."""
    if p_holm < 0.05 and ci_hi < 0.0:
        return "Evidencia favorable (a favor de QRC)"
    if p_holm < 0.05 and ci_lo > 0.0:
        return "Evidencia desfavorable (competidor supera a QRC)"
    if ci_hi < 0.0:
        return "Mejora puntual/IC favorable sin significancia Holm"
    if ci_lo > 0.0:
        return "Desventaja puntual/IC desfavorable sin significancia Holm"
    return "Resultado inconcluso (IC amplio o empate estadistico)"


def build_confirmatory_family(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evalua la familia confirmatoria preespecificada de QRC_Ising5 frente a:
      Zero, BiasCorr_Mean, AR_BIC, Ridge (benchmark principal), GBDT_NumPy, ESN_20 y RFF_20
    en muestra global (n=250), pre-congelado (n=190) y bloque congelado (n=60),
    aplicando Holm-Bonferroni y BH-FDR dentro de cada alcance.
    """
    comparators = ("Zero", "BiasCorr_Mean", "AR_BIC", "Ridge", "GBDT_NumPy", "ESN_20", "RFF_20")
    scopes = [
        ("global_n250", df),
        ("pre_congelado_n190", df[df["bloque_eval"] == "pre_congelado_2003m2_2019m11"]),
        ("congelado_n60", df[df["bloque_eval"] == "congelado_2019m12_2024m11"]),
    ]

    all_rows: List[Dict[str, object]] = []
    for scope_name, sub in scopes:
        y = sub["Y_H11"].to_numpy(dtype=float)
        e_qrc = y - sub["pred_QRC_Ising5"].to_numpy(dtype=float)
        rmse_qrc = float(np.sqrt(np.mean(e_qrc ** 2)))
        scope_rows: List[Dict[str, object]] = []

        for comp in comparators:
            e_comp = y - sub[f"pred_{comp}"].to_numpy(dtype=float)
            rmse_comp = float(np.sqrt(np.mean(e_comp ** 2)))
            d_mse, dm_sq, p_sq = u.diebold_mariano_test(e_qrc, e_comp, loss="sq")
            _, dm_abs, p_abs = u.diebold_mariano_test(e_qrc, e_comp, loss="abs")
            diff_r, ci_lo, ci_hi = u.block_bootstrap_rmse_diff(
                e_qrc, e_comp, block_len=12, n_boot=2000, seed=42
            )
            rel_gain_pct = 100.0 * (rmse_comp - rmse_qrc) / rmse_comp
            scope_rows.append(
                {
                    "alcance": scope_name,
                    "n": int(len(y)),
                    "modelo": "QRC_Ising5",
                    "comparador": comp,
                    "rmse_qrc": rmse_qrc,
                    "rmse_comp": rmse_comp,
                    "mejora_rel_pct": rel_gain_pct,
                    "rmse_diff": diff_r,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "dm_hln_sq": dm_sq,
                    "p_dm_sq_raw": p_sq,
                    "dm_hln_abs": dm_abs,
                    "p_dm_abs_raw": p_abs,
                }
            )

        p_raw_arr = np.array([float(r["p_dm_sq_raw"]) for r in scope_rows])
        p_holm = holm_bonferroni(p_raw_arr)
        p_fdr = benjamini_hochberg(p_raw_arr)
        for j, r in enumerate(scope_rows):
            r["p_holm_sq"] = float(p_holm[j])
            r["p_bh_fdr_sq"] = float(p_fdr[j])
            r["clasificacion_decision"] = classify_thesis_decision(
                float(r["rmse_diff"]),
                float(r["ci95_low"]),
                float(r["ci95_high"]),
                float(p_holm[j]),
            )
        all_rows.extend(scope_rows)

    return pd.DataFrame(all_rows)


def compute_mcs_t_max(
    losses: np.ndarray, model_names: Sequence[str], block_len: int = 12, n_boot: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """
    Model Confidence Set (Hansen, Lunde & Nason, 2011) usando estadistico T_max
    y bootstrap circular por bloques fijo compartido en todas las eliminaciones.
    """
    n, m_total = losses.shape
    rng = np.random.default_rng(seed)
    b_len = min(block_len, n)
    n_blocks = int(np.ceil(n / b_len))

    # Precomputar promedios bootstrap (B, M) una sola vez
    boot_means = np.empty((n_boot, m_total), dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(b_len)) % n for s in starts])[:n]
        boot_means[b] = np.mean(losses[idx], axis=0)

    sample_means = np.mean(losses, axis=0)
    active = list(range(m_total))
    elim_order: List[int] = []
    step_pvals: List[float] = []

    while len(active) > 1:
        sub_sample = sample_means[active]
        sub_boot = boot_means[:, active]

        d_i_dot = sub_sample - np.mean(sub_sample)
        d_boot_i_dot = sub_boot - np.mean(sub_boot, axis=1, keepdims=True)
        var_hat = np.maximum(np.mean((d_boot_i_dot - d_i_dot) ** 2, axis=0), 1e-14)
        std_hat = np.sqrt(var_hat)

        t_stats = d_i_dot / std_hat
        t_max_obs = float(np.max(t_stats))

        # Estadistico T_max de una cola sobre perdidas relativas centradas
        t_boot_signed = (d_boot_i_dot - d_i_dot) / std_hat
        t_max_boot = np.max(t_boot_signed, axis=1)
        p_step = float(np.mean(t_max_boot >= t_max_obs))

        worst_pos = int(np.argmax(t_stats))
        elim_order.append(active.pop(worst_pos))
        step_pvals.append(p_step)

    # Ultimo sobreviviente
    elim_order.append(active[0])
    step_pvals.append(1.0)

    # Imponer monotonicidad secuencial del MCS
    mcs_pvals = np.empty(m_total, dtype=float)
    running_p = 0.0
    for step_idx, m_idx in enumerate(elim_order):
        running_p = max(running_p, step_pvals[step_idx])
        mcs_pvals[m_idx] = min(1.0, running_p)

    rows = []
    rank_surv = {m_idx: (m_total - step_k) for step_k, m_idx in enumerate(elim_order)}
    for m_idx, name in enumerate(model_names):
        p_val = float(mcs_pvals[m_idx])
        rows.append(
            {
                "modelo": name,
                "mean_loss": float(sample_means[m_idx]),
                "rank_mcs": int(rank_surv[m_idx]),
                "p_mcs": p_val,
                "in_mcs_90": bool(p_val >= 0.10),
                "in_mcs_75": bool(p_val >= 0.25),
            }
        )
    return pd.DataFrame(rows).sort_values(["rank_mcs", "mean_loss"]).reset_index(drop=True)


def evaluate_mcs_all_scopes(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el Model Confidence Set bajo perdida cuadratica y absoluta por alcance."""
    scopes = [
        ("global_n250", df),
        ("pre_congelado_n190", df[df["bloque_eval"] == "pre_congelado_2003m2_2019m11"]),
        ("congelado_n60", df[df["bloque_eval"] == "congelado_2019m12_2024m11"]),
    ]
    out_frames: List[pd.DataFrame] = []
    for scope_name, sub in scopes:
        y = sub["Y_H11"].to_numpy(dtype=float)
        err_mat = np.column_stack([y - sub[f"pred_{m}"].to_numpy(dtype=float) for m in ALL_MODELS])
        for loss_name, loss_mat in (("MSE", err_mat ** 2), ("MAE", np.abs(err_mat))):
            res = compute_mcs_t_max(loss_mat, ALL_MODELS, block_len=12, n_boot=2000, seed=42)
            res.insert(0, "perdida", loss_name)
            res.insert(0, "alcance", scope_name)
            out_frames.append(res)
    return pd.concat(out_frames, ignore_index=True)


def verify_etapa7_gate(family_df: pd.DataFrame) -> Dict[str, object]:
    """Audita formalmente la condicion previa de activacion de la Etapa 7."""
    e6_sum = json.loads((ETAPA6_DIR / "etapa6_summary.json").read_text(encoding="utf-8"))
    e6_res = pd.read_csv(ETAPA6_DIR / "etapa6_resumen_escenarios.csv")

    g = family_df[family_df["alcance"] == "global_n250"].set_index("comparador")
    cond_beats_ridge = bool(g.loc["Ridge", "p_holm_sq"] < 0.05 and g.loc["Ridge", "ci95_high"] < 0.0)
    cond_beats_esn = bool(g.loc["ESN_20", "p_holm_sq"] < 0.05 and g.loc["ESN_20", "ci95_high"] < 0.0)
    cond_beats_rff = bool(g.loc["RFF_20", "p_holm_sq"] < 0.05 and g.loc["RFF_20", "ci95_high"] < 0.0)
    cond_syn_fav = bool(
        (e6_res["favorable_vs_Ridge"].sum() > 0) and (e6_res["favorable_vs_ESN20"].sum() > 0)
    )
    trigger = bool(cond_beats_ridge and cond_beats_esn and cond_beats_rff)

    return {
        "etapa7_activada": trigger,
        "condicion_1_supera_Ridge_holm95": cond_beats_ridge,
        "condicion_2_supera_ESN20_holm95": cond_beats_esn,
        "condicion_3_supera_RFF20_holm95": cond_beats_rff,
        "condicion_4_region_sintetica_favorable_no_vacia": cond_syn_fav,
        "nota_protocolo": (
            "Se aplica la regla de paro preespecificada en la Seccion 9.3.7 y Anexo I: "
            "no se ejecuta ni se monetiza analisis economico atribuible a QRC porque no existe "
            "mejora robusta sobre los controles clasicos equivalentes (RFF_20 supera a QRC con "
            "p_raw=0.0146; ESN_20 empata con p_raw=0.730; y region sintetica A = vacio)."
        ),
        "software_env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "etapa6_delta_min_pct": e6_sum["delta_min_pct"],
    }


def collect_global_reproducibility() -> pd.DataFrame:
    """Consolida inventario y checksums SHA-256 verificados de las Etapas 1 a 6."""
    records: List[Dict[str, object]] = []
    stage_dirs = [
        ("Etapa_1", ETAPA1_DIR),
        ("Etapa_2", ETAPA2_DIR),
        ("Etapa_3", ETAPA3_DIR),
        ("Etapa_4", ETAPA4_DIR),
        ("Etapa_6", ETAPA6_DIR),
    ]
    for stage, d in stage_dirs:
        for p in sorted(d.iterdir()):
            if p.is_file() and "checksums" not in p.name:
                records.append(
                    {
                        "etapa": stage,
                        "archivo": p.name,
                        "ruta_relativa": str(p.relative_to(ROOT_DIR)),
                        "bytes": int(p.stat().st_size),
                        "sha256": u.sha256_file(p),
                    }
                )
    return pd.DataFrame(records)


def make_figure(mcs_df: pd.DataFrame, fam_df: pd.DataFrame) -> List[Path]:
    """Genera grafica de sintesis final Etapa 8: MCS por regimenes + familia Holm."""
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.titlesize": 10.6,
            "axes.labelsize": 9.6,
            "legend.fontsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    TEX_IMG_DIR.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(12.2, 5.9), gridspec_kw={"width_ratios": [1.12, 1.05]}
    )

    # Panel izquierdo: MCS p-valores (MSE) por modelo y bloque
    mse_mcs = mcs_df[mcs_df["perdida"] == "MSE"]
    order_g = (
        mse_mcs[mse_mcs["alcance"] == "global_n250"]
        .sort_values("mean_loss", ascending=True)["modelo"]
        .tolist()
    )
    y_pos = np.arange(len(order_g))[::-1]
    scope_styles = [
        ("global_n250", "Global (n=250)", "#1B365D", "o", 0.16),
        ("pre_congelado_n190", "Pre-COVID (n=190)", "#2E8B57", "s", 0.0),
        ("congelado_n60", "Bloque congelado (n=60)", "#C84B31", "^", -0.16),
    ]
    ax_l.axvspan(0.10, 1.06, color="#E8F5E9", alpha=0.55, label=r"Supervivencia MCS$_{90\%}$ ($p \geq 0,10$)")
    ax_l.axvline(0.10, color="#2E8B57", ls="--", lw=1.1)
    for sc_key, sc_lbl, col, mk, off in scope_styles:
        sub = mse_mcs[mse_mcs["alcance"] == sc_key].set_index("modelo")
        vals = [float(sub.loc[m, "p_mcs"]) for m in order_g]
        ax_l.scatter(vals, y_pos + off, color=col, marker=mk, s=48, label=sc_lbl, zorder=4)
        for v, y_i in zip(vals, y_pos + off):
            ax_l.hlines(y_i, 0.0, v, color=col, lw=1.0, alpha=0.45)

    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(order_g)
    for tick_lbl in ax_l.get_yticklabels():
        if tick_lbl.get_text() == "QRC_Ising5":
            tick_lbl.set_fontweight("bold")
            tick_lbl.set_color("#1B365D")
    ax_l.set_xlim(-0.02, 1.06)
    ax_l.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ",")))
    ax_l.set_xlabel(r"$p$-valor del Model Confidence Set ($T_{\max}$, pérdida MSE)")
    ax_l.set_title("Supervivencia en MCS por bloque (ordenados por RMSE global)", pad=10)
    ax_l.grid(axis="x", alpha=0.25)
    ax_l.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, -0.15),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        fontsize=8.0,
    )

    # Panel derecho: Contrastes confirmatorios de QRC_Ising5 en global_n250
    fg = fam_df[fam_df["alcance"] == "global_n250"].copy()
    yg = np.arange(len(fg))[::-1]
    diffs = fg["rmse_diff"].to_numpy()

    ax_r.axvline(0.0, color="#333333", ls="--", lw=1.0, zorder=2)
    colors = []
    for _, r in fg.iterrows():
        if r["p_holm_sq"] < 0.05 and r["rmse_diff"] < 0:
            colors.append("#2E8B57")
        elif r["p_holm_sq"] < 0.05 and r["rmse_diff"] > 0:
            colors.append("#C84B31")
        elif r["ci95_high"] < 0:
            colors.append("#2B5B84")
        elif r["ci95_low"] > 0:
            colors.append("#B24A34")
        else:
            colors.append("#6C757D")

    x_col_text = 0.060
    for i_row, (_, r) in enumerate(fg.iterrows()):
        yi = yg[i_row]
        ax_r.errorbar(
            r["rmse_diff"],
            yi,
            xerr=[[max(0.0, r["rmse_diff"] - r["ci95_low"])], [max(0.0, r["ci95_high"] - r["rmse_diff"])]],
            fmt="o",
            color=colors[i_row],
            ecolor=colors[i_row],
            elinewidth=1.8,
            capsize=3.5,
            markersize=6.2,
            zorder=4,
        )
        p_txt = f"p_H={r['p_holm_sq']:.3f}".replace(".", ",")
        ax_r.text(
            x_col_text,
            yi,
            p_txt,
            va="center",
            ha="left",
            fontsize=8.2,
            color="#222222",
        )

    ax_r.set_yticks(yg)
    ax_r.set_yticklabels([f"QRC vs. {c}" for c in fg["comparador"]])
    ax_r.set_xlim(-0.235, 0.165)
    ax_r.set_xticks([-0.20, -0.15, -0.10, -0.05, 0.00, 0.05])
    ax_r.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}".replace(".", ",")))
    ax_r.set_xlabel(r"$\Delta\mathrm{RMSE} = \mathrm{RMSE}_{\mathrm{QRC}} - \mathrm{RMSE}_{\mathrm{comp}}$ (p.p., <0 favorece a QRC)")
    ax_r.set_title("Familia confirmatoria QRC (n=250, IC 95% y p Holm)", pad=10)
    ax_r.grid(axis="x", alpha=0.25)

    fig.subplots_adjust(wspace=0.40, bottom=0.21)
    for d in (OUT_DIR, TEX_IMG_DIR):
        p = d / "etapa8_mcs_inferencia.png"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        generated.append(p)
    plt.close(fig)
    return generated


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_only = "--plot-only" in sys.argv[1:]

    csv_paths = {
        "familia_holm": OUT_DIR / "etapa8_familia_confirmatoria_holm.csv",
        "mcs": OUT_DIR / "etapa8_model_confidence_set.csv",
        "repro": OUT_DIR / "etapa8_reproducibilidad_global.csv",
    }
    gate_path = OUT_DIR / "etapa8_verificacion_etapa7_gate.json"
    summary_path = OUT_DIR / "etapa8_summary.json"

    if plot_only:
        fam_df = pd.read_csv(csv_paths["familia_holm"])
        mcs_df = pd.read_csv(csv_paths["mcs"])
    else:
        t0 = time.perf_counter()
        df = load_unified_predictions()
        fam_df = build_confirmatory_family(df)
        mcs_df = evaluate_mcs_all_scopes(df)
        gate = verify_etapa7_gate(fam_df)
        repro_df = collect_global_reproducibility()

        fam_df.to_csv(csv_paths["familia_holm"], index=False)
        mcs_df.to_csv(csv_paths["mcs"], index=False)
        repro_df.to_csv(csv_paths["repro"], index=False)
        gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

        summary_payload = {
            "elapsed_sec": float(time.perf_counter() - t0),
            "etapa7_gate": gate,
            "familia_global": fam_df[fam_df["alcance"] == "global_n250"].to_dict(orient="records"),
            "mcs_global_mse": mcs_df[
                (mcs_df["alcance"] == "global_n250") & (mcs_df["perdida"] == "MSE")
            ].to_dict(orient="records"),
            "mcs_congelado_mse": mcs_df[
                (mcs_df["alcance"] == "congelado_n60") & (mcs_df["perdida"] == "MSE")
            ].to_dict(orient="records"),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    figs = make_figure(mcs_df, fam_df)
    hash_targets = list(csv_paths.values()) + [gate_path, summary_path] + [
        p for p in figs if p.parent == OUT_DIR
    ]
    chk_rows = [
        {"archivo": p.name, "sha256": u.sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(hash_targets, key=lambda x: x.name)
    ]
    pd.DataFrame(chk_rows).to_csv(OUT_DIR / "etapa8_checksums_sha256.csv", index=False)

    print("\n=== Familia Confirmatoria QRC_Ising5 con Holm-Bonferroni (Global n=250) ===")
    cols_f = ["comparador", "rmse_qrc", "rmse_comp", "rmse_diff", "p_dm_sq_raw", "p_holm_sq", "p_bh_fdr_sq", "clasificacion_decision"]
    print(fam_df[fam_df["alcance"] == "global_n250"][cols_f].to_string(index=False))
    print("\n=== Model Confidence Set (MSE, Global n=250) ===")
    print(mcs_df[(mcs_df["alcance"] == "global_n250") & (mcs_df["perdida"] == "MSE")].to_string(index=False))
    print("\n=== Model Confidence Set (MSE, Bloque congelado n=60) ===")
    print(mcs_df[(mcs_df["alcance"] == "congelado_n60") & (mcs_df["perdida"] == "MSE")].to_string(index=False))


if __name__ == "__main__":
    main()
