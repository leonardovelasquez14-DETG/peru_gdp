#!/usr/bin/env python3
"""
Etapa 1: Auditoría reproducible y reconstrucción de vintages del PBI peruano.

Lee los archivos originales de solo lectura en:
  - peru_gdp_rtd_output/output/
  - peruvian_gdp_revisions-main/metadata/wr_metadata.csv
  - peruvian_gdp_revisions-main/CITATION.cff

Genera los seis entregables auditados del Anexo A en output/analysis_output/
junto con un manifiesto de integridad SHA-256 y figuras diagnósticas en
output/analysis_output/figures/ y main_tex/images/.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE_VERSION = "1.0.0"
SENTINEL_BASE_YEAR = -999999.0
BASE_YEAR_VINTAGES = ("2000m7", "2014m3")
HORIZONS = (3, 6, 9, 10, 11, 12, 18, 24)
PRIMARY_HORIZON = 11  # Máxima cobertura anual continua (0 huecos internos)
ALT_HORIZON = 12      # Horizonte nominal de 4 trimestres (con 18 huecos internos)

ROOT_DIR = Path(__file__).resolve().parent.parent
RTD_OUTPUT_DIR = ROOT_DIR / "peru_gdp_rtd_output" / "output"
META_PATH = ROOT_DIR / "peruvian_gdp_revisions-main" / "metadata" / "wr_metadata.csv"
CITATION_PATH = ROOT_DIR / "peruvian_gdp_revisions-main" / "CITATION.cff"
OUT_DIR = ROOT_DIR / "output" / "analysis_output"
FIG_OUT_DIR = OUT_DIR / "figures"
TEX_IMG_DIR = ROOT_DIR / "main_tex" / "images"


def sha256_file(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo en bloques."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ym_series(series: pd.Series) -> pd.Series:
    """Convierte cadenas YYYYmm/YYYYmM (con o sin prefijo tp_) a Timestamp."""
    cleaned = series.astype(str).str.strip().str.replace(r"^tp_", "", regex=True)
    return pd.to_datetime(cleaned.str.replace("m", "-", regex=False), format="%Y-%m", errors="coerce")


def month_diff(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Diferencia exacta en meses calendario entre dos series Timestamp."""
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


def build_source_inventory(root: Path) -> pd.DataFrame:
    """(i) Construye inventario completo con SHA-256, dimensiones y duplicados."""
    records: List[Dict[str, object]] = []
    targets = sorted((root / "peru_gdp_rtd_output" / "output").rglob("*.csv"))
    targets.extend([META_PATH, CITATION_PATH])

    for path in targets:
        rel = path.relative_to(root).as_posix()
        row: Dict[str, object] = {
            "archivo": path.name,
            "ruta_relativa": rel,
            "tamano_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "filas": np.nan,
            "columnas": np.nan,
            "filas_duplicadas_exactas": np.nan,
            "filas_duplicadas_clave": np.nan,
            "valores_centinela": 0,
            "filas_vacias": 0,
            "notas": "",
        }
        if path.suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
            row["filas"] = len(df)
            row["columnas"] = len(df.columns)
            row["filas_duplicadas_exactas"] = int(df.duplicated().sum())
            if {"industry", "vintage"}.issubset(df.columns):
                row["filas_duplicadas_clave"] = int(df.duplicated(subset=["industry", "vintage"]).sum())
                data_cols = [c for c in df.columns if c not in ("industry", "vintage")]
            elif {"year", "wr"}.issubset(df.columns):
                row["filas_duplicadas_clave"] = int(df.duplicated(subset=["year", "wr"]).sum())
                data_cols = [c for c in df.columns if c not in ("year", "wr")]
            else:
                data_cols = [c for c in df.columns if c != "target_period"]
            num = df[data_cols].apply(pd.to_numeric, errors="coerce")
            row["valores_centinela"] = int((num == SENTINEL_BASE_YEAR).sum().sum())
            row["filas_vacias"] = int(df[data_cols].isna().all(axis=1).sum())
            if "benchmark" in path.name and "releases" not in path.name:
                row["notas"] = "Indicador binario (0/1) de actualización simultánea Tabla 1 y 2; no equivale a cambio de año base."
            elif "adjusted" in path.name:
                row["notas"] = f"Incluye centinela {SENTINEL_BASE_YEAR} para transiciones de año base."
            elif path.name == "wr_metadata.csv":
                row["notas"] = "Resolución (year, wr, month); sin fecha calendario exacta (día)."
        else:
            row["notas"] = "Metadatos oficiales del repositorio (DOI Zenodo 10.5281/zenodo.18099975, licencia MIT)."
        records.append(row)

    return pd.DataFrame(records)


def build_data_dictionary() -> pd.DataFrame:
    """(ii) Genera el diccionario de variables de los artefactos auditados."""
    rows = [
        ("vintages_long.csv", "industry", "string", "código sectorial", "Sector económico (agriculture, commerce, construction, electricity, fishing, gdp, manufacturing, mining, services)"),
        ("vintages_long.csv", "vintage", "string", "YYYYmM", "Identificador mensual de publicación (Nota Semanal BCRP)"),
        ("vintages_long.csv", "vintage_date", "date", "YYYY-MM-DD", "Primer día del mes del vintage de publicación"),
        ("vintages_long.csv", "target_period", "string", "YYYYmM", "Período económico de referencia"),
        ("vintages_long.csv", "target_date", "date", "YYYY-MM-DD", "Primer día del mes del período económico objetivo"),
        ("vintages_long.csv", "value", "float", "puntos porcentuales", "Variación porcentual interanual del PBI/sector en el vintage dado"),
        ("vintages_long.csv", "is_base_year_sentinel", "bool", "0/1", "True si la celda está marcada con -999999.0 en monthly_gdp_vintages_adjusted.csv"),
        ("vintages_long.csv", "benchmark_revision", "int", "0/1", "Indicador de revisión simultánea Tabla 1 y Tabla 2 en wr_metadata.csv"),
        ("vintages_long.csv", "base_year", "int", "año", "Año base asignado al vintage en wr_metadata.csv (1990, 1994, 2007)"),
        ("muestra_objetivo_gdp.csv", "target_date", "date", "YYYY-MM-DD", "Período objetivo mensual del PBI total"),
        ("muestra_objetivo_gdp.csv", "v0_date", "date", "YYYY-MM-DD", "Primer vintage disponible para el período objetivo"),
        ("muestra_objetivo_gdp.csv", "lag0_months", "int", "meses", "Diferencia calendario v0_date - target_date (rezago inicial observado)"),
        ("muestra_objetivo_gdp.csv", "y0", "float", "puntos porcentuales", "Primera tasa publicada disponible en el repositorio"),
        ("muestra_objetivo_gdp.csv", "y_H11 / y_H12", "float", "puntos porcentuales", "Tasa publicada en el primer vintage >= v0_date + H meses"),
        ("muestra_objetivo_gdp.csv", "Y_H11 / Y_H12", "float", "puntos porcentuales", "Revisión a horizonte fijo: y_H - y0 (Ecuación 3)"),
        ("muestra_objetivo_gdp.csv", "crosses_base_year_H11 / H12", "bool", "0/1", "True si y0 o yH están afectados por cambio de año base (-999999.0)"),
        ("fechas_publicacion.csv", "pub_lag_months", "int", "meses", "Diferencia entre vintage_date y el período objetivo más reciente publicado en ese vintage"),
        ("fechas_publicacion.csv", "exact_day_available", "bool", "0/1", "False: el archivo wr_metadata.csv solo registra (year, wr, month) sin día calendario exacto"),
    ]
    return pd.DataFrame(rows, columns=["archivo", "variable", "tipo", "unidad_formato", "definicion"])


def load_and_audit_vintages() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Carga, verifica duplicados/fugas temporales y construye panel largo unificado."""
    v_raw = pd.read_csv(RTD_OUTPUT_DIR / "vintages" / "monthly_gdp_vintages.csv", low_memory=False)
    v_adj = pd.read_csv(RTD_OUTPUT_DIR / "vintages" / "monthly_gdp_vintages_adjusted.csv", low_memory=False)
    v_ben = pd.read_csv(RTD_OUTPUT_DIR / "vintages" / "monthly_gdp_vintages_benchmark.csv", low_memory=False)
    meta_raw = pd.read_csv(META_PATH)

    tp_cols = [c for c in v_raw.columns if c.startswith("tp_")]

    # Verificar que los duplicados por (industry, vintage) son exactamente idénticos
    dup_key = v_raw.duplicated(subset=["industry", "vintage"], keep=False)
    dup_exact = v_raw.duplicated(keep=False)
    non_exact_dups = int((dup_key & ~dup_exact).sum())
    if non_exact_dups > 0:
        raise ValueError(f"Se detectaron {non_exact_dups} filas con clave duplicada y valores distintos.")

    raw_rows_total = len(v_raw)
    v_raw_d = v_raw.drop_duplicates(subset=["industry", "vintage"]).copy()
    v_adj_d = v_adj.drop_duplicates(subset=["industry", "vintage"]).copy()
    v_ben_d = v_ben.drop_duplicates(subset=["industry", "vintage"]).copy()

    # Deduplicar wr_metadata conservando la última aparición por (year, month)
    meta_rows_raw = len(meta_raw)
    meta_d = (
        meta_raw.drop_duplicates()
        .sort_values(["year", "month", "wr"])
        .drop_duplicates(subset=["year", "month"], keep="last")
        .copy()
    )
    meta_d["vintage_date"] = pd.to_datetime(
        meta_d["year"].astype(int).astype(str) + "-" + meta_d["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )

    long_raw = v_raw_d.melt(id_vars=["industry", "vintage"], value_vars=tp_cols, var_name="tp", value_name="value")
    long_raw["value"] = pd.to_numeric(long_raw["value"], errors="coerce")
    long_raw = long_raw.dropna(subset=["value"]).copy()

    long_adj = v_adj_d.melt(id_vars=["industry", "vintage"], value_vars=tp_cols, var_name="tp", value_name="value_adj")
    long_adj["value_adj"] = pd.to_numeric(long_adj["value_adj"], errors="coerce")
    long_adj = long_adj.dropna(subset=["value_adj"]).copy()

    long_ben = v_ben_d.melt(id_vars=["industry", "vintage"], value_vars=tp_cols, var_name="tp", value_name="value_ben")
    long_ben["value_ben"] = pd.to_numeric(long_ben["value_ben"], errors="coerce")
    long_ben = long_ben.dropna(subset=["value_ben"]).copy()

    df = long_raw.merge(long_adj, on=["industry", "vintage", "tp"], how="left")
    df = df.merge(long_ben, on=["industry", "vintage", "tp"], how="left")
    df["target_period"] = df["tp"].str.replace(r"^tp_", "", regex=True)
    df["vintage_date"] = parse_ym_series(df["vintage"])
    df["target_date"] = parse_ym_series(df["target_period"])

    future_mask = df["target_date"] > df["vintage_date"]
    n_future = int(future_mask.sum())
    df = df.loc[~future_mask].copy()

    pre_dedup_cells = len(df)
    df = df.drop_duplicates(subset=["industry", "vintage_date", "target_date"]).copy()
    df["is_base_year_sentinel"] = df["value_adj"].eq(SENTINEL_BASE_YEAR)

    df = df.merge(
        meta_d[
            [
                "vintage_date",
                "wr",
                "revision_calendar_tab_1",
                "revision_calendar_tab_2",
                "benchmark_revision",
                "base_year",
                "base_year_affected",
            ]
        ],
        on="vintage_date",
        how="left",
    )

    df = df.sort_values(["industry", "target_date", "vintage_date"]).reset_index(drop=True)
    cols = [
        "industry",
        "vintage",
        "vintage_date",
        "target_period",
        "target_date",
        "value",
        "is_base_year_sentinel",
        "value_ben",
        "wr",
        "benchmark_revision",
        "base_year",
        "base_year_affected",
    ]
    stats = {
        "raw_wide_rows": raw_rows_total,
        "dedup_wide_rows": len(v_raw_d),
        "dropped_duplicate_wide_rows": raw_rows_total - len(v_raw_d),
        "non_exact_dup_wide_rows": non_exact_dups,
        "future_cells_discarded": n_future,
        "long_rows_after_dedup": len(df),
        "long_cell_dups_removed": pre_dedup_cells - len(df),
        "meta_rows_raw": meta_rows_raw,
        "meta_rows_dedup_month": len(meta_d),
    }
    return df[cols], meta_d, v_raw_d, stats


def build_publication_calendar(long_df: pd.DataFrame, meta_d: pd.DataFrame) -> pd.DataFrame:
    """(iv) Tabla de fechas de publicación y retraso de publicación por vintage."""
    gdp = long_df[long_df["industry"] == "gdp"]
    latest_tp = gdp.groupby("vintage_date", as_index=False)["target_date"].max().rename(
        columns={"target_date": "max_target_date"}
    )
    cal = meta_d.merge(latest_tp, on="vintage_date", how="left")
    cal["pub_lag_months"] = month_diff(cal["vintage_date"], cal["max_target_date"])
    cal["vintage"] = cal["vintage_date"].dt.strftime("%Ym") + cal["vintage_date"].dt.month.astype(str)
    cal["exact_day_available"] = False
    cal["nota_temporalidad"] = "Solo mes de publicación (Nota Semanal); sin fecha calendario exacta."
    cols = [
        "vintage",
        "vintage_date",
        "year",
        "wr",
        "month",
        "revision_calendar_tab_1",
        "revision_calendar_tab_2",
        "benchmark_revision",
        "base_year",
        "base_year_affected",
        "max_target_date",
        "pub_lag_months",
        "exact_day_available",
        "nota_temporalidad",
    ]
    return cal[cols].sort_values("vintage_date").reset_index(drop=True)


def build_gdp_target_dataset(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construye la muestra objetivo Y_t^{(H)} = y_t^{(v_H)} - y_t^{(v_0)} y resumen multi-horizonte."""
    gdp = long_df[long_df["industry"] == "gdp"].sort_values(["target_date", "vintage_date"]).copy()
    records: List[Dict[str, object]] = []

    for t_date, grp in gdp.groupby("target_date"):
        v0_row = grp.iloc[0]
        vlast_row = grp.iloc[-1]
        v0_date = v0_row["vintage_date"]
        lag0 = int((v0_date.year - t_date.year) * 12 + (v0_date.month - t_date.month))
        span_months = int((vlast_row["vintage_date"].year - v0_date.year) * 12 + (vlast_row["vintage_date"].month - v0_date.month))

        rec: Dict[str, object] = {
            "target_period": v0_row["target_period"],
            "target_date": t_date,
            "v0_vintage": v0_row["vintage"],
            "v0_date": v0_date,
            "lag0_months": lag0,
            "y0": float(v0_row["value"]),
            "y0_sentinel": bool(v0_row["is_base_year_sentinel"]),
            "v_last_vintage": vlast_row["vintage"],
            "v_last_date": vlast_row["vintage_date"],
            "y_last": float(vlast_row["value"]),
            "y_last_sentinel": bool(vlast_row["is_base_year_sentinel"]),
            "n_vintages": int(len(grp)),
            "span_months": span_months,
            "Y_var_last_minus_first": float(vlast_row["value"] - v0_row["value"]),
            "Y_var_first_minus_last_legacy": float(v0_row["value"] - vlast_row["value"]),
        }

        for h in HORIZONS:
            cutoff = v0_date + pd.DateOffset(months=h)
            after = grp[grp["vintage_date"] >= cutoff]
            if len(after) > 0:
                r_h = after.iloc[0]
                rec[f"v_H{h}_date"] = r_h["vintage_date"]
                rec[f"y_H{h}"] = float(r_h["value"])
                rec[f"Y_H{h}"] = float(r_h["value"] - v0_row["value"])
                rec[f"crosses_base_year_H{h}"] = bool(v0_row["is_base_year_sentinel"] or r_h["is_base_year_sentinel"])
            else:
                rec[f"v_H{h}_date"] = pd.NaT
                rec[f"y_H{h}"] = np.nan
                rec[f"Y_H{h}"] = np.nan
                rec[f"crosses_base_year_H{h}"] = False

        # Motivo de exclusión para horizonte principal (H=11) y alternativo (H=12)
        for tag, h in [("H11", PRIMARY_HORIZON), ("H12", ALT_HORIZON)]:
            if lag0 > 3:
                rec[f"estado_{tag}"] = "excluido_sin_primera_publicacion"
            elif pd.isna(rec[f"Y_{tag}"]):
                rec[f"estado_{tag}"] = "excluido_horizonte_incompleto"
            elif rec[f"crosses_base_year_{tag}"]:
                rec[f"estado_{tag}"] = "excluido_cambio_anio_base"
            else:
                rec[f"estado_{tag}"] = "incluido_limpio"

        records.append(rec)

    target_df = pd.DataFrame(records).sort_values("target_date").reset_index(drop=True)

    # Tabla resumen de horizontes H=3..24
    horizon_rows: List[Dict[str, object]] = []
    for h in HORIZONS:
        has_v0 = target_df["lag0_months"] <= 3
        has_h = has_v0 & target_df[f"Y_H{h}"].notna()
        clean_h = has_h & (~target_df[f"crosses_base_year_H{h}"])
        n_raw = int(has_h.sum())
        n_clean = int(clean_h.sum())
        if n_raw > 0:
            sub_dates = target_df.loc[has_h, "target_date"]
            full_grid = set(pd.date_range(sub_dates.min(), sub_dates.max(), freq="MS"))
            n_gaps = int(len(full_grid - set(sub_dates)))
            s_raw = target_df.loc[has_h, f"Y_H{h}"]
            s_clean = target_df.loc[clean_h, f"Y_H{h}"]
            horizon_rows.append(
                {
                    "horizonte_H_meses": h,
                    "obs_brutas_con_v0": n_raw,
                    "obs_limpias_sin_cambio_base": n_clean,
                    "obs_afectadas_cambio_base": n_raw - n_clean,
                    "huecos_internos": n_gaps,
                    "inicio_periodo": sub_dates.min().strftime("%Y-%m"),
                    "fin_periodo": sub_dates.max().strftime("%Y-%m"),
                    "media_bruta_pp": round(float(s_raw.mean()), 4),
                    "sd_bruta_pp": round(float(s_raw.std()), 4),
                    "media_limpia_pp": round(float(s_clean.mean()), 4),
                    "sd_limpia_pp": round(float(s_clean.std()), 4),
                    "mediana_limpia_pp": round(float(s_clean.median()), 4),
                    "min_limpia_pp": round(float(s_clean.min()), 4),
                    "max_limpia_pp": round(float(s_clean.max()), 4),
                }
            )
        else:
            horizon_rows.append(
                {
                    "horizonte_H_meses": h,
                    "obs_brutas_con_v0": 0,
                    "obs_limpias_sin_cambio_base": 0,
                    "obs_afectadas_cambio_base": 0,
                    "huecos_internos": 0,
                    "inicio_periodo": None,
                    "fin_periodo": None,
                    "media_bruta_pp": np.nan,
                    "sd_bruta_pp": np.nan,
                    "media_limpia_pp": np.nan,
                    "sd_limpia_pp": np.nan,
                    "mediana_limpia_pp": np.nan,
                    "min_limpia_pp": np.nan,
                    "max_limpia_pp": np.nan,
                }
            )

    return target_df, pd.DataFrame(horizon_rows)


def build_exclusions_and_methodological_reports(
    long_df: pd.DataFrame, target_df: pd.DataFrame, cal_df: pd.DataFrame, audit_stats: Dict[str, int]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(v) Registro de exclusiones y (vi) reporte de cambios metodológicos y anomalías."""
    # (v) Registro de exclusiones detallado
    excl_rows: List[Dict[str, object]] = []
    excl_rows.append(
        {
            "nivel": "archivo_ancho",
            "industria": "todas",
            "criterio": "filas_duplicadas_exactas_1994m1_2024m6",
            "cantidad": audit_stats["dropped_duplicate_wide_rows"],
            "detalle": "3.131 pares (industry, vintage) duplicados exactamente en monthly_gdp_vintages*.csv; deduplicados sin pérdida de información.",
        }
    )
    excl_rows.append(
        {
            "nivel": "observacion_larga",
            "industria": "todas",
            "criterio": "target_date_mayor_vintage_date",
            "cantidad": audit_stats["future_cells_discarded"],
            "detalle": "Verificación de ausencia de fuga de información futura.",
        }
    )
    for tag, h in [("H11", PRIMARY_HORIZON), ("H12", ALT_HORIZON)]:
        vc = target_df[f"estado_{tag}"].value_counts()
        for estado, cnt in vc.items():
            if estado != "incluido_limpio":
                periods = target_df.loc[target_df[f"estado_{tag}"] == estado, "target_period"].tolist()
                sample_txt = ", ".join(periods[:5]) + (" ..." if len(periods) > 5 else "")
                excl_rows.append(
                    {
                        "nivel": f"objetivo_gdp_{tag}",
                        "industria": "gdp",
                        "criterio": estado,
                        "cantidad": int(cnt),
                        "detalle": f"Horizonte H={h} meses. Ejemplos: {sample_txt}",
                    }
                )

    # Restricciones sectoriales
    all_gdp_v = set(long_df[long_df["industry"] == "gdp"]["vintage"])
    for ind, grp in long_df.groupby("industry"):
        ind_v = set(grp["vintage"])
        missing_v = sorted(
            all_gdp_v - ind_v,
            key=lambda s: tuple(int(x) for x in s.split("m")),
        )
        if missing_v:
            excl_rows.append(
                {
                    "nivel": "cobertura_sectorial",
                    "industria": ind,
                    "criterio": "vintages_ausentes_respecto_a_gdp",
                    "cantidad": len(missing_v),
                    "detalle": f"Inicio vintage {grp['vintage_date'].min().strftime('%Y-%m')}; faltan p. ej. {', '.join(missing_v[:6])}",
                }
            )

    exclusions_df = pd.DataFrame(excl_rows)

    # (vi) Reporte de cambios metodológicos y saltos por vintage
    gdp = long_df[long_df["industry"] == "gdp"].sort_values(["target_date", "vintage_date"]).copy()
    gdp["prev_value"] = gdp.groupby("target_date")["value"].shift(1)
    gdp["prev_vintage"] = gdp.groupby("target_date")["vintage"].shift(1)
    gdp["step_revision"] = gdp["value"] - gdp["prev_value"]

    jumps = gdp[gdp["step_revision"].abs() >= 1.5].sort_values("step_revision", key=abs, ascending=False).copy()
    jumps["tipo_evento"] = np.where(
        jumps["vintage"].isin(BASE_YEAR_VINTAGES),
        "cambio_anio_base",
        np.where(
            ((jumps["target_period"] == "2004m1") & (jumps["vintage"].isin(["2004m5", "2004m6"])))
            | ((jumps["target_period"] == "1998m11") & (jumps["vintage"].isin(["1999m2", "1999m3"])))
            | ((jumps["target_period"] == "2007m11") & (jumps["vintage"].isin(["2008m6", "2008m8"]))),
            "anomalia_fuente_puntual",
            "revision_ordinaria_o_benchmark_grande",
        ),
    )
    method_cols = [
        "target_period",
        "prev_vintage",
        "vintage",
        "prev_value",
        "value",
        "step_revision",
        "benchmark_revision",
        "base_year",
        "is_base_year_sentinel",
        "tipo_evento",
    ]
    method_df = jumps[method_cols].reset_index(drop=True)

    # Resumen por sector para tabla de auditoría sectorial
    sec_rows: List[Dict[str, object]] = []
    for ind, grp in long_df.groupby("industry"):
        sec_rows.append(
            {
                "industry": ind,
                "observaciones_largas": len(grp),
                "vintages_disponibles": grp["vintage_date"].nunique(),
                "inicio_vintage": grp["vintage_date"].min().strftime("%Y-%m"),
                "fin_vintage": grp["vintage_date"].max().strftime("%Y-%m"),
                "inicio_target": grp["target_date"].min().strftime("%Y-%m"),
                "fin_target": grp["target_date"].max().strftime("%Y-%m"),
                "celdas_centinela_base": int(grp["is_base_year_sentinel"].sum()),
                "extremos_abs_gt_50": int((grp["value"].abs() > 50).sum()),
            }
        )
    sectors_df = pd.DataFrame(sec_rows).sort_values("industry").reset_index(drop=True)
    return exclusions_df, method_df, sectors_df


def generate_figures(long_df: pd.DataFrame, target_df: pd.DataFrame, horizon_df: pd.DataFrame) -> None:
    """Genera las 4 figuras diagnósticas y las copia a main_tex/images/."""
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_IMG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#444444", "axes.linewidth": 0.8})

    # 1. Triángulo de revisiones respecto a v0 (primeros 12 meses desde v0, PBI 2005-2024)
    gdp = long_df[long_df["industry"] == "gdp"].sort_values(["target_date", "vintage_date"]).copy()
    v0_map = gdp.groupby("target_date", as_index=False).first()[["target_date", "vintage_date", "value"]].rename(
        columns={"vintage_date": "v0_date", "value": "y0"}
    )
    gdp = gdp.merge(v0_map, on="target_date", how="left")
    gdp["age_from_v0"] = month_diff(gdp["vintage_date"], gdp["v0_date"])
    gdp["rev_vs_v0"] = gdp["value"] - gdp["y0"]
    sub_tri = gdp[(gdp["target_date"] >= "2000-01-01") & (gdp["age_from_v0"].between(0, 12))]
    piv = sub_tri.pivot_table(index="target_date", columns="age_from_v0", values="rev_vs_v0", aggfunc="first")

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-2.0, vmax=2.0, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Revisión acumulada $y_t^{(v)} - y_t^{(v_0)}$ (p.p.)")
    yticks_idx = np.linspace(0, len(piv.index) - 1, 8, dtype=int)
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels([piv.index[i].strftime("%Y-%m") for i in yticks_idx])
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_xlabel("Meses transcurridos desde la primera publicación ($h = v - v_0$)")
    ax.set_ylabel("Período objetivo ($t$)")
    ax.set_title("Evolución de las revisiones del PBI mensual desde $v_0$ (2000–2025)")
    fig.tight_layout()
    fig.savefig(FIG_OUT_DIR / "triangulo_gdp.png", dpi=180)
    plt.close(fig)

    # 2. Dispersión y0 vs y_H11 (limpio vs saltos de año base)
    valid = target_df[(target_df["lag0_months"] <= 3) & target_df[f"Y_H{PRIMARY_HORIZON}"].notna()].copy()
    clean = valid[~valid[f"crosses_base_year_H{PRIMARY_HORIZON}"]]
    brk = valid[valid[f"crosses_base_year_H{PRIMARY_HORIZON}"]]

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=180)
    ax.scatter(clean["y0"], clean[f"y_H{PRIMARY_HORIZON}"], s=18, alpha=0.72, color="#1F4E79", label=f"Sin cambio de base (n={len(clean)})")
    ax.scatter(brk["y0"], brk[f"y_H{PRIMARY_HORIZON}"], s=32, marker="^", color="#B22222", label=f"Cruce de año base (n={len(brk)})")
    lims = [min(valid["y0"].min(), valid[f"y_H{PRIMARY_HORIZON}"].min()) - 2, max(valid["y0"].max(), valid[f"y_H{PRIMARY_HORIZON}"].max()) + 2]
    ax.plot(lims, lims, "--", color="#555555", linewidth=1.1, label="Recta 45° ($Y_t = 0$)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Primera publicación $y_t^{(v_0)}$ (%)")
    ax.set_ylabel(f"Publicación a horizonte $H={PRIMARY_HORIZON}$ meses (%)")
    ax.set_title(f"Primera estimación vs. revisión a horizonte fijo ($H={PRIMARY_HORIZON}$ meses)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIG_OUT_DIR / "scatter_revisiones_gdp.png", dpi=180)
    plt.close(fig)

    # 3. Histograma de Y_H11 (y_H - y_0) limpio vs con rupturas
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=180)
    bins = np.linspace(-6.0, 5.5, 32)
    ax.hist(clean[f"Y_H{PRIMARY_HORIZON}"], bins=bins, alpha=0.78, color="#1F4E79", edgecolor="white", label=f"Muestra limpia (n={len(clean)}, media={clean[f'Y_H{PRIMARY_HORIZON}'].mean():+.3f})")
    ax.hist(brk[f"Y_H{PRIMARY_HORIZON}"], bins=bins, alpha=0.75, color="#B22222", edgecolor="white", label=f"Cruce año base (n={len(brk)})")
    ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.axvline(clean[f"Y_H{PRIMARY_HORIZON}"].mean(), color="#E07B00", linestyle="-", linewidth=1.5, label=f"Media limpia ({clean[f'Y_H{PRIMARY_HORIZON}'].mean():+.3f} p.p.)")
    ax.set_xlabel(f"Revisión $Y_t^{{({PRIMARY_HORIZON})}} = y_t^{{(v_{{{PRIMARY_HORIZON}}})}} - y_t^{{(v_0)}}$ (p.p.)")
    ax.set_ylabel("Frecuencia (meses)")
    ax.set_title(f"Distribución de las revisiones del PBI a horizonte fijo ($H={PRIMARY_HORIZON}$)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIG_OUT_DIR / "hist_revisiones_gdp.png", dpi=180)
    plt.close(fig)

    # 4. Cobertura y estadísticos por horizonte H=3..12
    hdf = horizon_df[horizon_df["obs_brutas_con_v0"] > 0].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 4.0), dpi=180)
    x = np.arange(len(hdf))
    ax1.bar(x - 0.18, hdf["obs_limpias_sin_cambio_base"], width=0.36, color="#1F4E79", label="Limpias")
    ax1.bar(x + 0.18, hdf["obs_brutas_con_v0"], width=0.36, color="#A6A6A6", label="Brutas (con base)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"H={h}" for h in hdf["horizonte_H_meses"]])
    ax1.set_ylim(0, 455)
    ax1.set_ylabel("Observaciones")
    ax1.set_title("Cobertura efectiva por horizonte")
    for idx, row_h in hdf.reset_index(drop=True).iterrows():
        if row_h["huecos_internos"] > 0:
            ax1.text(idx, row_h["obs_brutas_con_v0"] + 8, f"{int(row_h['huecos_internos'])} huecos", ha="center", fontsize=7.5, color="#B22222")
    ax1.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    ax2.errorbar(
        hdf["horizonte_H_meses"],
        hdf["media_limpia_pp"],
        yerr=hdf["sd_limpia_pp"],
        fmt="o-",
        color="#1F4E79",
        ecolor="#7FA3C4",
        capsize=4,
        label="Limpia (media ± DE)",
    )
    ax2.plot(hdf["horizonte_H_meses"], hdf["media_bruta_pp"], "s--", color="#B22222", label="Bruta (media)")
    ax2.axhline(0.0, color="#444444", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("Horizonte $H$ (meses desde $v_0$)")
    ax2.set_ylabel("Revisión $y_H - y_0$ (p.p.)")
    ax2.set_title("Media y dispersión según horizonte $H$")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIG_OUT_DIR / "metricas_horizonte_gdp.png", dpi=180)
    plt.close(fig)

    for png in FIG_OUT_DIR.glob("*.png"):
        shutil.copy2(png, TEX_IMG_DIR / png.name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inv_df = build_source_inventory(ROOT_DIR)
    dict_df = build_data_dictionary()
    long_df, meta_d, v_raw_d, audit_stats = load_and_audit_vintages()
    cal_df = build_publication_calendar(long_df, meta_d)
    target_df, horizon_df = build_gdp_target_dataset(long_df)
    excl_df, method_df, sectors_df = build_exclusions_and_methodological_reports(long_df, target_df, cal_df, audit_stats)

    # Guardar matriz de vintages (triángulo ancho deduplicado de GDP)
    gdp_wide = (
        v_raw_d[v_raw_d["industry"] == "gdp"]
        .assign(v_date=lambda d: parse_ym_series(d["vintage"]))
        .sort_values("v_date")
        .drop(columns=["v_date"])
    )

    outputs: Dict[str, pd.DataFrame] = {
        "inventario_fuentes.csv": inv_df,
        "diccionario_variables.csv": dict_df,
        "vintages_long.csv": long_df,
        "triangulo_gdp.csv": gdp_wide,
        "fechas_publicacion.csv": cal_df,
        "retraso_publicacion_gdp.csv": cal_df[["vintage", "vintage_date", "max_target_date", "pub_lag_months"]],
        "muestra_objetivo_gdp.csv": target_df,
        "resumen_horizontes_gdp.csv": horizon_df,
        "registro_exclusiones.csv": excl_df,
        "reporte_cambios_metodologicos.csv": method_df,
        "cobertura_sectorial.csv": sectors_df,
    }
    for fname, df_out in outputs.items():
        df_out.to_csv(OUT_DIR / fname, index=False)

    generate_figures(long_df, target_df, horizon_df)

    # Manifiesto SHA-256 y resumen JSON reproducible
    checksum_rows = [
        {"tipo": "salida_auditoria", "archivo": p.name, "tamano_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(OUT_DIR.glob("*.csv"))
    ] + [
        {"tipo": "figura", "archivo": f"figures/{p.name}", "tamano_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(FIG_OUT_DIR.glob("*.png"))
    ]
    pd.DataFrame(checksum_rows).to_csv(OUT_DIR / "checksums_sha256.csv", index=False)

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_stats": audit_stats,
        "horizon_summary": horizon_df.to_dict(orient="records"),
    }
    (OUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("Etapa 1 completada. Resumen:")
    print(json.dumps(audit_stats, indent=2))
    print("\nResumen por horizonte:")
    print(horizon_df.to_string(index=False))


if __name__ == "__main__":
    main()
