# Predicción de revisiones del PBI peruano en tiempo real (Econometría, ML y QRC)

Repositorio de investigación sobre la estructura sistemática y predictibilidad fuera de muestra de las revisiones de las primeras estimaciones del Producto Bruto Interno (PBI) mensual del Perú (`peru_gdp_rtd`), comparando benchmarks econométricos y de aprendizaje automático clásicos frente a arquitecturas de reservorio clásicas (ESN) y cuánticas (QRC).

## Estado del proyecto

| Etapa | Descripción | Estado | Scripts / Salidas principales |
| :--- | :--- | :--- | :--- |
| **Etapa 1** | Auditoría reproducible de *vintages* y cambios de base | Completada | `scripts/etapa1_auditoria_vintages.py` → `output/analysis_output/` |
| **Etapa 2** | Caracterización econométrica (sesgo, colas, ARCH, Mincer-Zarnowitz, RESET) | Completada | `scripts/etapa2_caracterizacion_econometrica.py` → `output/etapa2_econometria/` |
| **Etapas 3 y 5** | Referencias clásicas en validación expansiva causal en tiempo real | Completada | `scripts/rtd_eval_utils.py`, `scripts/etapa3_modelos_clasicos.py` → `output/etapa3_clasicos/` |
| **Etapa 8 (parcial)** | Inferencia DM-HLN, Clark-West y bootstrap por bloques sobre referentes clásicos | Completada | `output/etapa3_clasicos/etapa3_comparacion_estadistica.csv`, `etapa3_clark_west.csv` |
| **Etapa 4** | Modelos de reservorio clásico (ESN) y cuántico (QRC) | Pendiente | — |
| **Etapa 6** | Experimento sintético calibrado | Pendiente | — |
| **Etapa 7** | Controles de capacidad equivalente e inferencia final integrada | Pendiente | — |

El informe técnico completo y compilado se encuentra en `main_tex/main.tex` y `main_tex/main.pdf`.

## Hallazgos verificados hasta la fecha

- **Etapa 1:** sobre 44 307 observaciones industria-*vintage* en nueve industrias (1994–2025), tras descontar cruces de año base (`2000m7` y `2014m3`), el horizonte continuo principal $H=11$ meses reúne $n=350$ observaciones limpias (`0` huecos internos) frente a $n=331$ en $H=12$ (`18` huecos).
- **Etapa 2:** a horizonte $H=11$, la revisión $Y_t^{(11)} = y_t^{(v_{11})} - y_t^{(v_0)}$ presenta media positiva robusta de $+0{,}347$ p.p. ($\text{SE}_{\text{NW}}=0{,}0638$, $p=5{,}3\times 10^{-8}$), colas pesadas (asimetría $1{,}258$; curtosis en exceso $6{,}293$), persistencia anual observable ($\hat{\rho}_{12}=0{,}386$) y no linealidad (RESET $p=0{,}00096$) concentrada en el período COVID/post-COVID.
- **Etapas 3 y 5 ($n=250$ orígenes OOS, `2003m2`–`2024m11`; bloque congelado final `2019m12`–`2024m11`, $n=60$):**
  - En el tramo **pre-COVID** (`2003m2`–`2019m11`, $n=190$), `Ridge` encabeza los referentes clásicos ($\text{RMSE}=0{,}503$, $R^2_{\text{OOS}}=+0{,}094$) y supera significativamente a la corrección por media histórica expansiva ($\text{DM}_{\text{HLN}}=-2{,}724$, $p=0{,}0071$).
  - En el **bloque congelado final** ($n=60$), las especificaciones lineales con $y_0$ sufren extrapolación severa durante los choques extremos del COVID-19 (`Ridge` $\text{RMSE}=0{,}731$), mientras los árboles impulsados acotados en NumPy puro (`GBDT_NumPy`, implementado ante la ausencia de `xgboost` en el entorno) alcanzan $\text{RMSE}=0{,}349$ ($R^2_{\text{OOS}}=+0{,}087$) y $\text{RMSE}=0{,}495$ global, seguidos de `AR_BIC` ($\text{RMSE}=0{,}498$ global).

## Estructura de directorios

- `input/`, `raw/`, `peru_gdp_rtd_output/`: insumos originales y salidas de *vintages* en tiempo real del PBI peruano.
- `peruvian_gdp_revisions-main/`: código base de construcción del dataset *real-time* de revisiones.
- `scripts/`: scripts reproducibles de análisis y modelado (`etapa1_*`, `etapa2_*`, `etapa3_*`, `rtd_eval_utils.py`).
- `output/`: tablas CSV, resúmenes JSON, gráficos PNG y checksums SHA-256 de cada etapa.
- `main_tex/`: documento LaTeX principal (`main.tex`), bibliografía (`bibliographie.bib`), figuras (`images/`) y PDF compilado (`main.pdf`).

## Reproducción rápida

```bash
python3 scripts/etapa1_auditoria_vintages.py
python3 scripts/etapa2_caracterizacion_econometrica.py
python3 scripts/etapa3_modelos_clasicos.py
```

Compilación del informe:

```bash
cd main_tex
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
