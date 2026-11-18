# Predicción de revisiones del PBI peruano en tiempo real

Cuando se publica por primera vez el crecimiento mensual del Producto Bruto Interno (PBI) del Perú, la cifra es preliminar y suele corregirse en los meses siguientes conforme llegan más encuestas y registros. Este repositorio analiza si esas correcciones posteriores siguen patrones predecibles y compara si un modelo de **computación de reservorio cuántico (QRC)** logra predecirlas mejor que los modelos econométricos y de aprendizaje automático clásicos, respetando en cada mes únicamente la información disponible en ese momento.

El informe completo de resultados está compilado en [`main_tex/main.pdf`](main_tex/main.pdf) y los manuales explicativos de cada herramienta están en [`teoria/`](teoria/).

---

## ¿Qué se hizo en cada etapa? (explicación sencilla)

| Etapa | ¿Qué se hizo? | Resultado principal | Código y salidas |
| :--- | :--- | :--- | :--- |
| **Etapa 1: Limpieza y revisión de datos** | Se revisaron 44 307 registros históricos del PBI peruano (1994–2025), se quitaron filas repetidas y se separaron los años en los que cambió la metodología de cálculo (2000 y 2014). | Quedaron **350 meses limpios** comparando la primera cifra con la publicada 11 meses después. La primera estimación suele quedarse por debajo en **+0,35 puntos porcentuales**. | `scripts/etapa1_auditoria_vintages.py` → `output/analysis_output/` |
| **Etapa 2: Patrones estadísticos** | Se analizó si las revisiones eran puro azar o seguían reglas detectables antes de modelar. | No son azar: el primer trimestre se revisa más al alza, existe un patrón anual repetible (correlación anual `0,386`) y hubo saltos inusuales durante la pandemia. | `scripts/etapa2_caracterizacion_econometrica.py` → `output/etapa2_econometria/` |
| **Etapas 3 y 5: Modelos clásicos sin ver el futuro** | Se probaron 8 modelos clásicos (promedios históricos, autorregresivos, Ridge, factores sectoriales y árboles de decisión) prediciendo mes a mes entre 2003 y 2024, reservando los últimos 60 meses para la prueba final. | Antes de la pandemia (2003–2019), **Ridge** fue el mejor modelo lineal; pero durante el COVID-19 (2020–2021) falló al proyectar cifras extremas. Los **árboles acotados (GBDT)** evitaron ese colapso. | `scripts/rtd_eval_utils.py`, `scripts/etapa3_modelos_clasicos.py` → `output/etapa3_clasicos/` |
| **Etapa 4: Modelo cuántico vs. controles clásicos** | Se simuló un reservorio cuántico de 5 qubits y se comparó en igualdad de condiciones contra modelos clásicos no lineales del mismo tamaño (redes ESN y características aleatorias RFF). | El modelo cuántico contuvo bien los choques de la pandemia, pero **no superó a sus equivalentes clásicos**: empató con ESN y quedó detrás de **RFF** en precisión global. | `scripts/etapa4_reservorios_qrc.py` → `output/etapa4_reservorios/` |
| **Etapa 6: Simulaciones sintéticas y potencia** | Se generaron 12 escenarios artificiales parecidos a los datos peruanos para ver en qué condiciones ganaría el modelo cuántico, y se midió cuánta diferencia se necesita para detectarla con certeza. | El modelo cuántico **no ganó en ninguno de los 12 escenarios**, y con 250 meses reales solo se distinguen mejoras grandes (5,7% a 36,1%), no ventajas pequeñas del 2%. | `scripts/etapa6_experimento_sintetico.py` → `output/etapa6_sintetico/` |
| **Etapa 7: Evaluación económica** | Según la regla definida antes de empezar, solo se calcularían beneficios económicos si el modelo cuántico superaba claramente a los modelos clásicos. | **No se activó**: al no haber superioridad cuántica demostrada, se detuvo para no inventar ganancias económicas artificiales. | Verificación en `output/etapa8_inferencia/etapa8_verificacion_etapa7_gate.json` |
| **Etapa 8: Pruebas estadísticas finales y cierre** | Se aplicaron pruebas conjuntas entre los 12 modelos para evitar falsos positivos y se verificó la integridad de todos los archivos generados. | Ningún contraste global favorece al modelo cuántico tras corregir por comparaciones múltiples, y **RFF** lo supera antes de la pandemia. Los **52 archivos** finales pasan verificación SHA-256 sin errores. | `scripts/etapa8_inferencia_y_cierre.py` → `output/etapa8_inferencia/` |

---

## Conclusión en breve

1. **Las revisiones del PBI peruano son sistemáticas**, no puro ruido: en promedio se corrigen alrededor de **+0,35 puntos porcentuales** hacia arriba al cabo de once meses.
2. **Los modelos clásicos simples y acotados resuelven mejor el problema**: **RFF** obtuvo el menor error global (`RMSE = 0,456`), los **árboles acotados** lideraron durante la pandemia (`RMSE = 0,349`), y la regresión **Ridge** funcionó muy bien antes del COVID-19 (`RMSE = 0,503`), aunque falló frente a extremos pandémicos.
3. **El reservorio cuántico no aporta ventaja adicional hoy**: logró buen desempeño en pandemia (`RMSE = 0,363`), pero fue igualado o superado por alternativas clásicas más rápidas y sencillas.

---

## Estructura de carpetas

- `input/`, `raw/`, `peru_gdp_rtd_output/`: archivos originales del PBI en tiempo real (conservados sin modificaciones).
- `peruvian_gdp_revisions-main/`: código original de recolección y construcción de la base de revisiones del BCRP.
- `scripts/`: programas en Python para reproducir cada etapa de principio a fin.
- `output/`: tablas en CSV, resúmenes en JSON, gráficos y sumas de verificación SHA-256.
- `main_tex/`: informe completo en LaTeX (`main.tex` separado por capítulos en `sections/`, bibliografía en `bibliographie.bib` y documento final en `main.pdf`).
- `teoria/`: siete guías didácticas independientes en formato APA 7 sobre cómo funcionan los datos, modelos y pruebas usadas.

---

## Cómo reproducir los cálculos y el PDF

Ejecutar cada etapa en orden desde la raíz del proyecto:

```bash
python3 scripts/etapa1_auditoria_vintages.py
python3 scripts/etapa2_caracterizacion_econometrica.py
python3 scripts/etapa3_modelos_clasicos.py
python3 scripts/etapa4_reservorios_qrc.py
python3 scripts/etapa6_experimento_sintetico.py
python3 scripts/etapa8_inferencia_y_cierre.py
```

Para volver a compilar el informe en PDF:

```bash
cd main_tex
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
