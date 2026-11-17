#!/usr/bin/env bash
# Compila individualmente todos los manuales APA en teoria/
set -euo pipefail
cd "$(dirname "$0")"

for f in 01_datos_tiempo_real_y_lectura 02_herramienta_ocr_notas_semanales 03_pipeline_bcrp_scraping_y_limpieza 04_teoria_econometrica_revisiones 05_modelos_clasicos_y_ml 06_reservorios_esn_y_qrc 07_validacion_causal_e_inferencia; do
  echo "=== Compilando ${f}.tex ==="
  pdflatex -interaction=nonstopmode "${f}.tex" > /dev/null
  biber "${f}" > /dev/null
  pdflatex -interaction=nonstopmode "${f}.tex" > /dev/null
  pdflatex -interaction=nonstopmode "${f}.tex" > /dev/null
  if grep -nE "^!|Overfull|Undefined|Warning" "${f}.log" | grep -v "First Aid" ; then
    echo "ADVERTENCIA o ERROR en ${f}.log"
  else
    echo "OK limpio: ${f}.pdf"
  fi
done
