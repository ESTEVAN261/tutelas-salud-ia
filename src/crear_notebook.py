"""Construye notebooks/Fase2_Modelado.ipynb a partir de las funciones de src/fase2_modelado.py."""
import nbformat as nbf
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md("# Fase 2 — Desarrollo y Modelado\n"
       "**Predicción de riesgo de vencimiento de términos legales en tutelas de salud**\n\n"
       "Universidad Popular del Cesar · Inteligencia Artificial · 2026-2  \n"
       "Integrantes: Enor Antonio Acosta Ricardo · Estevan Yesid Sánchez Velasquez\n\n"
       "Recorre las fases 04 a 11 del ciclo de vida de ML sobre el dataset del equipo "
       "(`dataset_tutelas_salud_sintetico_5000.csv`, 5.000 tutelas, 25 columnas)."),
    code("import sys, os\nfrom pathlib import Path\n"
         "RAIZ = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "os.chdir(RAIZ); sys.path.insert(0, str(RAIZ))\n"
         "import numpy as np, pandas as pd\nfrom src import fase2_modelado as f2\n"
         "pd.set_option('display.max_columns', 30)"),
    md("## 1. Carga del dataset"),
    code("df_raw = pd.read_csv(f2.DATOS, encoding='utf-8-sig')\nprint(df_raw.shape)\ndf_raw.head()"),
    md("## 2. Limpieza y validación (fase 04)\nNulos, duplicados, tiempos imposibles, consistencia de las columnas "
       "calculadas (margen, % de plazo, nivel de riesgo) y valores atípicos."),
    code("df = f2.limpiar(df_raw)\nf2.bitacora['limpieza']"),
    md("## 3. Ingeniería de características (fase 05)\n"
       "Codificación ordinal de urgencia, complejidad y carga; binarias Sí/No; variables de fecha; interacción "
       "complejidad × carga.\n\n**Excluidas por fuga del objetivo** (se conocen solo después de la respuesta): "
       f"{', '.join('`'+c+'`' for c in ['fecha_respuesta_simulada','margen_dias','porcentaje_plazo_consumido','vencio_termino','nivel_riesgo_real','dias_retraso','fallo_primera_instancia'])}."),
    code("df = f2.crear_caracteristicas(df)\ndf[f2.NUM + f2.CAT].head()"),
    md("## 4. División 70 / 15 / 15 (fase 06), estratificada por nivel de riesgo"),
    code("train, val, test = f2.dividir(df)\nf2.bitacora['particion']"),
    code("X = lambda d: d[f2.NUM + f2.CAT]\n"
         "Xtr, ytr, Xva, yva, Xte, yte = X(train), train[f2.OBJETIVO], X(val), val[f2.OBJETIVO], X(test), test[f2.OBJETIVO]"),
    md("## 5. Modelos y ajuste de hiperparámetros (fases 07–09)\n"
       "M0 Dummy · M1 Regresión lineal (baseline) · M2 Ridge ajustado · M3a/M3b Random Forest · M4 XGBoost ajustado.  \n"
       "`RandomizedSearchCV`, validación cruzada 5-fold sobre entrenamiento (≈ 1 minuto)."),
    code("modelos, b_xgb = f2.entrenar_modelos(Xtr, ytr)\nf2.bitacora['hiperparametros']"),
    md("## 6. Evaluación y comparación (fase 10)"),
    code("filas = []\nfor n, m in modelos.items():\n"
         "    tr, va = f2.metricas(ytr, m.predict(Xtr)), f2.metricas(yva, m.predict(Xva))\n"
         "    filas.append({'modelo': n, **{f'{k}_train': v for k, v in tr.items()}, **{f'{k}_val': v for k, v in va.items()}})\n"
         "comp = pd.DataFrame(filas); comp.round(3)"),
    code("cand = comp[~comp.modelo.str.startswith('M0')]\nmejor_nombre = cand.loc[cand.RMSE_val.idxmin(), 'modelo']\n"
         "mejor = modelos[mejor_nombre]; p_te = mejor.predict(Xte)\nprint('Seleccionado:', mejor_nombre)\n"
         "for n in ['M1 · Regresión lineal (baseline)', 'M4 · XGBoost (ajustado)', mejor_nombre]:\n"
         "    print(n, {k: round(v, 3) for k, v in f2.metricas(yte, modelos[n].predict(Xte)).items()})"),
    md("## 7. Del tiempo predicho al riesgo y a la alerta (fase 11)\n"
       "`razón = tiempo predicho ÷ plazo`. Niveles del dataset: Bajo ≤ 0,60 < Medio ≤ 0,85 < Alto ≤ 1,00 < Vencido.  \n"
       "Alerta estándar: razón > 1,00. Alerta conservadora: razón ≥ 0,95 (umbral elegido en validación para recall ≥ 0,85)."),
    code("from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score\n"
         "pz = test.plazo_informe_dias.to_numpy(); venc = (yte.to_numpy() / pz) > 1\n"
         "print('AUC vencimiento:', round(roc_auc_score(venc, p_te / pz), 3))\n"
         "for u in [1.0001, 0.95]:\n"
         "    a = (p_te / pz) >= u\n"
         "    print(f'umbral {u:.2f}: recall {recall_score(venc, a):.2f} · precisión {precision_score(venc, a):.2f} · alertas {a.sum()}')\n"
         "pd.DataFrame(confusion_matrix(test.nivel_riesgo_real, f2.nivel_riesgo(p_te, pz), labels=f2.NIVELES),\n"
         "             index=['real '+n for n in f2.NIVELES], columns=['pred '+n for n in f2.NIVELES])"),
    md("## 8. Gráficas (generadas por `python src/fase2_modelado.py` en `results/figuras/`)"),
    code("from IPython.display import Image, display\n"
         "for f in sorted(Path('results/figuras').glob('*.png')):\n    print(f.name); display(Image(filename=str(f)))"),
    md("## 9. Conclusiones preliminares\n"
       "- Todos los modelos reducen el error del Dummy en ~32 %; R² ≈ 0,48 en validación y 0,55 en prueba.\n"
       "- Lineal, Ridge y XGBoost empatan: los efectos en estos datos son aditivos (XGBoost elige árboles de profundidad 1).\n"
       "- Random Forest por defecto sobreajusta (RMSE train 0,21 vs. validación 0,56).\n"
       "- Se selecciona Ridge: igual de preciso y el más simple de explicar.\n"
       "- Para el objetivo operativo (anticipar vencimientos) el modelo discrimina bien: AUC 0,92; la regla conservadora detecta 88 % de las tutelas que vencen."),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, RAIZ / "notebooks" / "Fase2_Modelado.ipynb")
print("ok")
