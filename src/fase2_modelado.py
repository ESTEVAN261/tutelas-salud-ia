"""
FASE 2 — Desarrollo y Modelado
Predicción del tiempo de respuesta de tutelas en salud y nivel de riesgo de vencimiento.
Dataset: data/raw/dataset_tutelas_salud_sintetico_5000.csv (dataset del equipo, Fase 1).

Etapas (ciclo de vida de ML visto en clase):
  04 Limpieza y validación         -> limpiar()
  05 Ingeniería de características -> crear_caracteristicas()
  06 División 70/15/15             -> dividir()
  07-09 Algoritmos, entrenamiento y ajuste de hiperparámetros -> entrenar_modelos()
  10 Evaluación (MAE, RMSE, R²)    -> main()
  11 Generalización y lógica de riesgo (Bajo/Medio/Alto/Vencido) -> main()

Uso:  python src/fase2_modelado.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (confusion_matrix, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "raw" / "dataset_tutelas_salud_sintetico_5000.csv"
RES, FIG = RAIZ / "results", RAIZ / "results" / "figuras"
for p in [FIG, RAIZ / "models", RAIZ / "data" / "processed"]:
    p.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
AZUL, NARANJA, GRIS, ROJO = "#1f4e79", "#e07a1f", "#8a8a8a", "#b3261e"

OBJETIVO = "tiempo_respuesta_dias"
# Columnas que se conocen DESPUÉS de la respuesta o se calculan con el tiempo real: fuga del objetivo
FUGA = ["fecha_respuesta_simulada", "margen_dias", "porcentaje_plazo_consumido", "vencio_termino",
        "nivel_riesgo_real", "dias_retraso", "fallo_primera_instancia"]
CAT = ["departamento", "zona", "tipo_entidad_accionada", "hecho_principal", "grupo_edad", "grupo_diagnostico"]
NUM = ["urgencia_ord", "complejidad_ord", "carga_ord", "complejidad_x_carga", "plazo_informe_dias",
       "numero_adjuntos", "solicitudes_previas", "sujeto_especial", "diagnostico_si", "negacion_previa_si",
       "mes", "dia_semana", "es_fin_semana"]
NIVELES = ["Bajo", "Medio", "Alto", "Vencido"]
bitacora = {}


def nivel_riesgo(tiempo, plazo, u_medio=0.60, u_alto=0.85, u_venc=1.0):
    """Regla de negocio del dataset: % del plazo consumido -> Bajo / Medio / Alto / Vencido."""
    r = np.asarray(tiempo, dtype=float) / np.asarray(plazo, dtype=float)
    return np.select([r > u_venc, r > u_alto, r > u_medio], ["Vencido", "Alto", "Medio"], "Bajo")


# ---------------------------------------------------------------- 04 Limpieza y validación
def limpiar(df):
    log = {"filas": len(df), "columnas": df.shape[1]}
    df = df.copy()
    df.columns = df.columns.str.replace("﻿", "", regex=False).str.strip()
    for c in df.select_dtypes("object"):
        df[c] = df[c].str.strip()
    log["nulos"] = int(df.isna().sum().sum())
    log["duplicados_exactos"] = int(df.duplicated().sum())
    log["ids_duplicados"] = int(df["id_tutela"].duplicated().sum())
    df = df.drop_duplicates()
    # Validaciones de consistencia interna
    log["tiempos_no_positivos"] = int((df[OBJETIVO] <= 0).sum())
    log["margen_inconsistente"] = int(((df["plazo_informe_dias"] - df[OBJETIVO] - df["margen_dias"]).abs() > 0.011).sum())
    pct = df[OBJETIVO] / df["plazo_informe_dias"]
    log["porcentaje_inconsistente"] = int(((pct - df["porcentaje_plazo_consumido"]).abs() > 0.001).sum())
    log["riesgo_inconsistente"] = int((nivel_riesgo(df[OBJETIVO], df["plazo_informe_dias"]) != df["nivel_riesgo_real"]).sum())
    q1, q3 = df[OBJETIVO].quantile([0.25, 0.75])
    log["atipicos_1_5_iqr"] = int((df[OBJETIVO] > q3 + 1.5 * (q3 - q1)).sum())
    log["atipicos_3_iqr"] = int((df[OBJETIVO] > q3 + 3 * (q3 - q1)).sum())
    log["umbral_3_iqr"] = float(q3 + 3 * (q3 - q1))
    log["vencidos_entre_atipicos_3_iqr"] = int(((df[OBJETIVO] > q3 + 3 * (q3 - q1)) & (df["vencio_termino"] == "Si")).sum())
    df = df[df[OBJETIVO] > 0]
    log["filas_finales"] = len(df)
    bitacora["limpieza"] = log
    return df


# ------------------------------------------------- 05 Ingeniería de características
def crear_caracteristicas(df):
    df = df.copy()
    f = pd.to_datetime(df["fecha_radicacion"])
    df["mes"], df["dia_semana"] = f.dt.month, f.dt.dayofweek
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["urgencia_ord"] = df["nivel_urgencia"].map({"Baja": 0, "Media": 1, "Alta": 2, "Critica": 3})
    df["complejidad_ord"] = df["complejidad_caso"].map({"Baja": 0, "Media": 1, "Alta": 2})
    df["carga_ord"] = df["carga_operativa"].map({"Baja": 0, "Media": 1, "Alta": 2})
    df["complejidad_x_carga"] = df["complejidad_ord"] * df["carga_ord"]
    df["sujeto_especial"] = (df["sujeto_especial_proteccion"] == "Si").astype(int)
    df["diagnostico_si"] = (df["diagnostico_mencionado"] == "Si").astype(int)
    df["negacion_previa_si"] = (df["negacion_previa"] == "Si").astype(int)
    return df


# ------------------------------------------------------------- 06 División
def dividir(df):
    train, temp = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["nivel_riesgo_real"])
    val, test = train_test_split(temp, test_size=0.50, random_state=SEED, stratify=temp["nivel_riesgo_real"])
    bitacora["particion"] = {"train": len(train), "val": len(val), "test": len(test)}
    return train, val, test


def prep(escalar):
    num = StandardScaler() if escalar else "passthrough"
    return ColumnTransformer([("num", num, NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])


def metricas(y, p):
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


# ------------------------------------------------ 07-09 Modelos y ajuste
def entrenar_modelos(Xtr, ytr):
    cv = KFold(5, shuffle=True, random_state=SEED)
    sc = "neg_root_mean_squared_error"
    m = {}
    m["M0 · Dummy (media)"] = Pipeline([("prep", prep(True)), ("modelo", DummyRegressor())]).fit(Xtr, ytr)
    m["M1 · Regresión lineal (baseline)"] = Pipeline([("prep", prep(True)), ("modelo", LinearRegression())]).fit(Xtr, ytr)
    b_ridge = RandomizedSearchCV(Pipeline([("prep", prep(True)), ("modelo", Ridge())]),
                                 {"modelo__alpha": loguniform(1e-2, 1e3)}, n_iter=20, cv=cv, scoring=sc,
                                 random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    m["M2 · Ridge (ajustado)"] = b_ridge.best_estimator_
    m["M3a · Random Forest (por defecto)"] = Pipeline([("prep", prep(False)), ("modelo", RandomForestRegressor(
        random_state=SEED, n_jobs=-1))]).fit(Xtr, ytr)
    b_rf = RandomizedSearchCV(Pipeline([("prep", prep(False)), ("modelo", RandomForestRegressor(random_state=SEED, n_jobs=-1))]),
                              {"modelo__n_estimators": [200, 400, 600], "modelo__max_depth": [4, 6, 8, 12, None],
                               "modelo__min_samples_leaf": [1, 5, 10, 20, 40], "modelo__max_features": [0.2, 0.4, 0.6, 1.0]},
                              n_iter=25, cv=cv, scoring=sc, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    m["M3b · Random Forest (ajustado)"] = b_rf.best_estimator_
    b_xgb = RandomizedSearchCV(Pipeline([("prep", prep(False)), ("modelo", XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist"))]),
                               {"modelo__n_estimators": randint(150, 900), "modelo__learning_rate": loguniform(0.005, 0.2),
                                "modelo__max_depth": randint(1, 7), "modelo__subsample": uniform(0.6, 0.4),
                                "modelo__colsample_bytree": uniform(0.4, 0.6), "modelo__min_child_weight": randint(1, 30),
                                "modelo__reg_lambda": loguniform(0.1, 30)},
                               n_iter=40, cv=cv, scoring=sc, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    m["M4 · XGBoost (ajustado)"] = b_xgb.best_estimator_
    lim = lambda d: {k.split("__")[-1]: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    bitacora["hiperparametros"] = {
        "Ridge": {"mejores": lim(b_ridge.best_params_), "rmse_cv": -b_ridge.best_score_, "combinaciones": 20},
        "RandomForest": {"mejores": lim(b_rf.best_params_), "rmse_cv": -b_rf.best_score_, "combinaciones": 25},
        "XGBoost": {"mejores": lim(b_xgb.best_params_), "rmse_cv": -b_xgb.best_score_, "combinaciones": 40}}
    pd.DataFrame(b_xgb.cv_results_).to_csv(RES / "busqueda_xgboost.csv", index=False)
    pd.DataFrame(b_rf.cv_results_).to_csv(RES / "busqueda_random_forest.csv", index=False)
    return m, b_xgb


def evaluar_alertas(y_real, pred, plazo, umbral):
    """Alerta = razón predicha >= umbral. Positivo real = la tutela venció (razón real > 1)."""
    real = (np.asarray(y_real) / plazo) > 1.0
    alerta = (np.asarray(pred) / plazo) >= umbral
    return {"umbral": umbral, "alertas": int(alerta.sum()), "vencidas_reales": int(real.sum()),
            "detectadas": int((real & alerta).sum()), "no_detectadas": int((real & ~alerta).sum()),
            "recall": float(recall_score(real, alerta)), "precision": float(precision_score(real, alerta)),
            "f1": float(f1_score(real, alerta))}


def main():
    df = crear_caracteristicas(limpiar(pd.read_csv(DATOS, encoding="utf-8-sig")))
    train, val, test = dividir(df)
    pd.concat([train.assign(particion="train"), val.assign(particion="val"), test.assign(particion="test")]).to_csv(
        RAIZ / "data" / "processed" / "tutelas_preprocesado.csv", index=False)
    bitacora["objetivo"] = {k: float(v) for k, v in df[OBJETIVO].describe().items()}
    bitacora["riesgo_real_global"] = df["nivel_riesgo_real"].value_counts().to_dict()
    X = lambda d: d[NUM + CAT]
    Xtr, ytr, Xva, yva, Xte, yte = X(train), train[OBJETIVO], X(val), val[OBJETIVO], X(test), test[OBJETIVO]

    modelos, b_xgb = entrenar_modelos(Xtr, ytr)

    # ---------------------------------------------------------- 10 Evaluación
    filas = []
    for n, m in modelos.items():
        tr, va = metricas(ytr, m.predict(Xtr)), metricas(yva, m.predict(Xva))
        filas.append({"modelo": n, **{f"{k}_train": v for k, v in tr.items()}, **{f"{k}_val": v for k, v in va.items()}})
    comp = pd.DataFrame(filas)
    comp.to_csv(RES / "comparacion_modelos.csv", index=False)
    print(comp.round(3).to_string(index=False))

    candidatos = comp[~comp.modelo.str.startswith("M0")]
    mejor_nombre = candidatos.loc[candidatos["RMSE_val"].idxmin(), "modelo"]
    mejor, base = modelos[mejor_nombre], modelos["M1 · Regresión lineal (baseline)"]
    p_te, p_va = mejor.predict(Xte), mejor.predict(Xva)
    fin = {"mejor_modelo": mejor_nombre, "test_mejor": metricas(yte, p_te), "test_baseline": metricas(yte, base.predict(Xte)),
           "test_dummy": metricas(yte, modelos["M0 · Dummy (media)"].predict(Xte)), "train_mejor": metricas(ytr, mejor.predict(Xtr))}
    fin["mejora_rmse_vs_baseline_pct"] = 100 * (1 - fin["test_mejor"]["RMSE"] / fin["test_baseline"]["RMSE"])
    fin["mejora_rmse_vs_dummy_pct"] = 100 * (1 - fin["test_mejor"]["RMSE"] / fin["test_dummy"]["RMSE"])
    fin["test_todos"] = {n: metricas(yte, m.predict(Xte)) for n, m in modelos.items()}
    fin["mae_por_plazo"] = {}
    for pz in [1, 2, 3]:
        mk = (test["plazo_informe_dias"] == pz).to_numpy()
        fin["mae_por_plazo"][f"{pz} dia(s)"] = {"n": int(mk.sum()), "MAE_dias": float(mean_absolute_error(yte[mk], p_te[mk])),
                                               "MAE_horas": float(24 * mean_absolute_error(yte[mk], p_te[mk]))}
    bitacora["evaluacion_final"] = fin
    joblib.dump(mejor, RAIZ / "models" / "mejor_modelo.joblib")

    # ------------------------------------------- 11 Riesgo y alertas de vencimiento
    pz_te, pz_va = test["plazo_informe_dias"].to_numpy(), val["plazo_informe_dias"].to_numpy()
    venc_va, venc_te = (yva.to_numpy() / pz_va) > 1, (yte.to_numpy() / pz_te) > 1
    bitacora["auc_vencimiento"] = {"val": float(roc_auc_score(venc_va, p_va / pz_va)),
                                   "test": float(roc_auc_score(venc_te, p_te / pz_te))}
    # Umbral conservador elegido en VALIDACIÓN: el mayor umbral con recall >= 0.85
    grid = np.round(np.arange(0.50, 1.21, 0.01), 2)
    curva = [evaluar_alertas(yva, p_va, pz_va, u) for u in grid]
    pd.DataFrame(curva).to_csv(RES / "curva_umbral_validacion.csv", index=False)
    u_cons = max(c["umbral"] for c in curva if c["recall"] >= 0.85)
    bitacora["alertas"] = {"Regla estándar (razón > 1,00)": evaluar_alertas(yte, p_te, pz_te, 1.0001),
                           f"Regla conservadora (razón ≥ {u_cons:.2f})": evaluar_alertas(yte, p_te, pz_te, u_cons)}
    bitacora["umbral_conservador"] = float(u_cons)
    real4 = test["nivel_riesgo_real"].to_numpy()
    pred4 = nivel_riesgo(p_te, pz_te)
    bitacora["riesgo_4_niveles"] = {"matriz": confusion_matrix(real4, pred4, labels=NIVELES).tolist(),
                                    "exactitud": float((real4 == pred4).mean()),
                                    "exactitud_dummy_mayoritaria": float((real4 == "Vencido").mean())}
    # Modelo ingenuo: alerta para todos
    bitacora["alerta_todos"] = evaluar_alertas(yte, np.full(len(yte), 99.0), pz_te, 1.0)

    # Enfoque alternativo: clasificar directamente si la tutela vencerá (regresión logística)
    from sklearn.linear_model import LogisticRegression
    clf = Pipeline([("prep", prep(True)), ("modelo", LogisticRegression(max_iter=2000, C=1.0))])
    clf.fit(Xtr, (ytr.to_numpy() / train["plazo_informe_dias"].to_numpy()) > 1)
    pr_va, pr_te = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    u_clf = max(u for u in np.round(np.arange(0.05, 0.96, 0.01), 2) if recall_score(venc_va, pr_va >= u) >= 0.85)
    a = pr_te >= u_clf
    bitacora["clasificador_directo"] = {
        "auc_test": float(roc_auc_score(venc_te, pr_te)), "umbral_prob": float(u_clf),
        "recall": float(recall_score(venc_te, a)), "precision": float(precision_score(venc_te, a)),
        "f1": float(f1_score(venc_te, a)), "alertas": int(a.sum()), "detectadas": int((a & venc_te).sum())}

    pi = permutation_importance(mejor, Xva, yva, n_repeats=10, random_state=SEED,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=Xva.columns).sort_values(ascending=False)
    imp.to_csv(RES / "importancia_variables.csv", header=["aumento_rmse"])
    bitacora["importancia"] = imp.round(4).to_dict()

    ej = test.head(8).copy()
    ej["tiempo_predicho"] = p_te[:8].round(2)
    ej["riesgo_predicho"] = nivel_riesgo(ej["tiempo_predicho"], ej["plazo_informe_dias"])
    ej[["id_tutela", "hecho_principal", "nivel_urgencia", "plazo_informe_dias", "tiempo_predicho", OBJETIVO,
        "riesgo_predicho", "nivel_riesgo_real"]].to_csv(RES / "ejemplo_alertas.csv", index=False)

    with open(RES / "metricas.json", "w", encoding="utf-8") as fh:
        json.dump(bitacora, fh, ensure_ascii=False, indent=2, default=float)
    graficas(df, train, comp, yte, p_te, test, imp, b_xgb, pd.DataFrame(curva), u_cons)
    print(json.dumps(bitacora, ensure_ascii=False, indent=1, default=float))


# ----------------------------------------------------------------- Gráficas
def graficas(df, train, comp, yte, p_te, test, imp, b_xgb, curva, u_cons):
    # 1. Objetivo y % de plazo
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(df[OBJETIVO], bins=50, color=AZUL, alpha=.85)
    ax[0].set(title="Tiempo de respuesta (5.000 tutelas)", xlabel="días", ylabel="frecuencia")
    datos = [df.loc[df.plazo_informe_dias == p, OBJETIVO] for p in [1, 2, 3]]
    ax[1].boxplot(datos, tick_labels=["plazo 1 día", "plazo 2 días", "plazo 3 días"], patch_artist=True,
                  boxprops=dict(facecolor="#dbe6f2", color=AZUL), medianprops=dict(color=NARANJA))
    for i, t in enumerate([1, 2, 3], 1):
        ax[1].hlines(t, i - .35, i + .35, colors=ROJO, linestyles="--")
    ax[1].set(title="Tiempo vs. plazo del informe (línea roja)", ylabel="días")
    fig.tight_layout(); fig.savefig(FIG / "01_distribucion_objetivo.png"); plt.close(fig)

    c = comp.set_index("modelo")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(c))
    ax.barh(y + .2, c["RMSE_val"], .4, label="RMSE", color=AZUL)
    ax.barh(y - .2, c["MAE_val"], .4, label="MAE", color=NARANJA)
    for i, (r, m) in enumerate(zip(c["RMSE_val"], c["MAE_val"])):
        ax.text(r + .005, i + .2, f"{r:.3f}", va="center", fontsize=8)
        ax.text(m + .005, i - .2, f"{m:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, c.index); ax.invert_yaxis()
    ax.set(xlabel="error en días (menor es mejor)", title="Comparación de modelos — conjunto de validación")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "02_comparacion_modelos.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(yte, p_te, s=8, alpha=.45, color=AZUL)
    lim = [0, max(yte.max(), p_te.max())]
    ax[0].plot(lim, lim, color=GRIS, ls="--")
    ax[0].set(xlabel="tiempo real (días)", ylabel="tiempo predicho (días)", title="Real vs. predicho (prueba)")
    res = yte - p_te
    ax[1].hist(res, bins=50, color=AZUL, alpha=.85); ax[1].axvline(0, color=NARANJA, ls="--")
    ax[1].set(title="Residuos (real − predicho)", xlabel="días")
    fig.tight_layout(); fig.savefig(FIG / "03_real_vs_predicho.png"); plt.close(fig)

    top = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(top.index, top.values, color=AZUL)
    ax.set(xlabel="aumento del RMSE al permutar la variable (días)", title="Importancia de variables (permutación, validación)")
    fig.tight_layout(); fig.savefig(FIG / "04_importancia_variables.png"); plt.close(fig)

    # 5. Curva de umbral (validación)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(curva.umbral, curva.recall, color=NARANJA, label="recall (vencidas detectadas)")
    ax.plot(curva.umbral, curva.precision, color=AZUL, label="precisión (alertas correctas)")
    ax.axvline(u_cons, color=GRIS, ls="--"); ax.text(u_cons + .01, .15, f"umbral conservador {u_cons:.2f}", color=GRIS)
    ax.axvline(1.0, color=GRIS, ls=":"); ax.text(1.01, .05, "regla estándar 1,00", color=GRIS)
    ax.set(xlabel="umbral de alerta (tiempo predicho ÷ plazo)", ylabel="valor", ylim=(0, 1.02),
           title="Elección del umbral de alerta (conjunto de validación)")
    ax.legend(frameon=False, loc="center left")
    fig.tight_layout(); fig.savefig(FIG / "05_curva_umbral.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(y - .2, c["RMSE_train"], .4, label="train", color=GRIS)
    ax.bar(y + .2, c["RMSE_val"], .4, label="validación", color=AZUL)
    ax.set_xticks(y, [s.split(" · ")[0] for s in c.index])
    ax.set(ylabel="RMSE (días)", title="Generalización: error en entrenamiento vs. validación")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "06_train_vs_validacion.png"); plt.close(fig)

    r = pd.DataFrame(b_xgb.cv_results_)
    g = r.groupby("param_modelo__max_depth")["mean_test_score"].max().mul(-1)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(g.index.astype(int), g.values, marker="o", color=AZUL)
    ax.set(xlabel="max_depth", ylabel="mejor RMSE en CV (días)", title="XGBoost: profundidad vs. error (CV 5-fold)")
    fig.tight_layout(); fig.savefig(FIG / "07_xgb_max_depth.png"); plt.close(fig)


if __name__ == "__main__":
    main()
