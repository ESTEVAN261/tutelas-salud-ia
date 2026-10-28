"""
Demo de avance — Fase 2
Predicción de riesgo de vencimiento de tutelas de salud.

Ejecutar desde la carpeta del proyecto:
    streamlit run app/demo_app.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.fase2_modelado import CAT, NUM, crear_caracteristicas, nivel_riesgo  # noqa: E402

UMBRAL_ALERTA = 0.95
COLORES = {"Bajo": "#2E6B47", "Medio": "#B7791F", "Alto": "#C2410C", "Vencido": "#B3261E"}
PLAZOS = {"Critica": [1], "Alta": [1, 2], "Media": [2, 3], "Baja": [3]}

st.set_page_config(page_title="Demo · Riesgo de vencimiento de tutelas", layout="wide")


@st.cache_resource
def cargar_modelo():
    return joblib.load(RAIZ / "models" / "mejor_modelo.joblib")


@st.cache_data
def cargar_datos():
    df = pd.read_csv(RAIZ / "data" / "processed" / "tutelas_preprocesado.csv")
    with open(RAIZ / "results" / "metricas.json", encoding="utf-8") as fh:
        met = json.load(fh)
    comp = pd.read_csv(RAIZ / "results" / "comparacion_modelos.csv")
    return df, met, comp


modelo = cargar_modelo()
datos, metricas, comparacion = cargar_datos()


def predecir(caso: pd.DataFrame):
    caso = crear_caracteristicas(caso)
    t = float(modelo.predict(caso[NUM + CAT])[0])
    t = max(t, 0.0)
    plazo = int(caso["plazo_informe_dias"].iloc[0])
    razon = t / plazo
    nivel = str(nivel_riesgo([t], [plazo])[0])
    return t, plazo, razon, nivel


def tarjeta_resultado(t, plazo, razon, nivel, real=None, nivel_real=None):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tiempo estimado", f"{t:.2f} días")
    c1.caption(f"≈ {t * 24:.0f} horas")
    c2.metric("Plazo del informe", f"{plazo} día(s)")
    c3.metric("Plazo consumido", f"{razon * 100:.1f} %")
    color = COLORES[nivel]
    c4.markdown(
        f"<div style='background:{color};color:white;border-radius:10px;padding:14px;text-align:center'>"
        f"<div style='font-size:14px'>Nivel de riesgo</div><div style='font-size:30px;font-weight:700'>{nivel}</div></div>",
        unsafe_allow_html=True)
    st.progress(min(razon / 1.5, 1.0), text=f"Razón tiempo / plazo = {razon:.2f}  (alerta desde {UMBRAL_ALERTA:.2f}, vence desde 1,00)")
    if razon >= UMBRAL_ALERTA:
        st.error(f"🔔 ALERTA: se estima que esta tutela consumirá el {razon * 100:.1f} % del plazo. Priorizar su gestión.")
    else:
        st.success(f"Sin alerta: se estima que consumirá el {razon * 100:.1f} % del plazo.")
    if real is not None:
        st.info(f"Valor real registrado: {real:.2f} días → nivel real **{nivel_real}**. "
                f"Error de la estimación: {abs(real - t):.2f} días ({abs(real - t) * 24:.1f} horas).")


st.title("Predicción de riesgo de vencimiento de tutelas de salud")
st.caption("Demo de avance · Fase 2 · Universidad Popular del Cesar · Inteligencia Artificial 2026-2 · "
           "Modelo: " + metricas["evaluacion_final"]["mejor_modelo"])

tab1, tab2, tab3 = st.tabs(["1 · Evaluar una tutela nueva", "2 · Probar con casos de prueba", "3 · Resultados del modelo"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.subheader("Ingrese los datos de la tutela al momento de radicarla")
    a, b, c = st.columns(3)
    with a:
        fecha = st.date_input("Fecha de radicación", date(2024, 10, 3))
        departamento = st.selectbox("Departamento", sorted(datos["departamento"].unique()), index=sorted(datos["departamento"].unique()).index("Cesar"))
        zona = st.radio("Zona", ["Urbana", "Rural"], horizontal=True)
        entidad = st.selectbox("Entidad accionada", sorted(datos["tipo_entidad_accionada"].unique()))
        hecho = st.selectbox("Hecho principal", sorted(datos["hecho_principal"].unique()))
    with b:
        urgencia = st.select_slider("Nivel de urgencia", ["Baja", "Media", "Alta", "Critica"], value="Alta")
        plazo = st.radio("Plazo del informe (días)", PLAZOS[urgencia], horizontal=True)
        complejidad = st.select_slider("Complejidad del caso", ["Baja", "Media", "Alta"], value="Media")
        carga = st.select_slider("Carga operativa del equipo", ["Baja", "Media", "Alta"], value="Media")
        edad = st.selectbox("Grupo de edad del paciente", ["0-17", "18-39", "40-59", "60+"], index=2)
    with c:
        sujeto = st.radio("¿Sujeto de especial protección?", ["No", "Si"], horizontal=True)
        diag = st.radio("¿Se menciona diagnóstico?", ["No", "Si"], horizontal=True)
        grupo_diag = st.selectbox("Grupo diagnóstico", sorted(datos["grupo_diagnostico"].unique()),
                                  index=sorted(datos["grupo_diagnostico"].unique()).index("No informado"),
                                  disabled=(diag == "No"))
        negacion = st.radio("¿Hubo negación previa?", ["No", "Si"], horizontal=True)
        adjuntos = st.number_input("Número de adjuntos", 0, 20, 6)
        previas = st.number_input("Solicitudes previas", 0, 10, 0)

    caso = pd.DataFrame([{
        "fecha_radicacion": str(fecha), "departamento": departamento, "zona": zona,
        "tipo_entidad_accionada": entidad, "hecho_principal": hecho, "nivel_urgencia": urgencia,
        "complejidad_caso": complejidad, "carga_operativa": carga, "grupo_edad": edad,
        "sujeto_especial_proteccion": sujeto, "diagnostico_mencionado": diag,
        "grupo_diagnostico": grupo_diag if diag == "Si" else "No informado", "negacion_previa": negacion,
        "numero_adjuntos": adjuntos, "solicitudes_previas": previas, "plazo_informe_dias": plazo}])

    if st.button("Estimar riesgo", type="primary"):
        st.divider()
        tarjeta_resultado(*predecir(caso))

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.subheader("Casos del conjunto de prueba (el modelo nunca los vio al entrenar)")
    prueba = datos[datos["particion"] == "test"].reset_index(drop=True)
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if st.button("Tomar otra tutela al azar"):
        st.session_state.idx = int(np.random.randint(len(prueba)))
    fila = prueba.iloc[[st.session_state.idx]]
    r = fila.iloc[0]
    st.write(f"**{r['id_tutela']}** · {r['hecho_principal']} · urgencia {r['nivel_urgencia']} · "
             f"complejidad {r['complejidad_caso']} · carga {r['carga_operativa']} · zona {r['zona']} · "
             f"{r['tipo_entidad_accionada']} · plazo {r['plazo_informe_dias']} día(s)")
    columnas = ["fecha_radicacion", "departamento", "zona", "tipo_entidad_accionada", "hecho_principal",
                "nivel_urgencia", "complejidad_caso", "carga_operativa", "grupo_edad", "sujeto_especial_proteccion",
                "diagnostico_mencionado", "grupo_diagnostico", "negacion_previa", "numero_adjuntos",
                "solicitudes_previas", "plazo_informe_dias"]
    tarjeta_resultado(*predecir(fila[columnas].copy()), real=float(r["tiempo_respuesta_dias"]),
                      nivel_real=r["nivel_riesgo_real"])

# ---------------------------------------------------------------- Tab 3
with tab3:
    fin = metricas["evaluacion_final"]
    st.subheader("Desempeño en el conjunto de prueba (750 tutelas)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{fin['test_mejor']['MAE']:.2f} días")
    m1.caption(f"≈ {fin['test_mejor']['MAE'] * 24:.1f} horas de error promedio")
    m2.metric("R²", f"{fin['test_mejor']['R2']:.2f}")
    m3.metric("Mejora vs. modelo ingenuo", f"{fin['mejora_rmse_vs_dummy_pct']:.1f} %")
    m4.metric("AUC vencimiento", f"{metricas['auc_vencimiento']['test']:.2f}")
    st.markdown("**Comparación de modelos (validación)**")
    tabla = comparacion[["modelo", "MAE_val", "RMSE_val", "R2_val", "RMSE_train"]].rename(columns={
        "modelo": "Modelo", "MAE_val": "MAE val", "RMSE_val": "RMSE val", "R2_val": "R² val", "RMSE_train": "RMSE train"})
    st.dataframe(tabla.style.format(precision=3), hide_index=True, use_container_width=True)
    st.markdown("**Reglas de alerta en prueba**")
    al = pd.DataFrame([{"Regla": k, "Alertas": v["alertas"], "Vencidas detectadas": f"{v['detectadas']} de {v['vencidas_reales']}",
                        "Recall": round(v["recall"], 2), "Precisión": round(v["precision"], 2)}
                       for k, v in metricas["alertas"].items()])
    st.dataframe(al, hide_index=True, use_container_width=True)
    g1, g2 = st.columns(2)
    g1.image(str(RAIZ / "results" / "figuras" / "02_comparacion_modelos.png"))
    g2.image(str(RAIZ / "results" / "figuras" / "03_real_vs_predicho.png"))
    g1.image(str(RAIZ / "results" / "figuras" / "04_importancia_variables.png"))
    g2.image(str(RAIZ / "results" / "figuras" / "05_curva_umbral.png"))
