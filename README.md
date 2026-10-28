# Predicción de riesgo de vencimiento de términos en tutelas de salud

Proyecto de aula — Inteligencia Artificial, Ingeniería de Sistemas (Universidad Popular del Cesar, 2026-2).
Integrantes: Enor Antonio Acosta Ricardo · Estevan Yesid Sánchez Velasquez.

El sistema estima el **tiempo de respuesta** (días) de una tutela de salud desde su radicación
(regresión supervisada) y lo compara con el **plazo del informe** (1, 2 o 3 días) para asignar un
**nivel de riesgo** (Bajo / Medio / Alto / Vencido) y emitir alertas tempranas.

```
datos del caso → modelo de regresión → tiempo estimado → tiempo / plazo → nivel de riesgo → alerta
```

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 1. Formulación y exploración | Problema, objetivos, estado del arte, dataset, EDA | Completa |
| 2. Desarrollo y modelado | Validación de datos, 6 modelos, ajuste de hiperparámetros, evaluación, alertas | **Completa** |
| 3. Integración y despliegue | Tablero (Streamlit/Power BI), alertas por correo/SMS, informe final | Pendiente |

## Resultados de la Fase 2 (conjunto de prueba, 750 tutelas)

| Modelo | MAE (días) | RMSE (días) | R² |
|---|---|---|---|
| Dummy (media) | 0.561 | 0.730 | 0.000 |
| Regresión lineal (baseline) | 0.352 | 0.492 | 0.545 |
| XGBoost ajustado | 0.353 | 0.493 | 0.543 |
| **Ridge ajustado (seleccionado)** | **0.351** | **0.492** | **0.545** |

- Los modelos reducen el RMSE del Dummy en 32,6 %; lineal, Ridge y XGBoost empatan (efectos aditivos).
- Detección de tutelas que vencen: AUC 0,92. Regla conservadora (razón ≥ 0,95): recall 0,88, precisión 0,75.

## Estructura

```
├── data/
│   ├── raw/dataset_tutelas_salud_sintetico_5000.csv  # dataset del equipo (Fase 1)
│   └── processed/tutelas_preprocesado.csv            # con variables nuevas y partición
├── notebooks/Fase2_Modelado.ipynb                    # notebook ejecutado (demo)
├── src/
│   ├── fase2_modelado.py                             # pipeline completo
│   └── crear_notebook.py                             # genera el notebook
├── models/mejor_modelo.joblib                        # pipeline final (preprocesamiento + Ridge)
└── results/                                          # métricas (json/csv) y figuras
```

## Cómo reproducir

```bash
pip install -r requirements.txt
python src/fase2_modelado.py      # ~1 minuto
```

## Demo de avance (aplicación)

```bash
pip install -r requirements.txt
streamlit run app/demo_app.py
```

En Windows basta con doble clic en `iniciar_demo.bat`. Se abre en el navegador con tres pestañas:

1. **Evaluar una tutela nueva**: se ingresan los datos de radicación y el modelo devuelve días estimados, % del plazo, nivel de riesgo y alerta.
2. **Probar con casos de prueba**: toma tutelas que el modelo nunca vio y compara la predicción con el valor real.
3. **Resultados del modelo**: métricas, comparación de modelos, reglas de alerta y gráficas.

## Uso del modelo guardado

```python
import joblib, pandas as pd
from src.fase2_modelado import crear_caracteristicas, nivel_riesgo, NUM, CAT

modelo = joblib.load("models/mejor_modelo.joblib")
caso = pd.DataFrame([{
    "fecha_radicacion": "2024-10-03", "departamento": "Cesar", "zona": "Rural",
    "tipo_entidad_accionada": "EPS", "hecho_principal": "Cirugia", "nivel_urgencia": "Alta",
    "complejidad_caso": "Alta", "carga_operativa": "Alta", "grupo_edad": "60+",
    "sujeto_especial_proteccion": "Si", "diagnostico_mencionado": "Si",
    "grupo_diagnostico": "Sistema circulatorio", "negacion_previa": "Si",
    "numero_adjuntos": 8, "solicitudes_previas": 2, "plazo_informe_dias": 1}])
caso = crear_caracteristicas(caso)
t = modelo.predict(caso[NUM + CAT])
print(t, nivel_riesgo(t, caso["plazo_informe_dias"]))
```

## Nota sobre los datos

El dataset es sintético (simulado a partir de fuentes oficiales: Defensoría del Pueblo, PQRSD Supersalud,
Decreto 2591 de 1991). Los resultados validan la metodología, no el desempeño en una entidad real.
