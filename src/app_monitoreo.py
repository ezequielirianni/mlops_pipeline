"""
app_monitoreo.py

Tablero de Streamlit para el monitoreo de data drift del modelo de pago de
créditos. Toda la lógica (umbrales, métricas, simulación de batches,
recomendaciones, análisis temporal) vive en model_monitoring.py: esta app
solo lee sus salidas y las visualiza, así los umbrales se definen en un único
lugar.

Ejecución (desde la raíz del repo o desde src/):
    streamlit run src/app_monitoreo.py

Requiere haber corrido antes model_training_evaluation.py (mejor_modelo.pkl).
Si todavía no hay batches de monitoreo, se pueden generar desde la barra
lateral o con: python src/model_monitoring.py --batches 3
"""

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import model_monitoring as mm

st.set_page_config(page_title="Monitoreo de drift", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Estilo: colores de estado (con ícono + etiqueta, nunca solo color) y de series
# ---------------------------------------------------------------------------

COLOR_ESTADO = {"estable": "#0ca30c", "alerta": "#fab219", "drift": "#d03b3b"}
ICONO_ESTADO = {"estable": "🟢", "alerta": "🟡", "drift": "🔴"}
ETIQUETA_ESTADO = {"estable": "Estable", "alerta": "Alerta", "drift": "Drift"}
COLOR_REFERENCIA = "#2a78d6"  # población de entrenamiento
COLOR_ACTUAL = "#eb6834"      # batch actual
PALETA_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MAX_SERIES = len(PALETA_SERIES)


def etiqueta_estado(estado: str) -> str:
    return f"{ICONO_ESTADO[estado]} {ETIQUETA_ESTADO[estado]}"


def umbral_critico(metrica: str) -> float:
    return mm.UMBRALES[metrica]


# ---------------------------------------------------------------------------
# Carga de datos (cacheada; file_mtime es parte de la clave de caché, así se
# vuelve a leer cada archivo cuando cambia en disco)
# ---------------------------------------------------------------------------

def mtime(filename: str) -> float:
    path = mm.get_output_path(filename)
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_report(file_mtime: float):
    path = mm.get_output_path(mm.REPORT_FILENAME)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_history(file_mtime: float) -> pd.DataFrame:
    return mm.load_history()


@st.cache_data(show_spinner=False)
def load_batch_table(file_mtime: float) -> pd.DataFrame:
    path = mm.get_output_path(mm.BATCH_TABLE_FILENAME)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner="Cargando población de referencia y modelo...")
def load_reference_resources():
    """Referencia (X_train), base del batch (X_test), modelo y pronósticos de referencia."""
    reference, base = mm.load_reference_and_base()
    model = mm.load_model()
    proba_ref = mm.predict_no_pago(model, reference)
    return reference, base, model, proba_ref


def drift_table(report: dict) -> pd.DataFrame:
    """Tabla de métricas por variable (una fila por variable + la predicción)."""
    filas = []
    items = {**report["drift_por_variable"], "prediccion": report["drift_prediccion"]}
    for variable, m in items.items():
        umbral = umbral_critico(m["metrica_principal"])
        filas.append({
            "Variable": variable,
            "Tipo": m["tipo"],
            "Estado": etiqueta_estado(m["estado"]),
            "Métrica principal": m["metrica_principal"].upper(),
            "Valor": m["valor_principal"],
            "Umbral crítico": umbral,
            "% del umbral": m["valor_principal"] / umbral,
            "PSI": m["psi"],
            "KS": m.get("ks"),
            "JS": m["js"],
            "Chi² p-valor": m.get("chi2_pvalue"),
            "estado": m["estado"],
        })
    orden = {"drift": 0, "alerta": 1, "estable": 2}
    tabla = pd.DataFrame(filas)
    return tabla.sort_values(["estado", "% del umbral"], key=lambda s: s.map(orden) if s.name == "estado" else -s)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def risk_bars(tabla: pd.DataFrame) -> alt.Chart:
    """Barras de riesgo: métrica principal expresada como % de su umbral
    crítico, para comparar numéricas (PSI) y categóricas (JS) en un mismo eje."""
    datos = tabla.assign(EstadoEtiqueta=tabla["estado"].map(ETIQUETA_ESTADO))
    base = alt.Chart(datos)
    barras = base.mark_bar(cornerRadiusEnd=4, height=14).encode(
        x=alt.X("% del umbral:Q", title="Métrica principal / umbral crítico", axis=alt.Axis(format="%")),
        y=alt.Y("Variable:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=260)),
        color=alt.Color(
            "EstadoEtiqueta:N", title="Estado",
            scale=alt.Scale(domain=list(ETIQUETA_ESTADO.values()), range=list(COLOR_ESTADO.values())),
        ),
        tooltip=[
            "Variable", "Métrica principal",
            alt.Tooltip("Valor:Q", format=".3f"),
            alt.Tooltip("Umbral crítico:Q", format=".2f"),
            alt.Tooltip("% del umbral:Q", format=".0%"),
            alt.Tooltip("EstadoEtiqueta:N", title="Estado"),
        ],
    )
    linea = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(strokeDash=[4, 4], color="#898781").encode(x="x:Q")
    return (barras + linea).properties(height=28 * len(datos))


def numeric_comparison(ref: pd.Series, cur: pd.Series, titulo: str) -> alt.Chart:
    """Histograma normalizado referencia vs actual con los mismos bins. El rango
    se recorta a los percentiles 1-99 de ambas muestras para que los outliers
    no aplasten el gráfico."""
    ref, cur = ref.dropna(), cur.dropna()
    lo = min(ref.quantile(0.01), cur.quantile(0.01))
    hi = max(ref.quantile(0.99), cur.quantile(0.99))
    edges = np.linspace(lo, hi, 31)
    filas = []
    for nombre, serie in [("Referencia (entrenamiento)", ref), ("Batch actual", cur)]:
        conteos = np.histogram(serie.clip(lo, hi), bins=edges)[0]
        for i, c in enumerate(conteos):
            filas.append({"Población": nombre, "desde": edges[i], "hasta": edges[i + 1], "Proporción": c / len(serie)})
    datos = pd.DataFrame(filas)
    return alt.Chart(datos).mark_bar(opacity=0.55, binSpacing=0).encode(
        x=alt.X("desde:Q", title=titulo, bin="binned"),
        x2="hasta:Q",
        y=alt.Y("Proporción:Q", stack=None, axis=alt.Axis(format="%")),
        color=alt.Color(
            "Población:N",
            scale=alt.Scale(domain=["Referencia (entrenamiento)", "Batch actual"], range=[COLOR_REFERENCIA, COLOR_ACTUAL]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[
            "Población",
            alt.Tooltip("desde:Q", format=",.2f"),
            alt.Tooltip("hasta:Q", format=",.2f"),
            alt.Tooltip("Proporción:Q", format=".1%"),
        ],
    ).properties(height=320)


def categorical_comparison(ref: pd.Series, cur: pd.Series, titulo: str) -> alt.Chart:
    """Barras agrupadas con la proporción de cada categoría (nulos como 'Sin dato')."""
    p, q, categorias = mm._categorical_distributions(ref, cur)
    datos = pd.DataFrame({
        "Categoría": categorias * 2,
        "Población": ["Referencia (entrenamiento)"] * len(categorias) + ["Batch actual"] * len(categorias),
        "Proporción": np.concatenate([p, q]),
    })
    return alt.Chart(datos).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Categoría:N", title=titulo, axis=alt.Axis(labelAngle=0)),
        xOffset="Población:N",
        y=alt.Y("Proporción:Q", axis=alt.Axis(format="%")),
        color=alt.Color(
            "Población:N",
            scale=alt.Scale(domain=["Referencia (entrenamiento)", "Batch actual"], range=[COLOR_REFERENCIA, COLOR_ACTUAL]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=["Categoría", "Población", alt.Tooltip("Proporción:Q", format=".1%")],
    ).properties(height=320)


def temporal_chart(historial: pd.DataFrame, variables: list) -> alt.Chart:
    """Evolución de la métrica principal (como % del umbral crítico) por batch."""
    datos = historial[historial["variable"].isin(variables)].copy()
    datos["% del umbral"] = datos["valor_principal"] / datos["metrica_principal"].map(mm.UMBRALES)
    colores = alt.Scale(domain=variables, range=PALETA_SERIES[: len(variables)])
    lineas = alt.Chart(datos).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
        x=alt.X("batch_id:O", title="Batch", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("% del umbral:Q", title="Métrica principal / umbral crítico", axis=alt.Axis(format="%")),
        color=alt.Color("variable:N", title="Variable", scale=colores),
        tooltip=[
            alt.Tooltip("batch_id:O", title="Batch"),
            "variable",
            alt.Tooltip("metrica_principal:N", title="Métrica"),
            alt.Tooltip("valor_principal:Q", title="Valor", format=".3f"),
            alt.Tooltip("% del umbral:Q", format=".0%"),
            "estado",
        ],
    )
    umbral = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[4, 4], color="#d03b3b").encode(y="y:Q")
    return (lineas + umbral).properties(height=360)


def states_per_batch_chart(historial: pd.DataFrame) -> alt.Chart:
    """Cantidad de variables en cada estado por batch (sin la predicción)."""
    datos = (
        historial[historial["variable"] != "prediccion"]
        .groupby(["batch_id", "estado"]).size().reset_index(name="Variables")
    )
    datos["Estado"] = datos["estado"].map(ETIQUETA_ESTADO)
    return alt.Chart(datos).mark_bar().encode(
        x=alt.X("batch_id:O", title="Batch", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Variables:Q", title="Cantidad de variables"),
        color=alt.Color(
            "Estado:N",
            scale=alt.Scale(domain=list(ETIQUETA_ESTADO.values()), range=list(COLOR_ESTADO.values())),
            sort=list(ETIQUETA_ESTADO.values()),
        ),
        order=alt.Order("estado:N"),
        tooltip=[alt.Tooltip("batch_id:O", title="Batch"), "Estado", "Variables"],
    ).properties(height=260)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

st.sidebar.title("Monitoreo de drift")
st.sidebar.caption("Modelo de predicción de pago de créditos")

if st.sidebar.button("▶️ Ejecutar nuevo batch", width="stretch",
                     help="Simula un nuevo lote de producción, calcula el drift y lo agrega al historial."):
    reference, base, model, _ = load_reference_resources()
    with st.spinner("Calculando drift del nuevo batch..."):
        mm.run_monitoring(reference, base, model)
    st.rerun()

report = load_report(mtime(mm.REPORT_FILENAME))
historial = load_history(mtime(mm.HISTORY_FILENAME))

with st.sidebar.expander("Umbrales configurados", expanded=False):
    st.markdown(
        f"""
- **PSI** (numéricas, decide): alerta > {mm.UMBRALES_ALERTA['psi']}, drift > {mm.UMBRALES['psi']}
- **Jensen-Shannon** (categóricas, decide): alerta > {mm.UMBRALES_ALERTA['js']}, drift > {mm.UMBRALES['js']}
- **KS** (numéricas, apoyo): estadístico > {mm.UMBRALES['ks']}
- **Chi²** (categóricas, apoyo): p-valor < {mm.UMBRALES['chi2_pvalue']}
- **Reentrenar** si hay ≥ {mm.MIN_VARIABLES_REENTRENAR} variables en drift o drift en la predicción
"""
    )

if report is None:
    st.title("Monitoreo de data drift")
    st.info(
        "Todavía no hay batches de monitoreo. Usá el botón **Ejecutar nuevo batch** de la barra "
        "lateral o corré `python src/model_monitoring.py --batches 3`."
    )
    st.stop()

st.sidebar.markdown(
    f"**Último batch:** #{report['batch_id']}  \n"
    f"**Fecha:** {report['fecha_monitoreo'].replace('T', ' ')}  \n"
    f"**Registros:** {report['n_batch']:,} (referencia: {report['n_referencia']:,})  \n"
    f"**Batches en el historial:** {historial['batch_id'].nunique() if not historial.empty else 0}"
)

# ---------------------------------------------------------------------------
# Encabezado: indicadores generales del último batch
# ---------------------------------------------------------------------------

st.title("Monitoreo de data drift")
st.caption(
    f"Batch #{report['batch_id']} comparado contra la población de entrenamiento. "
    "Métrica principal: PSI en numéricas y Jensen-Shannon en categóricas."
)

pred = report["drift_prediccion"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Estado global", etiqueta_estado(report["estado_global"]))
c2.metric(f"{ICONO_ESTADO['drift']} Variables en drift", report["resumen"]["drift"])
c3.metric(f"{ICONO_ESTADO['alerta']} Variables en alerta", report["resumen"]["alerta"])
c4.metric("PSI de la predicción", f"{pred['psi']:.3f}", help=f"Estado: {ETIQUETA_ESTADO[pred['estado']]}")
c5.metric(
    "Prob. media de no pago",
    f"{pred['prob_media_actual']:.1%}",
    delta=f"{(pred['prob_media_actual'] - pred['prob_media_referencia']) * 100:+.1f} pp vs referencia",
    delta_color="inverse",
)

tab_metricas, tab_temporal, tab_recomendaciones = st.tabs(
    ["📈 Visualización de métricas", "🕒 Análisis temporal", "🚨 Recomendaciones y alertas"]
)

# ---------------------------------------------------------------------------
# Tab 1: visualización de métricas
# ---------------------------------------------------------------------------

with tab_metricas:
    tabla = drift_table(report)

    st.subheader("Riesgo por variable")
    st.caption("Cada barra muestra la métrica principal como porcentaje de su umbral crítico (línea punteada = 100%).")
    st.altair_chart(risk_bars(tabla), width="stretch")

    st.subheader("Métricas de drift por variable")
    st.dataframe(
        tabla.drop(columns=["estado"]),
        hide_index=True,
        width="stretch",
        column_config={
            "Valor": st.column_config.NumberColumn(format="%.3f"),
            "Umbral crítico": st.column_config.NumberColumn(format="%.2f"),
            "% del umbral": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=max(1.0, tabla["% del umbral"].max())),
            "PSI": st.column_config.NumberColumn(format="%.3f"),
            "KS": st.column_config.NumberColumn(format="%.3f"),
            "JS": st.column_config.NumberColumn(format="%.3f"),
            "Chi² p-valor": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "KS y Chi² se muestran como evidencia complementaria. Chi² tiende a dar p-valores muy bajos "
        "con muestras grandes aun ante diferencias chicas, por eso no decide el estado."
    )

    st.subheader("Distribución histórica vs actual")
    batch_tabla = load_batch_table(mtime(mm.BATCH_TABLE_FILENAME))
    reference, _, _, proba_ref = load_reference_resources()
    tipos = mm.get_feature_types()
    opciones = list(tabla["Variable"])  # ordenadas por gravedad
    variable = st.selectbox("Variable", opciones, index=0)

    if variable == "prediccion":
        st.altair_chart(
            numeric_comparison(pd.Series(proba_ref), batch_tabla["prob_no_pago"], "Probabilidad de no pago predicha"),
            width="stretch",
        )
    elif tipos[variable] == "numerica":
        st.altair_chart(numeric_comparison(reference[variable], batch_tabla[variable], variable), width="stretch")
    else:
        st.altair_chart(categorical_comparison(reference[variable], batch_tabla[variable], variable), width="stretch")

    with st.expander("Tabla de monitoreo: datos del batch + pronósticos entregados"):
        st.dataframe(batch_tabla, hide_index=True, width="stretch")
        st.download_button(
            "Descargar CSV", batch_tabla.to_csv(index=False).encode("utf-8"),
            file_name=mm.BATCH_TABLE_FILENAME, mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Tab 2: análisis temporal
# ---------------------------------------------------------------------------

with tab_temporal:
    n_batches = historial["batch_id"].nunique() if not historial.empty else 0
    if n_batches < 2:
        st.info("Se necesitan al menos 2 batches en el historial para el análisis temporal. Ejecutá un nuevo batch desde la barra lateral.")
    else:
        st.subheader("Evolución del drift por variable")
        ultimo = historial[historial["batch_id"] == historial["batch_id"].max()]
        por_defecto = ultimo[ultimo["estado"] != "estable"].sort_values("valor_principal", ascending=False)["variable"].tolist()
        por_defecto = (por_defecto or ultimo.sort_values("valor_principal", ascending=False)["variable"].tolist())[:5]
        seleccion = st.multiselect(
            f"Variables a graficar (hasta {MAX_SERIES})",
            sorted(historial["variable"].unique()),
            default=por_defecto,
            max_selections=MAX_SERIES,
        )
        if seleccion:
            st.altair_chart(temporal_chart(historial, seleccion), width="stretch")
            st.caption("Línea roja punteada: umbral crítico (100%). La variable 'prediccion' es el drift de la probabilidad de no pago.")

        st.subheader("Tendencias y cambios abruptos")
        tendencias = mm.analyze_trends(historial)
        flecha = {"creciente": "↗️ Creciente", "decreciente": "↘️ Decreciente", "estable": "➡️ Estable"}
        st.dataframe(
            tendencias.assign(
                tendencia=tendencias["tendencia"].map(flecha),
                cambio_abrupto=tendencias["cambio_abrupto"].map({True: "⚠️ Sí", False: "No"}),
            ).rename(columns={
                "variable": "Variable", "valor_actual": "Valor actual", "cambio": "Cambio vs batch anterior",
                "cambio_abrupto": "Cambio abrupto", "pendiente": "Pendiente", "tendencia": "Tendencia",
            }),
            hide_index=True,
            width="stretch",
            column_config={
                "Valor actual": st.column_config.NumberColumn(format="%.3f"),
                "Cambio vs batch anterior": st.column_config.NumberColumn(format="%+.3f"),
                "Pendiente": st.column_config.NumberColumn(format="%+.3f"),
            },
        )
        st.caption(
            f"Cambio abrupto: variación absoluta ≥ {mm.UMBRAL_CAMBIO_ABRUPTO} de la métrica principal entre los "
            "dos últimos batches. Tendencia: pendiente de los últimos 5 batches."
        )

        st.subheader("Estado de las variables por batch")
        st.altair_chart(states_per_batch_chart(historial), width="stretch")

# ---------------------------------------------------------------------------
# Tab 3: recomendaciones y alertas
# ---------------------------------------------------------------------------

with tab_recomendaciones:
    st.subheader(f"Alertas del batch #{report['batch_id']}")
    mostrar = {"critico": st.error, "alerta": st.warning, "ok": st.success}
    iconos = {"critico": "🔴", "alerta": "🟡", "ok": "🟢"}
    for rec in report["recomendaciones"]:
        mostrar[rec["nivel"]](rec["mensaje"], icon=iconos[rec["nivel"]])

    tendencias = mm.analyze_trends(historial)
    if not tendencias.empty:
        abruptos = tendencias[tendencias["cambio_abrupto"]]
        crecientes = tendencias[(tendencias["tendencia"] == "creciente") & ~tendencias["cambio_abrupto"]]
        if not abruptos.empty or not crecientes.empty:
            st.subheader("Alertas temporales")
        for _, fila in abruptos.iterrows():
            st.warning(
                f"Cambio abrupto en '{fila['variable']}': la métrica principal varió {fila['cambio']:+.3f} "
                "respecto del batch anterior.", icon="⚠️",
            )
        for _, fila in crecientes.iterrows():
            st.info(
                f"Tendencia creciente en '{fila['variable']}' (pendiente {fila['pendiente']:+.3f} por batch): "
                "vigilar antes de que supere el umbral.", icon="↗️",
            )

    with st.expander("Cómo interpretar las métricas"):
        st.markdown(
            """
- **PSI (Population Stability Index)**: mide cuánto se redistribuyó la población entre intervalos de la variable.
  Menor a 0.10 estable, entre 0.10 y 0.25 cambio moderado, mayor a 0.25 cambio significativo.
- **Kolmogorov-Smirnov**: distancia máxima entre las distribuciones acumuladas (0 a 1).
- **Jensen-Shannon**: distancia entre dos distribuciones (0 = idénticas, 1 = sin solapamiento).
- **Chi-cuadrado**: test de independencia sobre las frecuencias de cada categoría; un p-valor bajo indica que las proporciones cambiaron.
- **Drift de la predicción**: PSI sobre la probabilidad de no pago. Anticipa cambios en el comportamiento del modelo
  antes de conocer si los clientes efectivamente pagaron.
"""
        )
