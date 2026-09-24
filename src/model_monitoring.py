"""
model_monitoring.py

Tercera componente del flujo de modelos operativos.
Responsabilidad: monitorear la población que llega al modelo y detectar
data drift, es decir, cambios en la distribución de las variables respecto de
la población con la que se entrenó, que puedan afectar su desempeño.

Flujo de una ejecución (un "batch" de monitoreo):

1. Referencia: la población de entrenamiento (X_train de ft_engineering.py) y
   los pronósticos del modelo sobre ella.
2. Batch actual: como no hay datos productivos, se SIMULA el lote que llegaría
   a producción. Se toma una muestra del histórico no usado para entrenar
   (X_test) y se le aplica una perturbación controlada y reproducible:
     - numéricas: ruido gaussiano del 10% de su desvío estándar típico
       (calculado sin el 1% más extremo de cada cola, ver robust_std);
     - desplazamiento deliberado de la media en puntaje_datacredito y
       salario_cliente, para que el drift sea visible;
     - categóricas: proporciones distintas a las del histórico.
3. Tabla de monitoreo: el batch junto con los pronósticos entregados por el
   modelo (mejor_modelo.pkl).
4. Métricas de drift por variable: PSI, Kolmogorov-Smirnov, Jensen-Shannon y
   Chi-cuadrado, más el drift de la predicción.
5. Estado por variable (semáforo), alertas y recomendaciones.
6. Salidas en la raíz del repo:
     - drift_report.json: reporte completo del último batch.
     - drift_history.csv: una fila por variable y por batch, acumulada entre
       ejecuciones (base del análisis temporal en app_monitoreo.py).
     - ultimo_batch_predicciones.csv: la tabla de datos + pronósticos del
       último batch.

Periodicidad: cada ejecución representa un período de monitoreo (por ejemplo,
un corte semanal o mensual de solicitudes). El número de batch se usa como
semilla, así cada período es distinto pero reproducible.

Uso:
    python model_monitoring.py              # corre 1 batch nuevo
    python model_monitoring.py --batches 5  # corre 5 batches seguidos
    python model_monitoring.py --reset      # borra el historial antes de correr
"""

import argparse
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

from ft_engineering import find_repo_root, get_feature_lists, get_train_test_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42  # misma semilla base que el resto del proyecto

# ---------------------------------------------------------------------------
# Umbrales de drift
# ---------------------------------------------------------------------------

# Umbral crítico por métrica: si se supera, la variable está en DRIFT.
#   - psi y js: valores mayores al umbral indican drift.
#   - ks: se usa el estadístico D (distancia máxima entre distribuciones
#     acumuladas), no el p-valor.
#   - chi2_pvalue: p-valor MENOR al umbral indica drift.
UMBRALES = {
    "psi": 0.25,
    "ks": 0.25,
    "js": 0.30,
    "chi2_pvalue": 0.05,
}

# Umbral de ALERTA (amarillo) para las métricas que deciden el semáforo.
# PSI 0.10 es el corte estándar de la industria de crédito para "cambio
# moderado"; para JS se toma la mitad del umbral crítico con el mismo criterio.
UMBRALES_ALERTA = {
    "psi": 0.10,
    "js": 0.15,
}

# Métrica que decide el estado de cada tipo de variable. El resto de las
# métricas se calculan y reportan como evidencia complementaria:
#   - PSI para numéricas: resume el cambio en toda la distribución y tiene
#     umbrales de interpretación estándar.
#   - Jensen-Shannon para categóricas: acotada entre 0 y 1 y estable aun con
#     categorías poco frecuentes.
# Chi-cuadrado no decide porque con muestras grandes marca como significativas
# diferencias muy chicas; KS, porque solo aplica a numéricas.
METRICA_PRINCIPAL = {
    "numerica": "psi",
    "categorica": "js",
}

ESTADOS = ["estable", "alerta", "drift"]  # orden de gravedad

# Si al menos esta cantidad de variables está en drift (o la predicción lo
# está), se recomienda reentrenar el modelo.
MIN_VARIABLES_REENTRENAR = 3

# Cambio absoluto de la métrica principal entre dos batches consecutivos a
# partir del cual se considera un "cambio abrupto" en el análisis temporal.
UMBRAL_CAMBIO_ABRUPTO = 0.10

# Pendiente mínima (por batch) de la métrica principal para considerar que una
# variable tiene tendencia creciente o decreciente en los últimos batches.
UMBRAL_PENDIENTE = 0.02

# Variables que provienen de la consulta externa a DataCrédito (ver EDA). Si
# alguna entra en drift, la recomendación sugiere revisar esa integración.
VARIABLES_DATACREDITO = {
    "puntaje_datacredito", "huella_consulta", "promedio_ingresos_datacredito",
    "tendencia_ingresos_limpia",
}

# ---------------------------------------------------------------------------
# Parámetros de la simulación del batch
# ---------------------------------------------------------------------------

BATCH_SIZE = 1000
RUIDO_FRACCION_STD = 0.10  # ruido gaussiano: 10% del desvío estándar de cada numérica

# Desplazamiento deliberado de la media, en desvíos estándar de la referencia.
# Simula un deterioro de la población: clientes con peor puntaje crediticio y
# salarios declarados más altos (por ejemplo, un cambio en el canal de venta).
# Los tamaños se eligieron para que ambas superen el umbral de PSI (0.25) en
# todos los batches sin dominar la escala de los gráficos. Con los desvíos
# recortados de la referencia equivalen a unos -32 puntos de puntaje y
# +820 mil de salario. (Antes de sanear los rangos en ft_engineering, los
# errores inflaban los desvíos y bastaba con desplazamientos menores.)
DESPLAZAMIENTOS_MEDIA = {
    "puntaje_datacredito": -0.7,
    "salario_cliente": 0.3,
}

# Categóricas: las nuevas proporciones se obtienen mezclando las de la
# referencia con una distribución uniforme (0 = sin cambio, 1 = uniforme).
MEZCLA_CATEGORICAS = 0.5

# Numéricas que no pueden ser negativas: después del ruido se acotan en 0.
NUMERICAS_NO_NEGATIVAS = {
    "edad_cliente", "capital_prestado", "plazo_meses", "puntaje_datacredito",
    "huella_consulta", "salario_cliente", "total_otros_prestamos", "cuota_pactada",
    "cant_creditosvigentes", "saldo_mora", "saldo_principal",
    "promedio_ingresos_datacredito", "ratio_cuota_salario",
}

# ---------------------------------------------------------------------------
# Archivos de salida (raíz del repo, al lado de mejor_modelo.pkl)
# ---------------------------------------------------------------------------

MODEL_FILENAME = "mejor_modelo.pkl"
REPORT_FILENAME = "drift_report.json"
HISTORY_FILENAME = "drift_history.csv"
BATCH_TABLE_FILENAME = "ultimo_batch_predicciones.csv"

CLASE_NO_PAGA = 0  # clase de interés del modelo (ver model_training_evaluation.py)
SIN_DATO = "Sin dato"  # etiqueta para los nulos de las categóricas en las métricas
EPS = 1e-6  # evita log(0) y divisiones por cero en PSI


def get_output_path(filename: str):
    return find_repo_root() / filename


def robust_std(serie: pd.Series) -> float:
    """Desvío estándar sin el 1% más extremo de cada cola (dispersión típica).

    No corrige errores de carga (eso ya lo hace ft_engineering.validar_rangos):
    resuelve variables con colas muy largas de valores VÁLIDOS. Por ejemplo,
    total_otros_prestamos tiene un desvío común de ~24.7 millones por unos
    pocos clientes con préstamos muy grandes, pero la dispersión típica es de
    ~1.6 millones. Con el desvío común, un ruido del "10%" sería mayor que la
    mayoría de los valores y generaría drift artificial (PSI ~1.0) en una
    variable que no se quiso desplazar."""
    s = serie.dropna()
    lo, hi = s.quantile([0.01, 0.99])
    return float(s[(s >= lo) & (s <= hi)].std())


def is_integer_valued(serie: pd.Series) -> bool:
    """True si la variable solo toma valores enteros (plazos, conteos, montos redondos)."""
    s = serie.dropna()
    return bool(np.all(s == s.round()))


# ---------------------------------------------------------------------------
# Carga de la referencia y del modelo
# ---------------------------------------------------------------------------

def load_model():
    """Carga el pipeline entrenado (preprocesador + modelo) desde la raíz del repo."""
    path = get_output_path(MODEL_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecutar antes model_training_evaluation.py."
        )
    return joblib.load(path)


def get_feature_types() -> dict:
    """{variable: 'numerica' | 'categorica'} para las 16 features del modelo."""
    features = get_feature_lists()
    tipos = {col: "numerica" for col in features["numeric"]}
    tipos.update({col: "categorica" for col in features["categorical"] + features["categorical_ordinal"]})
    return tipos


def load_reference_and_base():
    """Devuelve (referencia, base_batch):
    - referencia: X_train, la población con la que se entrenó el modelo.
    - base_batch: X_test, histórico no usado en el entrenamiento, del que se
      muestrea el batch simulado.
    """
    X_train, X_test, _, _ = get_train_test_data()
    return X_train, X_test


def predict_no_pago(model, X: pd.DataFrame) -> np.ndarray:
    """Probabilidad de NO pago (clase 0), buscando la columna por etiqueta."""
    idx = list(model.classes_).index(CLASE_NO_PAGA)
    return model.predict_proba(X)[:, idx]


# ---------------------------------------------------------------------------
# Simulación del batch actual
# ---------------------------------------------------------------------------

def simulate_batch(reference: pd.DataFrame, base: pd.DataFrame, seed: int,
                   batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """Simula el lote 'actual' que llegaría a producción.

    Toma una muestra de 'base' y le aplica una perturbación controlada y
    reproducible (misma semilla -> mismo batch). Los parámetros de la
    perturbación (desvíos, proporciones) se calculan sobre la referencia.
    """
    rng = np.random.default_rng(seed)
    batch = base.sample(n=min(batch_size, len(base)), random_state=seed).reset_index(drop=True)

    for col, tipo in get_feature_types().items():
        if tipo == "numerica":
            std = robust_std(reference[col])
            ruido = rng.normal(0, RUIDO_FRACCION_STD * std, size=len(batch))
            desplazamiento = DESPLAZAMIENTOS_MEDIA.get(col, 0.0) * std
            batch[col] = batch[col] + ruido + desplazamiento  # los NaN siguen siendo NaN
            # Las variables enteras (plazo en meses, cantidad de créditos, etc.)
            # se redondean: un plazo de 12.3 meses no existe, y los decimales
            # generarían drift artificial en variables con muchos valores repetidos.
            if is_integer_valued(reference[col]):
                batch[col] = batch[col].round()
            if col in NUMERICAS_NO_NEGATIVAS:
                batch[col] = batch[col].clip(lower=0)
        else:
            # Nuevas proporciones: mezcla entre las de la referencia y una uniforme.
            # Los nulos se tratan como una categoría más para conservarlos.
            ref_props = reference[col].astype(object).fillna(SIN_DATO).value_counts(normalize=True)
            uniforme = 1 / len(ref_props)
            nuevas_props = (1 - MEZCLA_CATEGORICAS) * ref_props + MEZCLA_CATEGORICAS * uniforme
            valores = rng.choice(ref_props.index.to_numpy(), size=len(batch), p=nuevas_props.to_numpy())
            nueva_col = pd.Series(valores).replace(SIN_DATO, np.nan)
            batch[col] = nueva_col.astype(reference[col].dtype)

    return batch


def build_monitoring_table(batch: pd.DataFrame, model, batch_id: int, timestamp: str) -> pd.DataFrame:
    """Tabla de monitoreo: los datos del batch junto con los pronósticos entregados."""
    tabla = batch.copy()
    tabla["prob_no_pago"] = predict_no_pago(model, batch)
    tabla["prediccion"] = model.predict(batch)
    tabla["prediccion_etiqueta"] = np.where(tabla["prediccion"] == CLASE_NO_PAGA, "No paga a tiempo", "Paga a tiempo")
    tabla.insert(0, "batch_id", batch_id)
    tabla.insert(1, "fecha_monitoreo", timestamp)
    return tabla


# ---------------------------------------------------------------------------
# Métricas de drift
# ---------------------------------------------------------------------------

def _numeric_distributions(ref: pd.Series, cur: pd.Series, bins: int = 10):
    """Proporciones por bin de ref y cur. Los bordes salen de los cuantiles de
    la referencia (bins de igual población) y se extienden a -inf/+inf para
    que los valores nuevos fuera de rango caigan en los bins extremos."""
    ref, cur = ref.dropna(), cur.dropna()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, bins=edges)[0] / len(ref)
    q = np.histogram(cur, bins=edges)[0] / len(cur)
    return p, q


def _categorical_distributions(ref: pd.Series, cur: pd.Series):
    """Proporciones por categoría (nulos como 'Sin dato') sobre la unión de categorías."""
    ref = ref.astype(object).fillna(SIN_DATO).astype(str)
    cur = cur.astype(object).fillna(SIN_DATO).astype(str)
    categorias = sorted(set(ref) | set(cur))
    p = ref.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    q = cur.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    return p, q, categorias


def psi_from_distributions(p: np.ndarray, q: np.ndarray) -> float:
    """Population Stability Index: suma de (q - p) * ln(q / p)."""
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def js_from_distributions(p: np.ndarray, q: np.ndarray) -> float:
    """Distancia de Jensen-Shannon en base 2: 0 = idénticas, 1 = sin solapamiento."""
    return float(jensenshannon(p, q, base=2))


def chi2_pvalue(ref: pd.Series, cur: pd.Series) -> float:
    """Test Chi-cuadrado sobre la tabla de contingencia de frecuencias
    (referencia vs actual). p-valor bajo = las proporciones difieren."""
    ref = ref.astype(object).fillna(SIN_DATO).astype(str)
    cur = cur.astype(object).fillna(SIN_DATO).astype(str)
    tabla = pd.DataFrame({"ref": ref.value_counts(), "cur": cur.value_counts()}).fillna(0)
    tabla = tabla[(tabla["ref"] + tabla["cur"]) > 0]
    return float(chi2_contingency(tabla.T.to_numpy())[1])


def classify_state(valor: float, metrica: str) -> str:
    """Semáforo de la métrica principal: estable / alerta / drift."""
    if valor > UMBRALES[metrica]:
        return "drift"
    if valor > UMBRALES_ALERTA[metrica]:
        return "alerta"
    return "estable"


def compute_feature_drift(ref: pd.Series, cur: pd.Series, tipo: str) -> dict:
    """Calcula las 4 métricas aplicables a una variable y su estado."""
    if tipo == "numerica":
        p, q = _numeric_distributions(ref, cur)
        ks = ks_2samp(ref.dropna(), cur.dropna())
        metricas = {
            "psi": psi_from_distributions(p, q),
            "js": js_from_distributions(p, q),
            "ks": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "chi2_pvalue": None,
        }
        soporte = metricas["ks"] > UMBRALES["ks"]
    else:
        p, q, _ = _categorical_distributions(ref, cur)
        metricas = {
            "psi": psi_from_distributions(p, q),
            "js": js_from_distributions(p, q),
            "ks": None,
            "ks_pvalue": None,
            "chi2_pvalue": chi2_pvalue(ref, cur),
        }
        soporte = metricas["chi2_pvalue"] < UMBRALES["chi2_pvalue"]

    metrica = METRICA_PRINCIPAL[tipo]
    return {
        "tipo": tipo,
        **metricas,
        "metrica_principal": metrica,
        "valor_principal": metricas[metrica],
        "estado": classify_state(metricas[metrica], metrica),
        # True si la métrica complementaria (KS para numéricas, Chi² para
        # categóricas) también supera su umbral.
        "confirmado_por_test": bool(soporte),
    }


def compute_prediction_drift(proba_ref: np.ndarray, proba_cur: np.ndarray) -> dict:
    """Drift de la salida del modelo: compara la distribución de la
    probabilidad de no pago. Es una señal temprana, disponible aun antes de
    conocer si los clientes del batch efectivamente pagaron."""
    p, q = _numeric_distributions(pd.Series(proba_ref), pd.Series(proba_cur))
    psi = psi_from_distributions(p, q)
    return {
        "tipo": "prediccion",
        "psi": psi,
        "js": js_from_distributions(p, q),
        "ks": float(ks_2samp(proba_ref, proba_cur).statistic),
        "metrica_principal": "psi",
        "valor_principal": psi,
        "estado": classify_state(psi, "psi"),
        "prob_media_referencia": float(np.mean(proba_ref)),
        "prob_media_actual": float(np.mean(proba_cur)),
    }


# ---------------------------------------------------------------------------
# Alertas y recomendaciones
# ---------------------------------------------------------------------------

def build_recommendations(drift_por_variable: dict, drift_prediccion: dict) -> list:
    """Genera mensajes automáticos según los estados. Cada mensaje tiene un
    nivel ('critico', 'alerta' o 'ok') para mostrarlo en el tablero."""
    en_drift = [v for v, m in drift_por_variable.items() if m["estado"] == "drift"]
    en_alerta = [v for v, m in drift_por_variable.items() if m["estado"] == "alerta"]
    mensajes = []

    if len(en_drift) >= MIN_VARIABLES_REENTRENAR or drift_prediccion["estado"] == "drift":
        motivo = []
        if len(en_drift) >= MIN_VARIABLES_REENTRENAR:
            motivo.append(f"{len(en_drift)} variables en drift")
        if drift_prediccion["estado"] == "drift":
            motivo.append(f"drift en la predicción (PSI={drift_prediccion['psi']:.3f})")
        mensajes.append({
            "nivel": "critico",
            "mensaje": (
                f"Se recomienda REENTRENAR el modelo: {' y '.join(motivo)}. La población actual "
                "difiere de la de entrenamiento y el desempeño reportado puede no sostenerse."
            ),
        })

    for var in en_drift:
        m = drift_por_variable[var]
        texto = (
            f"Revisar la variable '{var}': {m['metrica_principal'].upper()}={m['valor_principal']:.3f} "
            f"supera el umbral crítico ({UMBRALES[m['metrica_principal']]})."
        )
        if var in VARIABLES_DATACREDITO:
            texto += " Proviene de DataCrédito: verificar la integración con esa fuente."
        else:
            texto += " Confirmar si es un cambio real de la población o un problema de carga."
        mensajes.append({"nivel": "critico", "mensaje": texto})

    if en_alerta:
        mensajes.append({
            "nivel": "alerta",
            "mensaje": f"Cambio moderado en {', '.join(en_alerta)}: monitorear en los próximos batches.",
        })

    if drift_prediccion["estado"] == "alerta":
        mensajes.append({
            "nivel": "alerta",
            "mensaje": f"La distribución de la probabilidad de no pago cambió moderadamente (PSI={drift_prediccion['psi']:.3f}).",
        })

    if not mensajes:
        mensajes.append({"nivel": "ok", "mensaje": "Sin drift relevante: no se requieren acciones."})

    return mensajes


def overall_state(drift_por_variable: dict, drift_prediccion: dict) -> str:
    """Estado global del batch: el más grave entre variables y predicción."""
    estados = [m["estado"] for m in drift_por_variable.values()] + [drift_prediccion["estado"]]
    return max(estados, key=ESTADOS.index)


# ---------------------------------------------------------------------------
# Historial y análisis temporal
# ---------------------------------------------------------------------------

def load_history() -> pd.DataFrame:
    path = get_output_path(HISTORY_FILENAME)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def next_batch_id() -> int:
    historial = load_history()
    return 1 if historial.empty else int(historial["batch_id"].max()) + 1


def append_history(report: dict):
    """Agrega al historial una fila por variable (y una para la predicción)."""
    filas = []
    for variable, m in {**report["drift_por_variable"], "prediccion": report["drift_prediccion"]}.items():
        filas.append({
            "batch_id": report["batch_id"],
            "fecha_monitoreo": report["fecha_monitoreo"],
            "variable": variable,
            "tipo": m["tipo"],
            "psi": m["psi"],
            "ks": m.get("ks"),
            "js": m["js"],
            "chi2_pvalue": m.get("chi2_pvalue"),
            "metrica_principal": m["metrica_principal"],
            "valor_principal": m["valor_principal"],
            "estado": m["estado"],
        })
    historial = pd.concat([load_history(), pd.DataFrame(filas)], ignore_index=True)
    historial.to_csv(get_output_path(HISTORY_FILENAME), index=False)
    return historial


def analyze_trends(historial: pd.DataFrame, ventana: int = 5) -> pd.DataFrame:
    """Análisis temporal por variable sobre la métrica principal:
    - cambio: diferencia entre los dos últimos batches;
    - cambio_abrupto: |cambio| >= UMBRAL_CAMBIO_ABRUPTO;
    - tendencia: pendiente de los últimos 'ventana' batches (creciente,
      decreciente o estable). Requiere al menos 2 batches.
    """
    if historial.empty or historial["batch_id"].nunique() < 2:
        return pd.DataFrame()

    filas = []
    for variable, grupo in historial.sort_values("batch_id").groupby("variable"):
        valores = grupo["valor_principal"].to_numpy()[-ventana:]
        batches = grupo["batch_id"].to_numpy()[-ventana:]
        cambio = float(valores[-1] - valores[-2])
        pendiente = float(np.polyfit(batches, valores, 1)[0]) if len(valores) >= 2 else 0.0
        if pendiente > UMBRAL_PENDIENTE:
            tendencia = "creciente"
        elif pendiente < -UMBRAL_PENDIENTE:
            tendencia = "decreciente"
        else:
            tendencia = "estable"
        filas.append({
            "variable": variable,
            "valor_actual": float(valores[-1]),
            "cambio": cambio,
            "cambio_abrupto": abs(cambio) >= UMBRAL_CAMBIO_ABRUPTO,
            "pendiente": pendiente,
            "tendencia": tendencia,
        })
    return pd.DataFrame(filas).sort_values("valor_actual", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ejecución de un batch de monitoreo
# ---------------------------------------------------------------------------

def run_monitoring(reference=None, base=None, model=None) -> dict:
    """Ejecuta un batch completo de monitoreo y guarda sus salidas.
    Los argumentos opcionales permiten reutilizar referencia, base y modelo ya
    cargados al correr varios batches seguidos."""
    if reference is None or base is None:
        reference, base = load_reference_and_base()
    model = model or load_model()

    batch_id = next_batch_id()
    seed = RANDOM_STATE + batch_id
    timestamp = datetime.now().isoformat(timespec="seconds")
    logger.info(f"Batch {batch_id} (semilla {seed})")

    batch = simulate_batch(reference, base, seed=seed)
    tabla = build_monitoring_table(batch, model, batch_id, timestamp)
    tabla.to_csv(get_output_path(BATCH_TABLE_FILENAME), index=False)

    drift_por_variable = {
        col: compute_feature_drift(reference[col], batch[col], tipo)
        for col, tipo in get_feature_types().items()
    }
    drift_prediccion = compute_prediction_drift(predict_no_pago(model, reference), tabla["prob_no_pago"].to_numpy())

    report = {
        "batch_id": batch_id,
        "semilla": seed,
        "fecha_monitoreo": timestamp,
        "n_referencia": int(len(reference)),
        "n_batch": int(len(batch)),
        "umbrales": UMBRALES,
        "umbrales_alerta": UMBRALES_ALERTA,
        "metrica_principal": METRICA_PRINCIPAL,
        "estado_global": overall_state(drift_por_variable, drift_prediccion),
        "resumen": {estado: sum(m["estado"] == estado for m in drift_por_variable.values()) for estado in ESTADOS},
        "tasa_no_pago_predicha": float((tabla["prediccion"] == CLASE_NO_PAGA).mean()),
        "drift_prediccion": drift_prediccion,
        "drift_por_variable": drift_por_variable,
        "recomendaciones": build_recommendations(drift_por_variable, drift_prediccion),
    }

    with open(get_output_path(REPORT_FILENAME), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    append_history(report)

    logger.info(
        f"Batch {batch_id}: estado global={report['estado_global'].upper()} | "
        f"variables: {report['resumen']} | PSI predicción={drift_prediccion['psi']:.3f}"
    )
    return report


def print_report(report: dict):
    """Resumen legible del batch en consola."""
    print(f"\nBatch {report['batch_id']} - estado global: {report['estado_global'].upper()}")
    filas = [
        {
            "variable": var,
            "tipo": m["tipo"],
            "PSI": round(m["psi"], 3),
            "KS": None if m["ks"] is None else round(m["ks"], 3),
            "JS": round(m["js"], 3),
            "Chi2 p": None if m["chi2_pvalue"] is None else round(m["chi2_pvalue"], 4),
            "estado": m["estado"],
        }
        for var, m in report["drift_por_variable"].items()
    ]
    tabla = pd.DataFrame(filas).sort_values("estado", key=lambda s: s.map(ESTADOS.index), ascending=False)
    print(tabla.to_string(index=False))
    pred = report["drift_prediccion"]
    print(
        f"\nPredicción: PSI={pred['psi']:.3f} ({pred['estado']}) | prob. media de no pago "
        f"{pred['prob_media_referencia']:.3f} (ref) -> {pred['prob_media_actual']:.3f} (actual)"
    )
    print("\nRecomendaciones:")
    for r in report["recomendaciones"]:
        print(f"  [{r['nivel'].upper()}] {r['mensaje']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoreo de data drift del modelo de pago de créditos")
    parser.add_argument("--batches", type=int, default=1, help="cantidad de batches a simular (default: 1)")
    parser.add_argument("--reset", action="store_true", help="borra el historial antes de correr")
    args = parser.parse_args()

    if args.reset:
        get_output_path(HISTORY_FILENAME).unlink(missing_ok=True)
        logger.info("Historial de drift reiniciado.")

    reference, base = load_reference_and_base()
    model = load_model()
    for _ in range(args.batches):
        report = run_monitoring(reference, base, model)

    print_report(report)
