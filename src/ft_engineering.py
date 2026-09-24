"""
ft_engineering.py

Primera componente del flujo de creación de modelos operativos.
Responsabilidad: cargar el dataset crudo, aplicar limpieza y feature engineering,
y devolver los conjuntos de entrenamiento/evaluación listos para que
model_training_evaluation.py construya y entrene los modelos.

Uso esperado desde model_training_evaluation.py:

    from ft_engineering import build_preprocessor, find_repo_root, get_train_test_data

    X_train, X_test, y_train, y_test = get_train_test_data()
    preprocessor = build_preprocessor()

El desbalance de clases NO se resuelve en este módulo: se compensa durante el
entrenamiento con sample_weight (ver model_training_evaluation.py), que funciona
igual para los 4 modelos candidatos.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolución de rutas (independiente de desde dónde se ejecute el script)
# ---------------------------------------------------------------------------

def find_repo_root(marker: str = "requirements.txt", start: Path = None) -> Path:
    """Busca hacia arriba desde 'start' (o el directorio actual) hasta encontrar
    la carpeta que contiene 'marker'. Es independiente de desde dónde se ejecute
    el script (src/, raíz del repo, o el runner de Jenkins en producción)."""
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"No se encontró '{marker}' en ningún directorio padre a partir de {current}"
    )


# ---------------------------------------------------------------------------
# Definición de columnas
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    'tipo_credito', 'fecha_prestamo', 'capital_prestado', 'plazo_meses', 'edad_cliente',
    'tipo_laboral', 'salario_cliente', 'total_otros_prestamos', 'cuota_pactada', 'puntaje',
    'puntaje_datacredito', 'cant_creditosvigentes', 'huella_consulta', 'saldo_mora',
    'saldo_total', 'saldo_principal', 'saldo_mora_codeudor', 'creditos_sectorFinanciero',
    'creditos_sectorCooperativo', 'creditos_sectorReal', 'promedio_ingresos_datacredito',
    'tendencia_ingresos', 'Pago_atiempo'
}

TARGET_COLUMN = "Pago_atiempo"

# Variables excluidas explícitamente y por qué:
#   - puntaje: sospecha de data leakage (corr. 0.92 con el target, sin confirmar origen).
#   - tendencia_ingresos: versión cruda/corrupta, se reemplaza por tendencia_ingresos_limpia.
#   - fecha_prestamo: no disponible/comparable al evaluar un cliente nuevo (el crédito
#     nuevo todavía no fue otorgado). No se deriva "antigüedad" por el mismo motivo.
#   - saldo_total, saldo_mora_codeudor: se dejan fuera de esta primera versión del modelo
#     para mantener el set de features acotado; candidatas a sumar en una iteración futura.
#   - creditos_sectorFinanciero/Cooperativo/Real: se excluyen sin reemplazo. Se había
#     considerado un agregado (total_creditos_sectores), pero se descartó: correlaciona
#     0.95 con cant_creditosvigentes, es decir, es prácticamente la misma información ya
#     presente en otra columna, y agregarla sería redundante.
EXCLUDED_COLUMNS = {
    "puntaje", "tendencia_ingresos", "fecha_prestamo", "saldo_total", "saldo_mora_codeudor",
    "creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal",
}

CATEGORIAS_VALIDAS_TENDENCIA = ["Estable", "Creciente", "Decreciente"]
ORDEN_TENDENCIA_INGRESOS = ["Decreciente", "Estable", "Creciente"]

# Rango razonable de edad, según hallazgo del EDA (había registros de hasta 123 años)
EDAD_MIN, EDAD_MAX = 18, 90

# Rangos válidos por variable (mínimo, máximo; None = sin límite), ambos inclusive.
# Un valor fuera de rango se considera IMPOSIBLE (error de carga o código
# especial) y se convierte en nulo, para que lo complete el imputer del
# pipeline. No se recorta: recortar inventaría un valor válido (por ejemplo,
# un salario de 22 mil millones pasaría a "gana 40 millones").
# Los límites son supuestos documentados, no confirmados por el negocio:
#   - edad_cliente: [18, 90]. Había 150 registros con 122-123 años.
#   - puntaje_datacredito: [1, 950]. La escala de DataCrédito llega a 950; el 0
#     (145 registros) se interpreta como "sin puntaje" y hubo un -7 y un 999.
#   - salario_cliente: [100.000, 100.000.000] mensuales. Había 24 salarios en 0,
#     otros por debajo de 100 mil y valores de hasta 22 mil millones.
#   - total_otros_prestamos: hasta 1.000 millones (13 registros de hasta 6.787 M).
#   - promedio_ingresos_datacredito: mayor a 0 (7 registros en 0 = sin dato).
# Los valores extremos pero posibles (por ejemplo, un salario de 60 millones)
# no se tocan acá: los acota después cap_outliers_sin_leakage.
RANGOS_VALIDOS = {
    "edad_cliente": (EDAD_MIN, EDAD_MAX),
    "puntaje_datacredito": (1, 950),
    "salario_cliente": (100_000, 100_000_000),
    "total_otros_prestamos": (None, 1_000_000_000),
    "promedio_ingresos_datacredito": (1, None),
}


# ---------------------------------------------------------------------------
# Carga y limpieza
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """Carga el dataset crudo desde Base_de_datos.xlsx y valida columnas esperadas."""
    repo_root = find_repo_root()
    data_path = repo_root / "Base_de_datos.xlsx"

    logger.info(f"Cargando dataset desde: {data_path}")
    df = pd.read_excel(data_path)

    columnas_faltantes = EXPECTED_COLUMNS - set(df.columns)
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {columnas_faltantes}")

    logger.info(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


def validar_rangos(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte en nulo todo valor fuera de RANGOS_VALIDOS. Son reglas de
    dominio con límites fijos (no se calculan con los datos), así que se pueden
    aplicar a todo el dataset antes del split sin generar data leakage."""
    df = df.copy()
    for col, (minimo, maximo) in RANGOS_VALIDOS.items():
        fuera = pd.Series(False, index=df.index)
        if minimo is not None:
            fuera |= df[col] < minimo
        if maximo is not None:
            fuera |= df[col] > maximo
        if fuera.any():
            logger.warning(
                f"{int(fuera.sum())} valores de '{col}' fuera del rango válido "
                f"[{minimo}, {maximo}], se convierten en nulos."
            )
            df[col] = df[col].mask(fuera)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza definida en comprension_eda.ipynb: corrección de
    'tendencia_ingresos', valores imposibles a nulo (validar_rangos) y tipado."""
    df = validar_rangos(df)

    # tendencia_ingresos: se conservan solo las 3 categorías válidas; los valores
    # numéricos corruptos (y los nulos originales) quedan como NaN para que los
    # impute el preprocesador.
    n_corruptos = (
        df["tendencia_ingresos"].notna() & ~df["tendencia_ingresos"].isin(CATEGORIAS_VALIDAS_TENDENCIA)
    ).sum()
    if n_corruptos:
        logger.warning(f"{n_corruptos} valores inválidos en tendencia_ingresos, se reemplazan por NaN.")
    df["tendencia_ingresos_limpia"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(CATEGORIAS_VALIDAS_TENDENCIA), np.nan
    )

    # Tipo nominal
    df["tipo_credito"] = df["tipo_credito"].astype("category")
    df["tipo_laboral"] = df["tipo_laboral"].astype("category")

    # tendencia_ingresos_limpia se deja como texto (no category todavía): el
    # OrdinalEncoder del preprocesador espera los valores de texto originales
    # para poder mapearlos según ORDEN_TENDENCIA_INGRESOS.

    # Target binario
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los atributos derivados que se decidió incluir en esta iteración."""
    df = df.copy()

    # Relación cuota / salario (proxy de nivel de endeudamiento). División segura:
    # salario <= 0 o nulo se reemplaza por NaN antes de dividir (sin warnings de
    # división por cero) y el ratio queda NaN para que lo impute el preprocesador.
    salario_valido = df["salario_cliente"].where(df["salario_cliente"] > 0)
    df["ratio_cuota_salario"] = df["cuota_pactada"] / salario_valido

    return df


# ---------------------------------------------------------------------------
# Definición de features para el ColumnTransformer
# ---------------------------------------------------------------------------

def get_feature_lists() -> dict:
    """Devuelve las listas de columnas por tipo, ya excluyendo target y columnas
    descartadas (leakage / no disponibles / reemplazadas)."""
    numeric_features = [
        "edad_cliente", "capital_prestado", "plazo_meses", "puntaje_datacredito",
        "huella_consulta", "salario_cliente", "total_otros_prestamos", "cuota_pactada",
        "cant_creditosvigentes", "saldo_mora", "saldo_principal", "promedio_ingresos_datacredito",
        "ratio_cuota_salario",
    ]

    # tipo_credito es nominal: sus códigos no tienen orden real (el diccionario de datos
    # aclara que "no es importante qué tipo de crédito" es cada código).
    categorical_features = ["tipo_laboral", "tipo_credito"]

    # tendencia_ingresos_limpia SÍ es ordinal: Decreciente < Estable < Creciente
    # representa una escala real de evolución de ingresos en el tiempo.
    categorical_ordinal_features = ["tendencia_ingresos_limpia"]

    return {
        "numeric": numeric_features,
        "categorical": categorical_features,
        "categorical_ordinal": categorical_ordinal_features,
    }


# ---------------------------------------------------------------------------
# ColumnTransformer (numeric / categoric / categoric ordinales)
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Arma el ColumnTransformer con una rama por tipo de variable, siguiendo
    el esquema: SimpleImputer (+ Scaler/Encoder según corresponda)."""
    features = get_feature_lists()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    categorical_ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[ORDEN_TENDENCIA_INGRESOS])),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, features["numeric"]),
            ("categoric", categorical_transformer, features["categorical"]),
            ("categoric_ordinal", categorical_ordinal_transformer, features["categorical_ordinal"]),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Acotamiento de outliers sin leakage
# ---------------------------------------------------------------------------

def cap_outliers_sin_leakage(df_train: pd.DataFrame, df_test: pd.DataFrame, column: str, upper_quantile: float = 0.99):
    """Calcula un percentil de corte SOLO sobre df_train y lo aplica a ambos
    conjuntos. Evita que el set de evaluación influya en el umbral usado para
    limpiar el set de entrenamiento (data leakage)."""
    cap = df_train[column].quantile(upper_quantile)
    logger.info(f"Cap de '{column}' calculado sobre train (percentil {upper_quantile}): {cap:,.2f}")
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[column] = df_train[column].clip(upper=cap)
    df_test[column] = df_test[column].clip(upper=cap)
    return df_train, df_test


# ---------------------------------------------------------------------------
# Función principal: expone los datasets de entrenamiento y evaluación
# ---------------------------------------------------------------------------

def get_train_test_data(test_size: float = 0.2, random_state: int = 42):
    """Punto de entrada principal del módulo. Ejecuta todo el flujo de
    ingeniería de características y devuelve los conjuntos de entrenamiento
    y evaluación, ya limpios pero SIN codificar/escalar todavía.

    La codificación/escalado (build_preprocessor) se aplica recién dentro del
    Pipeline de cada modelo en model_training_evaluation.py, ajustándose
    únicamente sobre el set de entrenamiento, para evitar fugas de
    información desde el set de evaluación.

    Se usa split estratificado (stratify=y) porque el target está fuertemente
    desbalanceado (~95%/5%, ver EDA), para que ambos conjuntos mantengan la
    misma proporción de clases. El desbalance en sí se compensa en el
    entrenamiento con sample_weight (ver model_training_evaluation.py), no acá.

    El acotamiento de outliers estadísticos (salario_cliente, ratio_cuota_salario)
    se calcula después del split, usando solo train, para no filtrar información
    del set de evaluación hacia el umbral de corte.

    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrame / pd.Series)
    """
    df = load_raw_data()
    df = clean_data(df)

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COLUMN]
    )

    # salario_cliente: los valores imposibles (0, miles de millones) ya son nulos por
    # validar_rangos; acá se acotan los extremos pero posibles (percentil 99 de train).
    df_train, df_test = cap_outliers_sin_leakage(df_train, df_test, "salario_cliente", upper_quantile=0.99)

    # Las derivadas se calculan DESPUÉS de acotar salario_cliente, para que
    # ratio_cuota_salario ya use el valor corregido y no herede el outlier.
    df_train = add_derived_features(df_train)
    df_test = add_derived_features(df_test)

    # El propio ratio puede seguir siendo extremo por denominadores muy chicos;
    # se acota también, con el mismo criterio (umbral calculado solo con train).
    df_train, df_test = cap_outliers_sin_leakage(df_train, df_test, "ratio_cuota_salario", upper_quantile=0.99)

    features = get_feature_lists()
    feature_columns = features["numeric"] + features["categorical"] + features["categorical_ordinal"]

    X_train = df_train[feature_columns]
    y_train = df_train[TARGET_COLUMN]
    X_test = df_test[feature_columns]
    y_test = df_test[TARGET_COLUMN]

    logger.info(f"Split realizado: train={X_train.shape[0]} filas, test={X_test.shape[0]} filas")
    logger.info(f"Distribución del target en train:\n{y_train.value_counts(normalize=True).round(3)}")
    logger.info(f"Distribución del target en test:\n{y_test.value_counts(normalize=True).round(3)}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Ejecución directa: sanity check rápido del módulo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data()
    preprocessor = build_preprocessor()

    print("\nResumen de feature engineering:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Features numéricas: {len(get_feature_lists()['numeric'])}")
    print(f"  Features categóricas (nominal): {len(get_feature_lists()['categorical'])}")
    print(f"  Features categóricas (ordinal): {len(get_feature_lists()['categorical_ordinal'])}")

    # Sanity check: el preprocesador debe poder ajustarse sobre el set de entrenamiento
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    print(f"  Shape luego de aplicar el ColumnTransformer: {X_train_transformed.shape}")
