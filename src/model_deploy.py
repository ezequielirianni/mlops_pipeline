"""
model_deploy.py

Cuarta componente del flujo de modelos operativos.
Responsabilidad: disponibilizar el mejor modelo (mejor_modelo.pkl) como un
servicio web con FastAPI, que recibe datos crudos de uno o varios clientes
(predicción por lotes), aplica las transformaciones necesarias y devuelve la
probabilidad de NO pago y la predicción.

Endpoints:
    GET  /             información básica del servicio
    GET  /health       estado del servicio y del modelo cargado
    POST /predict      lote de clientes en JSON
    POST /predict/csv  lote de clientes en un archivo CSV

Validación de entrada (Pydantic): antes de predecir se valida cada cliente.
Si algún valor está fuera de rango (los mismos RANGOS_VALIDOS de
ft_engineering.py), falta un campo obligatorio o una categoría no es válida,
la API NO predice: responde 422 con un mensaje claro por cliente y campo.
Así el modelo nunca recibe datos que no tienen sentido.

Transformaciones aplicadas a los datos ya validados (las mismas que en
ft_engineering.py, para que el modelo reciba lo mismo que en el entrenamiento):
    1. validar_rangos (sin efecto si la validación pasó; se mantiene como resguardo).
    2. tendencia_ingresos -> tendencia_ingresos_limpia.
    3. salario_cliente acotado al percentil 99 de train.
    4. ratio_cuota_salario = cuota_pactada / salario_cliente.
    5. ratio_cuota_salario acotado al percentil 99 de train.
El imputado, el escalado y el encoding los hace el propio pipeline del .pkl.

Los umbrales de los pasos 3 y 5 se recalculan al iniciar el servicio con el
mismo split que usó el entrenamiento (función get_caps), así coinciden
exactamente con los del modelo guardado.

Ejecución local (desde src/):
    uvicorn model_deploy:app --reload
    Documentación interactiva: http://localhost:8000/docs
"""

import io
import logging
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from sklearn.model_selection import train_test_split

from ft_engineering import (
    CATEGORIAS_VALIDAS_TENDENCIA,
    RANGOS_VALIDOS,
    TARGET_COLUMN,
    add_derived_features,
    clean_data,
    find_repo_root,
    get_feature_lists,
    load_raw_data,
    validar_rangos,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_FILENAME = "mejor_modelo.pkl"
CLASE_NO_PAGA = 0
CAP_QUANTILE = 0.99   # mismo percentil que ft_engineering.cap_outliers_sin_leakage
TEST_SIZE = 0.2       # mismo split que ft_engineering.get_train_test_data
RANDOM_STATE = 42

# Columnas crudas que debe enviar el cliente de la API
COLUMNAS_ENTRADA = [
    "edad_cliente", "capital_prestado", "plazo_meses", "puntaje_datacredito", "huella_consulta",
    "salario_cliente", "total_otros_prestamos", "cuota_pactada", "cant_creditosvigentes",
    "saldo_mora", "saldo_principal", "promedio_ingresos_datacredito",
    "tipo_laboral", "tipo_credito", "tendencia_ingresos",
]

# Estado del servicio: se completa una sola vez al iniciar (ver lifespan)
estado = {"model": None, "caps": None}


# ---------------------------------------------------------------------------
# Esquemas de entrada y salida (validación automática con Pydantic)
# ---------------------------------------------------------------------------

def rango(columna: str) -> dict:
    """Traduce RANGOS_VALIDOS de ft_engineering.py a restricciones de Pydantic
    (ge = mayor o igual, le = menor o igual). Un único lugar define los rangos
    para el entrenamiento y para la API."""
    minimo, maximo = RANGOS_VALIDOS[columna]
    return {k: v for k, v in (("ge", minimo), ("le", maximo)) if v is not None}


TIPOS_LABORALES = ("Empleado", "Independiente")


class Cliente(BaseModel):
    """Datos crudos de una solicitud de crédito. Los campos opcionales pueden
    llegar vacíos (null): el pipeline del modelo los imputa. Los valores fuera
    de rango se rechazan con un mensaje claro (ver validation_exception_handler)."""
    edad_cliente: float = Field(..., examples=[35], **rango("edad_cliente"))
    capital_prestado: float = Field(..., gt=0, examples=[2000000])
    plazo_meses: float = Field(..., gt=0, examples=[12])
    puntaje_datacredito: Optional[float] = Field(None, examples=[780], **rango("puntaje_datacredito"))
    huella_consulta: Optional[float] = Field(None, ge=0, examples=[4])
    salario_cliente: Optional[float] = Field(None, examples=[3000000], **rango("salario_cliente"))
    total_otros_prestamos: Optional[float] = Field(None, ge=0, examples=[1000000], **rango("total_otros_prestamos"))
    cuota_pactada: float = Field(..., gt=0, examples=[190000])
    cant_creditosvigentes: Optional[float] = Field(None, ge=0, examples=[5])
    saldo_mora: Optional[float] = Field(None, ge=0, examples=[0])
    saldo_principal: Optional[float] = Field(None, ge=0, examples=[15000])
    promedio_ingresos_datacredito: Optional[float] = Field(None, examples=[1200000], **rango("promedio_ingresos_datacredito"))
    tipo_laboral: Optional[Literal[TIPOS_LABORALES]] = Field(None, examples=["Empleado"])
    tipo_credito: Optional[int] = Field(None, ge=0, examples=[4])
    tendencia_ingresos: Optional[Literal[tuple(CATEGORIAS_VALIDAS_TENDENCIA)]] = Field(None, examples=["Creciente"])


class LoteClientes(BaseModel):
    clientes: List[Cliente] = Field(..., min_length=1)


class Prediccion(BaseModel):
    indice: int
    prob_no_pago: float
    prediccion: int
    etiqueta: str


class RespuestaPrediccion(BaseModel):
    modelo: str
    cantidad: int
    predicciones: List[Prediccion]


# ---------------------------------------------------------------------------
# Carga del modelo y de los umbrales de acotamiento
# ---------------------------------------------------------------------------

def load_model():
    path = find_repo_root() / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Ejecutar antes model_training_evaluation.py.")
    logger.info(f"Cargando modelo desde {path}")
    return joblib.load(path)


def get_caps() -> dict:
    """Recalcula los umbrales del percentil 99 exactamente como en el
    entrenamiento: mismo split estratificado, solo con train, y el del ratio
    después de acotar el salario."""
    df = clean_data(load_raw_data())
    df_train, _ = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN]
    )
    cap_salario = float(df_train["salario_cliente"].quantile(CAP_QUANTILE))
    df_train = df_train.assign(salario_cliente=df_train["salario_cliente"].clip(upper=cap_salario))
    cap_ratio = float(add_derived_features(df_train)["ratio_cuota_salario"].quantile(CAP_QUANTILE))
    return {"salario_cliente": cap_salario, "ratio_cuota_salario": cap_ratio}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y los umbrales una sola vez al iniciar el servicio."""
    estado["model"] = load_model()
    estado["caps"] = get_caps()
    logger.info(f"Servicio listo. Umbrales de acotamiento: {estado['caps']}")
    yield


app = FastAPI(
    title="API de predicción de pago de créditos",
    description="Estima la probabilidad de que un cliente nuevo NO pague a tiempo su crédito. Admite predicción por lotes.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Mensajes de error claros para datos inválidos
# ---------------------------------------------------------------------------

MENSAJE_RECHAZO = (
    "No se puede predecir: hay datos inválidos o fuera de rango. "
    "Corregir los campos indicados y volver a enviar la solicitud."
)


def _formato_numero(valor) -> str:
    if valor is None:
        return "sin límite"
    return f"{valor:,.0f}".replace(",", ".")


def mensaje_campo(campo: str, error: dict) -> str:
    """Traduce un error de Pydantic a un mensaje en español para ese campo."""
    tipo, ctx = error["type"], error.get("ctx", {})
    if campo in RANGOS_VALIDOS and tipo in ("greater_than_equal", "less_than_equal"):
        minimo, maximo = RANGOS_VALIDOS[campo]
        return (
            f"'{campo}' está fuera del rango válido [{_formato_numero(minimo)}, {_formato_numero(maximo)}]; "
            "el modelo no puede predecir con ese valor."
        )
    if tipo == "greater_than_equal":
        return f"'{campo}' debe ser mayor o igual a {ctx.get('ge')}."
    if tipo == "greater_than":
        return f"'{campo}' debe ser mayor a {ctx.get('gt')}."
    if tipo == "missing":
        return f"'{campo}' es obligatorio."
    if tipo == "literal_error":
        opciones = str(ctx.get("expected", "")).replace(" or ", " o ")
        return f"'{campo}' no es una categoría válida. Opciones: {opciones}."
    if tipo in ("float_parsing", "float_type", "int_parsing", "int_type", "int_from_float"):
        return f"'{campo}' debe ser un número."
    return f"'{campo}': {error['msg']}"


def formatear_errores(errores: list, cliente: int = None) -> list:
    """Convierte los errores de Pydantic en una lista {cliente, campo, valor, mensaje}.
    Para JSON el índice del cliente sale de la ubicación del error; para CSV se pasa."""
    salida = []
    for e in errores:
        loc = [p for p in e["loc"] if p != "body"]
        if cliente is None and len(loc) >= 3 and loc[0] == "clientes":
            idx, campo = loc[1], loc[2]
        else:
            idx, campo = cliente, (loc[-1] if loc else "solicitud")
        salida.append({
            "cliente": idx,
            "campo": campo,
            "valor": e.get("input") if e["type"] != "missing" else None,
            "mensaje": mensaje_campo(str(campo), e),
        })
    return salida


def respuesta_rechazo(errores: list) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": MENSAJE_RECHAZO, "errores": errores})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Reemplaza la respuesta 422 genérica de FastAPI por mensajes claros."""
    return respuesta_rechazo(formatear_errores(exc.errors()))


# ---------------------------------------------------------------------------
# Lógica de predicción
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a los datos crudos las mismas transformaciones que ft_engineering.py
    y devuelve las columnas que espera el pipeline del modelo."""
    faltantes = set(COLUMNAS_ENTRADA) - set(df.columns)
    if faltantes:
        raise HTTPException(status_code=422, detail=f"Faltan columnas: {sorted(faltantes)}")

    df = validar_rangos(df[COLUMNAS_ENTRADA].copy())
    caps = estado["caps"]

    df["tendencia_ingresos_limpia"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(CATEGORIAS_VALIDAS_TENDENCIA), np.nan
    )
    df["salario_cliente"] = df["salario_cliente"].clip(upper=caps["salario_cliente"])
    df = add_derived_features(df)
    df["ratio_cuota_salario"] = df["ratio_cuota_salario"].clip(upper=caps["ratio_cuota_salario"])

    df["tipo_credito"] = df["tipo_credito"].astype("category")
    df["tipo_laboral"] = df["tipo_laboral"].astype("category")

    features = get_feature_lists()
    return df[features["numeric"] + features["categorical"] + features["categorical_ordinal"]]


def predict_dataframe(df_crudo: pd.DataFrame) -> RespuestaPrediccion:
    model = estado["model"]
    X = prepare_features(df_crudo)
    idx = list(model.classes_).index(CLASE_NO_PAGA)
    proba = model.predict_proba(X)[:, idx]
    pred = model.predict(X)
    predicciones = [
        Prediccion(
            indice=i,
            prob_no_pago=round(float(p), 4),
            prediccion=int(y),
            etiqueta="No paga a tiempo" if int(y) == CLASE_NO_PAGA else "Paga a tiempo",
        )
        for i, (p, y) in enumerate(zip(proba, pred))
    ]
    return RespuestaPrediccion(
        modelo=type(model.named_steps["classifier"]).__name__,
        cantidad=len(predicciones),
        predicciones=predicciones,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "servicio": "API de predicción de pago de créditos",
        "documentacion": "/docs",
        "endpoints": ["/health", "/predict", "/predict/csv"],
    }


@app.get("/health")
def health():
    model = estado["model"]
    return {
        "estado": "ok" if model is not None else "sin modelo",
        "modelo": type(model.named_steps["classifier"]).__name__ if model is not None else None,
        "columnas_entrada": COLUMNAS_ENTRADA,
        "umbrales_acotamiento": estado["caps"],
    }


@app.post("/predict", response_model=RespuestaPrediccion)
def predict(lote: LoteClientes):
    """Predicción por lotes a partir de una lista de clientes en JSON."""
    df = pd.DataFrame([c.model_dump() for c in lote.clientes])
    return predict_dataframe(df)


@app.post("/predict/csv", response_model=RespuestaPrediccion)
async def predict_csv(archivo: UploadFile = File(..., description="CSV con una fila por cliente")):
    """Predicción por lotes a partir de un archivo CSV con las columnas de entrada."""
    contenido = await archivo.read()
    try:
        df = pd.read_csv(io.BytesIO(contenido))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {exc}")
    if df.empty:
        raise HTTPException(status_code=400, detail="El CSV no tiene filas.")

    faltantes = set(COLUMNAS_ENTRADA) - set(df.columns)
    if faltantes:
        raise HTTPException(status_code=422, detail=f"Faltan columnas: {sorted(faltantes)}")

    # Misma validación que el endpoint JSON, fila por fila (celdas vacías = null)
    clientes, errores = [], []
    filas = df[COLUMNAS_ENTRADA].astype(object).where(df[COLUMNAS_ENTRADA].notna(), None).to_dict("records")
    for i, fila in enumerate(filas):
        try:
            clientes.append(Cliente.model_validate(fila))
        except ValidationError as exc:
            errores.extend(formatear_errores(exc.errors(), cliente=i))
    if errores:
        return respuesta_rechazo(errores)

    return predict_dataframe(pd.DataFrame([c.model_dump() for c in clientes]))
