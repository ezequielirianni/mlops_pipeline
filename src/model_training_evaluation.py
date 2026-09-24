"""
model_training_evaluation.py

Segunda componente del flujo de creación de modelos operativos.
Responsabilidad: entrenar los modelos supervisados candidatos sobre las
features generadas por ft_engineering.py, evaluarlos con métricas
consistentes, compararlos gráficamente y seleccionar el de mejor performance.

Métrica principal de selección: ROC-AUC. Con un target desbalanceado
(~95%/5%, ver comprension_eda.ipynb), accuracy es engañosa -- un modelo que
siempre predice "paga a tiempo" ya tendría ~95% de accuracy sin haber
aprendido nada. ROC-AUC no depende del umbral de clasificación ni de la
proporción de clases, por eso es la métrica que decide el modelo ganador.
F1, precision y recall se reportan igual, como métricas secundarias de
diagnóstico (por ejemplo, para entender el trade-off entre falsos positivos
y falsos negativos de cada modelo).

Clase de interés: 0 (NO paga a tiempo). Es la clase minoritaria (~5%) y la
que el negocio necesita detectar. Por defecto sklearn calcula F1/precision/
recall sobre la clase 1 ("paga a tiempo"), lo que da precision ~0.96 en
cualquier modelo solo porque el 95% de los clientes paga: no informa nada.
Por eso todas las métricas se calculan con pos_label=POSITIVE_CLASS=0:
  - recall    = de los clientes que NO pagan, qué % detecta el modelo.
  - precision = de los clientes que el modelo marca como riesgosos, qué %
                realmente no paga.
ROC-AUC no cambia al invertir la clase (es simétrica), pero se calcula con
la probabilidad de la clase 0 por consistencia.

Estrategia de selección: validación cruzada estratificada (5 folds) sobre
train. El modelo ganador se elige por el ROC-AUC promedio entre folds, no por
el resultado en test. Motivo: test tiene solo ~100 clientes de la clase 0, y
con tan pocos casos el ranking entre modelos en un único split es inestable
(las diferencias entre modelos, ~0.02, son menores que la variación entre
folds, ~0.03). Test queda reservado para la evaluación final: todos los
modelos se reentrenan con el train completo y se reportan sobre test, pero
ese resultado no participa en la elección.

No se usa GridSearchCV ni búsqueda de hiperparámetros: se fijan valores
manuales razonables por modelo (ver get_candidate_models), documentando el porqué.

Nota sobre la performance esperada: los modelos quedan en un ROC-AUC de
~0.65-0.68. Es un techo que imponen los datos, no los algoritmos: la variable
'puntaje' (correlación 0.92 con el target) se excluyó por sospecha de data
leakage (ver comprension_eda.ipynb), y sin ella las features disponibles para
un cliente nuevo tienen poca señal predictiva. Los 4 modelos, con distintos
hiperparámetros, convergen al mismo rango.
"""

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ft_engineering import build_preprocessor, find_repo_root, get_train_test_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42  # misma semilla que ft_engineering.get_train_test_data, para reproducibilidad

# Clase sobre la que se calculan las métricas: 0 = NO paga a tiempo (ver docstring del módulo)
POSITIVE_CLASS = 0

# El modelo se guarda en la raíz del repo, al mismo nivel que Base_de_datos.xlsx,
# independientemente de desde dónde se ejecute el script.
MODEL_FILENAME = "mejor_modelo.pkl"

# Cantidad de folds de la validación cruzada. Con 5 folds cada fold de
# validación tiene ~80 clientes de la clase 0 (clase minoritaria): menos folds
# darían pocas estimaciones, más folds dejarían muy pocos positivos por fold.
N_FOLDS = 5


# ---------------------------------------------------------------------------
# build_model: función reutilizable para armar cada pipeline (preprocesador + modelo)
# ---------------------------------------------------------------------------

def build_model(estimator) -> Pipeline:
    """Arma el pipeline completo (ColumnTransformer + clasificador) para un
    estimador dado. Reutilizada para los 4 modelos candidatos, así el
    preprocesamiento (imputación, escalado, encoding) es idéntico y se ajusta
    únicamente sobre el set de entrenamiento, nunca sobre test."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", estimator),
    ])


def get_candidate_models() -> dict:
    """Define los 4 modelos candidatos con hiperparámetros fijados a mano
    (sin GridSearchCV). Los valores elegidos son razonables para un dataset
    de ~8600 filas de entrenamiento con fuerte desbalance de clases, no un
    óptimo buscado exhaustivamente.
    """
    return {
        "Regresion Logistica": LogisticRegression(
            max_iter=1000,          # el default (100) no siempre converge con muchas dummies del OneHot
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,       # suficientes árboles para estabilizar sin disparar tiempo de cómputo
            max_depth=8,            # limita overfitting dado que el dataset no es enorme
            min_samples_leaf=5,     # evita hojas hiperespecíficas en la clase minoritaria
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,            # GB usa árboles chicos (stumps/shallow) por diseño
            learning_rate=0.05,     # learning rate bajo + más estimadores, más estable que agresivo
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# summarize_classification: función reutilizable para evaluar cada modelo
# ---------------------------------------------------------------------------

def summarize_classification(model_name: str, y_true, y_pred, y_proba_pos) -> dict:
    """Calcula el set completo de métricas para un modelo ya entrenado y
    evaluado sobre test. Se llama una vez por modelo, evitando repetir el
    mismo bloque de cálculo 4 veces.

    Todas las métricas se calculan sobre POSITIVE_CLASS (0 = no paga a tiempo).
    y_proba_pos debe ser la probabilidad predicha para esa misma clase.

    ROC-AUC es la métrica de selección (ver docstring del módulo). El resto
    se reporta como diagnóstico secundario. Accuracy no depende de la clase
    positiva y se deja solo como referencia.
    """
    y_true_pos = (y_true == POSITIVE_CLASS).astype(int)  # 1 = cliente que NO paga

    metrics = {
        "modelo": model_name,
        "roc_auc": roc_auc_score(y_true_pos, y_proba_pos),
        "f1": f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    logger.info(
        f"[{model_name}] ROC-AUC={metrics['roc_auc']:.4f} | F1={metrics['f1']:.4f} | "
        f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | "
        f"Accuracy={metrics['accuracy']:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Helpers comunes a validación cruzada y evaluación final
# ---------------------------------------------------------------------------

def get_balanced_sample_weight(y) -> np.ndarray:
    """Pesos por fila para compensar el desbalance de clases.

    Se usa sample_weight en vez de class_weight: class_weight solo lo aceptan
    LogisticRegression y RandomForest; GradientBoosting no lo tiene y XGBoost
    usa otro parámetro (scale_pos_weight). sample_weight con criterio
    "balanced" le da a cada fila un peso inversamente proporcional a la
    frecuencia de su clase, y se pasa en el fit() de los 4 modelos por igual,
    así la comparación entre ellos es justa.

    Se calcula siempre sobre el conjunto con el que se entrena en ese momento
    (cada fold de CV, o el train completo), nunca sobre datos de validación.
    """
    return compute_sample_weight(class_weight="balanced", y=y)


def get_positive_proba(pipeline: Pipeline, X) -> np.ndarray:
    """Probabilidad predicha para POSITIVE_CLASS (0 = no paga a tiempo),
    buscando su columna por etiqueta en pipeline.classes_ y no por posición fija."""
    idx_pos = list(pipeline.classes_).index(POSITIVE_CLASS)
    return pipeline.predict_proba(X)[:, idx_pos]


# ---------------------------------------------------------------------------
# Selección: validación cruzada estratificada sobre train
# ---------------------------------------------------------------------------

def cross_validate_models(X_train, y_train) -> pd.DataFrame:
    """Evalúa cada modelo candidato con StratifiedKFold (N_FOLDS) usando solo
    el set de entrenamiento. En cada fold se arma un pipeline nuevo, se
    calculan los pesos de clase con ese fold y se mide ROC-AUC sobre la parte
    de validación. El preprocesador también se ajusta dentro de cada fold
    (va dentro del Pipeline), así la validación no ve datos del propio fold.

    Devuelve un DataFrame con la media y el desvío del ROC-AUC por modelo,
    ordenado de mayor a menor media: es la tabla que decide el modelo ganador.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_train_pos = (y_train == POSITIVE_CLASS).astype(int)
    resultados = []

    for nombre, estimator in get_candidate_models().items():
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_tr, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_tr = y_train.iloc[train_idx]

            pipeline = build_model(clone(estimator))
            pipeline.fit(
                X_fold_tr, y_fold_tr,
                classifier__sample_weight=get_balanced_sample_weight(y_fold_tr),
            )
            scores.append(roc_auc_score(y_train_pos.iloc[val_idx], get_positive_proba(pipeline, X_fold_val)))

        media, desvio = float(pd.Series(scores).mean()), float(pd.Series(scores).std(ddof=0))
        logger.info(f"[CV {nombre}] ROC-AUC={media:.4f} +- {desvio:.4f} (folds: {[round(s, 3) for s in scores]})")
        resultados.append({"modelo": nombre, "cv_roc_auc_mean": media, "cv_roc_auc_std": desvio})

    return pd.DataFrame(resultados).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluación final: reentrenamiento con train completo y medición en test
# ---------------------------------------------------------------------------

def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """Reentrena cada modelo candidato con el train completo y lo evalúa sobre
    test. Estas métricas son el reporte final de performance, pero NO se usan
    para elegir el modelo (eso lo decide cross_validate_models).

    Devuelve:
    - una tabla (DataFrame) con todas las métricas de test por modelo
    - un diccionario {nombre_modelo: pipeline_entrenado} para graficar y para
      guardar el modelo ganador sin tener que reentrenar
    """
    sample_weight = get_balanced_sample_weight(y_train)
    resultados = []
    pipelines_entrenados = {}

    for nombre, estimator in get_candidate_models().items():
        logger.info(f"Entrenando con train completo: {nombre}")
        pipeline = build_model(estimator)
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)

        y_pred = pipeline.predict(X_test)
        y_proba_pos = get_positive_proba(pipeline, X_test)

        resultados.append(summarize_classification(nombre, y_test, y_pred, y_proba_pos))
        pipelines_entrenados[nombre] = pipeline

    return pd.DataFrame(resultados), pipelines_entrenados


# ---------------------------------------------------------------------------
# Gráficos comparativos
# ---------------------------------------------------------------------------

def plot_comparison_dupla(pipelines_entrenados: dict, tabla_resumen: pd.DataFrame, X_test, y_test):
    """Gráfico doble de la evaluación final en test: a la izquierda, las curvas
    ROC de los 4 modelos superpuestas; a la derecha, F1/Precision/Recall lado a
    lado por modelo, como contexto secundario. Ambos paneles se calculan sobre
    la clase 0 (no paga a tiempo). Es un reporte: la selección ya la hizo la
    validación cruzada."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Izquierda: curvas ROC (clase positiva = no paga a tiempo)
    for nombre, pipeline in pipelines_entrenados.items():
        RocCurveDisplay.from_estimator(
            pipeline, X_test, y_test, pos_label=POSITIVE_CLASS, ax=axes[0], name=nombre
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC=0.5)")
    axes[0].set_title("Curvas ROC en test - Clase 0 (no paga a tiempo)")
    axes[0].legend(loc="lower right")

    # Derecha: métricas secundarias
    metricas_secundarias = tabla_resumen.set_index("modelo")[["f1", "precision", "recall"]]
    metricas_secundarias.plot(kind="bar", ax=axes[1], color=["#DD8452", "#55A868", "#C44E52"])
    axes[1].set_title("Métricas secundarias en test - Clase 0 (no paga a tiempo)")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def print_roc_auc_table(tabla_cv: pd.DataFrame, tabla_test: pd.DataFrame) -> pd.DataFrame:
    """Tabla simple y directa: ROC-AUC de validación cruzada (el que decide,
    ver docstring del módulo) junto al ROC-AUC de test (reporte final),
    ordenada por la media de CV."""
    tabla = tabla_cv.merge(tabla_test[["modelo", "roc_auc"]], on="modelo")
    tabla = tabla.rename(columns={
        "modelo": "Modelo",
        "cv_roc_auc_mean": "ROC-AUC CV (media)",
        "cv_roc_auc_std": "CV (desvío)",
        "roc_auc": "ROC-AUC test",
    }).round(4)
    print(f"\nTabla de comparación por ROC-AUC (selección por CV {N_FOLDS} folds, test solo como reporte):")
    print(tabla.to_string(index=False))
    return tabla


# ---------------------------------------------------------------------------
# Selección del mejor modelo
# ---------------------------------------------------------------------------

def select_best_model(tabla_cv: pd.DataFrame, pipelines_entrenados: dict):
    """Selecciona el modelo con mayor ROC-AUC promedio en validación cruzada
    (ver docstring del módulo) y guarda en disco, con joblib, su pipeline ya
    reentrenado con el train completo, para que model_deploy.py pueda
    cargarlo sin tener que reentrenar."""
    mejor_nombre = tabla_cv.iloc[0]["modelo"]
    mejor_pipeline = pipelines_entrenados[mejor_nombre]
    logger.info(
        f"Modelo seleccionado: {mejor_nombre} "
        f"(ROC-AUC CV={tabla_cv.iloc[0]['cv_roc_auc_mean']:.4f} +- {tabla_cv.iloc[0]['cv_roc_auc_std']:.4f})"
    )

    output_path = find_repo_root() / MODEL_FILENAME
    joblib.dump(mejor_pipeline, output_path)
    logger.info(f"Pipeline completo (preprocesador + modelo) guardado en: {output_path}")

    return mejor_nombre, mejor_pipeline


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data()

    # 1) Selección: validación cruzada sobre train
    tabla_cv = cross_validate_models(X_train, y_train)

    # 2) Evaluación final: reentrenamiento con train completo y reporte en test
    tabla_test, pipelines_entrenados = train_and_evaluate_all(X_train, X_test, y_train, y_test)

    # Las tablas y gráficos de test se muestran en el orden del ranking de CV
    orden = tabla_cv["modelo"].tolist()
    tabla_test = tabla_test.set_index("modelo").loc[orden].reset_index()

    print("\nMétricas en test (clase 0 = no paga a tiempo), en orden del ranking de CV:")
    print(tabla_test.to_string(index=False))

    plot_comparison_dupla(pipelines_entrenados, tabla_test, X_test, y_test)
    print_roc_auc_table(tabla_cv, tabla_test)

    # 3) Guardado del ganador (elegido por CV)
    mejor_nombre, mejor_pipeline = select_best_model(tabla_cv, pipelines_entrenados)
    print(f"\nModelo de mejor performance: {mejor_nombre}")
