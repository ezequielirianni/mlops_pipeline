# MLOps Pipeline — Predicción de Comportamiento de Pago de Créditos

Proyecto de MLOps: modelo predictivo de comportamiento de pago de créditos para clientes nuevos.

## Caso de negocio

Una empresa financiera necesita anticipar, **al momento de otorgar un crédito**, si un cliente nuevo va a pagar a tiempo. El modelo estima la probabilidad de que un cliente **no pague a tiempo** (`Pago_atiempo = 0`), que es la clase que el negocio necesita detectar para decidir el otorgamiento o sus condiciones.

Por eso solo se usan variables disponibles antes de otorgar el crédito: se descartan las que dependen de la historia del préstamo (fecha, antigüedad) o que podrían contener información del resultado.

## Estructura del proyecto

```
mlops_pipeline/
├── src/
│   ├── config.json                    # Configuración del proyecto (project_code)
│   ├── cargar_datos.ipynb             # Carga del dataset de ejemplo
│   ├── comprension_eda.ipynb          # Análisis exploratorio de datos (EDA)
│   ├── ft_engineering.py              # Limpieza, feature engineering y split train/test
│   ├── model_training_evaluation.py   # Entrenamiento, evaluación y selección del modelo
│   ├── model_deploy.py                # Despliegue del modelo (próximos avances)
│   └── model_monitoring.py            # Monitoreo del modelo (próximos avances)
├── Base_de_datos.xlsx                 # Dataset no productivo de ejemplo
├── mejor_modelo.pkl                   # Mejor modelo entrenado (generado por model_training_evaluation.py)
├── requirements.txt                   # Dependencias del proyecto
├── set_up.bat                         # Script de setup del entorno virtual (Windows)
├── .gitignore
└── README.md
```

## Cómo levantar el entorno

1. Cloná el repositorio:

```
git clone https://github.com/ezequielirianni/mlops_pipeline.git
cd mlops_pipeline
```

2. Ejecutá el script de setup (Windows):

```
.\set_up.bat
```

Esto crea el entorno virtual `creditos-mlops-venv`, instala las dependencias de `requirements.txt` y registra el kernel de Jupyter como `creditos-mlops-venv Python ETL`.

3. Activá el entorno virtual manualmente si hace falta:

```
.\creditos-mlops-venv\Scripts\Activate.ps1
```

4. Abrí los notebooks en Jupyter Lab o VS Code, seleccionando el kernel `creditos-mlops-venv Python ETL`.

## Cómo entrenar el modelo

Con el entorno activado:

```
cd src
python model_training_evaluation.py
```

El script toma los datos ya procesados por `ft_engineering.py`, entrena y compara los modelos candidatos, muestra las métricas y el gráfico comparativo, y guarda el mejor modelo en `mejor_modelo.pkl` (raíz del repo). El resultado es reproducible (`random_state=42`): volver a ejecutarlo con los mismos datos genera el mismo modelo y sobrescribe el archivo.

Para un chequeo rápido solo del feature engineering: `python ft_engineering.py`.

## Flujo de trabajo (branches)

- `main`: rama estable, protegida. Requiere Pull Request y al menos 1 aprobación para recibir cambios.
- `developer`: rama de desarrollo activo.
- `certification`: reservada para etapas de certificación/QA (uso en avances posteriores).

## Estado del proyecto

- [x] V1.0.0 — Estructura base de carpetas
- [x] V1.0.1 — Carga de datos y EDA completo (corrección post-review)
- [x] V1.1.0 — Feature engineering, entrenamiento y evaluación de modelos

## Hallazgos clave del EDA (V1.0.1)

- Fuerte desbalance de clases en la variable objetivo `Pago_atiempo` (~95% / ~5%).
- Posible **data leakage** en la variable `puntaje` (correlación de 0.92 con el target) — pendiente de verificación con el equipo de datos antes de usarla como feature.
- Los nulos de `tendencia_ingresos` y `promedio_ingresos_datacredito` coinciden casi en su totalidad en las mismas filas, sugiriendo un origen común (falla en la consulta externa a DataCrédito).
- Se detectaron y corrigieron valores inválidos en `tendencia_ingresos` (números filtrados dentro de una columna categórica).
- Se detectaron 150 registros con `edad_cliente` fuera de un rango razonable (hasta 123 años), documentado como regla de validación pendiente.

## Feature engineering (V1.1.0)

`ft_engineering.py` aplica la limpieza definida en el EDA y devuelve los conjuntos de entrenamiento y evaluación:

- `tendencia_ingresos_limpia`: se conservan solo las categorías válidas (`Decreciente`, `Estable`, `Creciente`); los valores inválidos pasan a nulo.
- `edad_cliente` acotada al rango [18, 90] (aplica la regla pendiente del EDA).
- `puntaje_datacredito` acotado a ≥ 0 (se encontró 1 registro con valor negativo).
- Split train/test 80/20 **estratificado** (`random_state=42`), realizado **antes** de calcular umbrales estadísticos, para evitar data leakage.
- `salario_cliente` (valores de hasta 22 mil millones) y `ratio_cuota_salario` se acotan al percentil 99 **calculado solo con train**.
- Feature derivada: `ratio_cuota_salario` (cuota / salario), proxy del nivel de endeudamiento.

**Features finales (16):** 13 numéricas (imputación + escalado), 2 categóricas nominales `tipo_laboral` y `tipo_credito` (imputación + OneHot) y 1 ordinal `tendencia_ingresos_limpia` (imputación + OrdinalEncoder).

**Variables excluidas:** `puntaje` (sospecha de data leakage, no se confirmó su origen), `fecha_prestamo` (no disponible para un cliente nuevo), `tendencia_ingresos` cruda, `saldo_total`, `saldo_mora_codeudor` y las columnas de créditos por sector (su total correlaciona 0.95 con `cant_creditosvigentes`, información redundante).

## Entrenamiento y evaluación (V1.1.0)

`model_training_evaluation.py` entrena 4 modelos: **Regresión Logística, Random Forest, Gradient Boosting y XGBoost**, con hiperparámetros fijados manualmente y justificados en el código.

- **Desbalance de clases:** `sample_weight` balanceado en el entrenamiento de los 4 modelos (`class_weight` no está disponible en Gradient Boosting ni XGBoost).
- **Métrica principal: ROC-AUC**, porque accuracy es engañosa con un 95% de clientes que pagan.
- **Métricas secundarias:** F1, precision y recall calculadas sobre la **clase 0 (no paga a tiempo)**, que es la que interesa detectar.
- **Selección del modelo:** por validación cruzada estratificada de 5 folds sobre train (ROC-AUC promedio). Test se usa solo para el reporte final, porque con ~100 casos de la clase 0 un único split no es suficiente para ordenar los modelos de forma confiable.

### Resultados

| Modelo | ROC-AUC CV (media ± desvío) | ROC-AUC test |
|---|---|---|
| **Gradient Boosting** (seleccionado) | **0.682 ± 0.028** | 0.645 |
| Random Forest | 0.676 ± 0.024 | 0.648 |
| Regresión Logística | 0.655 ± 0.035 | 0.675 |
| XGBoost | 0.652 ± 0.023 | 0.632 |

- Las diferencias entre modelos son menores que la variación entre folds: los 4 tienen un desempeño prácticamente equivalente.
- La precision sobre la clase 0 (~0.09) es baja, pero casi duplica la tasa base (4.7% de clientes que no pagan).
- **Techo de performance (~0.65–0.68):** se explica principalmente por la exclusión de `puntaje` (correlación 0.92 con el target). Sin esa variable, la información disponible para un cliente nuevo tiene poca señal predictiva. Se priorizó un modelo sin leakage aunque su performance sea moderada.
- **Limitación conocida:** los umbrales del percentil 99 se calculan sobre todo train antes de la validación cruzada, lo que genera una filtración mínima entre folds. No afecta la evaluación en test.

El pipeline completo del modelo seleccionado (preprocesamiento + modelo) se guarda con `joblib` en `mejor_modelo.pkl`.
