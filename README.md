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
│   ├── model_monitoring.py            # Monitoreo y detección de data drift
│   ├── app_monitoreo.py               # Tablero de monitoreo en Streamlit
│   └── model_deploy.py                # Despliegue del modelo (próximos avances)
├── Base_de_datos.xlsx                 # Dataset no productivo de ejemplo
├── mejor_modelo.pkl                   # Mejor modelo entrenado (generado por model_training_evaluation.py)
├── drift_report.json                  # Reporte del último batch de monitoreo (generado)
├── drift_history.csv                  # Historial de drift por batch (generado)
├── ultimo_batch_predicciones.csv      # Datos + pronósticos del último batch (generado)
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
- [x] V1.2.0 — Monitoreo de data drift y tablero en Streamlit

## Hallazgos clave del EDA (V1.0.1)

- Fuerte desbalance de clases en la variable objetivo `Pago_atiempo` (~95% / ~5%).
- Posible **data leakage** en la variable `puntaje` (correlación de 0.92 con el target) — pendiente de verificación con el equipo de datos antes de usarla como feature.
- Los nulos de `tendencia_ingresos` y `promedio_ingresos_datacredito` coinciden casi en su totalidad en las mismas filas, sugiriendo un origen común (falla en la consulta externa a DataCrédito).
- Se detectaron y corrigieron valores inválidos en `tendencia_ingresos` (números filtrados dentro de una columna categórica).
- Se detectaron 150 registros con `edad_cliente` fuera de un rango razonable (hasta 123 años), documentado como regla de validación pendiente.

## Feature engineering (V1.1.0)

`ft_engineering.py` aplica la limpieza definida en el EDA y devuelve los conjuntos de entrenamiento y evaluación:

- `tendencia_ingresos_limpia`: se conservan solo las categorías válidas (`Decreciente`, `Estable`, `Creciente`); los valores inválidos pasan a nulo.
- **Valores imposibles → nulos** (`validar_rangos`): en lugar de recortarlos, se convierten en nulos y los completa el imputer del pipeline, para no inventar valores válidos. Rangos (supuestos documentados en el código):
  - `edad_cliente` fuera de [18, 90] → 150 registros (había edades de 122–123 años; aplica la regla pendiente del EDA).
  - `puntaje_datacredito` fuera de [1, 950] → 152 registros (0 = sin puntaje, un -7 y valores por encima de la escala de DataCrédito).
  - `salario_cliente` fuera de [100 mil, 100 millones] → 105 registros (salarios en 0 y de hasta 22 mil millones).
  - `total_otros_prestamos` mayor a 1.000 millones → 13 registros.
  - `promedio_ingresos_datacredito` igual a 0 → 7 registros.
- Split train/test 80/20 **estratificado** (`random_state=42`), realizado **antes** de calcular umbrales estadísticos, para evitar data leakage.
- Los valores extremos pero posibles de `salario_cliente` y `ratio_cuota_salario` se acotan al percentil 99 **calculado solo con train**.
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
| **Gradient Boosting** (seleccionado) | **0.682 ± 0.033** | 0.649 |
| Random Forest | 0.672 ± 0.025 | 0.646 |
| Regresión Logística | 0.660 ± 0.029 | 0.680 |
| XGBoost | 0.649 ± 0.026 | 0.628 |

- Las diferencias entre modelos son menores que la variación entre folds: los 4 tienen un desempeño prácticamente equivalente.
- La precision sobre la clase 0 (~0.09) es baja, pero casi duplica la tasa base (4.7% de clientes que no pagan).
- **Techo de performance (~0.65–0.68):** se explica principalmente por la exclusión de `puntaje` (correlación 0.92 con el target). Sin esa variable, la información disponible para un cliente nuevo tiene poca señal predictiva. Se priorizó un modelo sin leakage aunque su performance sea moderada.
- **Limitación conocida:** los umbrales del percentil 99 se calculan sobre todo train antes de la validación cruzada, lo que genera una filtración mínima entre folds. No afecta la evaluación en test.

El pipeline completo del modelo seleccionado (preprocesamiento + modelo) se guarda con `joblib` en `mejor_modelo.pkl`.

## Monitoreo de data drift (V1.2.0)

`model_monitoring.py` compara la población que llega al modelo contra la población de entrenamiento para detectar cambios que puedan afectar su desempeño.

**Batch simulado.** Como no hay datos productivos, cada ejecución simula el lote que llegaría a producción: toma 1000 registros del histórico no usado para entrenar y les aplica una perturbación controlada y reproducible (la semilla depende del número de batch):

- Numéricas: ruido gaussiano del 10% del desvío estándar típico, y las variables enteras se redondean. El desvío típico se calcula sin el 1% más extremo de cada cola: no es para corregir errores (eso ya lo hace `validar_rangos`), sino porque algunas variables tienen colas largas de valores válidos. Por ejemplo, el desvío común de `total_otros_prestamos` es de ~24.7 millones, pero su dispersión típica es de ~1.6 millones; con el desvío común, el ruido solo generaba drift artificial (PSI ~1.0) en esa variable.
- Desplazamiento deliberado de la media en `puntaje_datacredito` (−0.7 desvíos, unos −32 puntos) y `salario_cliente` (+0.3 desvíos, unos +820 mil), para que el drift sea visible.
- Categóricas: proporciones distintas a las del histórico (mezcla 50/50 con una distribución uniforme).

**Métricas por variable.** Se calculan las cuatro métricas y una de ellas decide el estado:

| Métrica | Aplica a | Umbral crítico | Rol |
|---|---|---|---|
| PSI | Numéricas | > 0.25 (alerta > 0.10) | Decide el estado |
| Jensen-Shannon | Categóricas | > 0.30 (alerta > 0.15) | Decide el estado |
| Kolmogorov-Smirnov | Numéricas | estadístico > 0.25 | Complementaria |
| Chi-cuadrado | Categóricas | p-valor < 0.05 | Complementaria (con muestras grandes marca diferencias muy chicas) |

También se mide el drift de la predicción (PSI sobre la probabilidad de no pago), una señal temprana disponible antes de saber si los clientes pagaron.

**Recomendaciones automáticas:** reentrenar si hay 3 o más variables en drift o drift en la predicción; revisar cada variable en drift (con aviso especial para las que vienen de DataCrédito); monitorear las que están en alerta.

**Salidas** (en la raíz del repo): `drift_report.json` con el último batch, `drift_history.csv` con el historial acumulado y `ultimo_batch_predicciones.csv` con los datos del batch junto con los pronósticos del modelo.

**Hallazgos.** En cada batch se detecta drift en las dos variables desplazadas y en `tipo_credito`, y la probabilidad media de no pago predicha sube unos 4–5 puntos. El sistema recomienda reentrenar. El saneamiento de rangos cambió la lectura del drift: antes, los valores erróneos (puntajes en 0, salarios de miles de millones) inflaban los desvíos estándar, y un desplazamiento de "0.4 desvíos" era en realidad mucho más grande. `promedio_ingresos_datacredito` queda en alerta aunque no se desplazó: el 40% de sus valores está concentrado en una franja angosta, y aun un ruido chico los redistribuye. Es un ejemplo de por qué las variables muy concentradas son sensibles al monitoreo.

```
cd src
python model_monitoring.py --reset --batches 5   # genera 5 batches desde cero
python model_monitoring.py                        # agrega 1 batch nuevo al historial
```

### Tablero en Streamlit — `app_monitoreo.py`

Lee las salidas del monitoreo y toma los umbrales de `model_monitoring.py` (un único lugar de configuración). Tiene tres pestañas:

- **Visualización de métricas:** indicadores generales, barras de riesgo (métrica principal como % de su umbral), tabla de métricas por variable con semáforo y comparación de distribución histórica vs actual.
- **Análisis temporal:** evolución del drift por batch, tabla de tendencias y cambios abruptos, estado de las variables por batch.
- **Recomendaciones y alertas:** mensajes automáticos del último batch y alertas temporales.

Desde la barra lateral se puede ejecutar un batch nuevo.

```
streamlit run src/app_monitoreo.py
```

## Próximos pasos

- Disponibilizar el modelo mediante una API (FastAPI) y empaquetarlo en una imagen Docker.
- Agregar indicadores de faltante al imputer: los registros con valores imposibles tienen una tasa de no pago algo mayor (por ejemplo, 7.3% en edades inválidas y 6.9% en puntaje 0 frente al 4.7% general), información que hoy se pierde al imputar.
- CI/CD: automatizar con GitHub Actions el entrenamiento y el monitoreo en cada Pull Request a `main`.
