# MLOps Pipeline — Predicción de Comportamiento de Pago de Créditos

Proyecto de MLOps: modelo predictivo de comportamiento de pago de créditos para clientes nuevos.

## Estructura del proyecto

```
mlops_pipeline/
├── src/
│   ├── config.json                    # Configuración del proyecto (project_code)
│   ├── cargar_datos.ipynb             # Carga del dataset de ejemplo
│   ├── comprension_eda.ipynb          # Análisis exploratorio de datos (EDA)
│   ├── ft_engineering.py              # Feature engineering (próximos avances)
│   ├── model_training_evaluation.py   # Entrenamiento y evaluación (próximos avances)
│   ├── model_deploy.py                # Despliegue del modelo (próximos avances)
│   └── model_monitoring.py            # Monitoreo del modelo (próximos avances)
├── Base_de_datos.xlsx                 # Dataset no productivo de ejemplo
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

## Flujo de trabajo (branches)

- `main`: rama estable, protegida. Requiere Pull Request y al menos 1 aprobación para recibir cambios.
- `developer`: rama de desarrollo activo. Acá se trabajan los notebooks `cargar_datos.ipynb` y `comprension_eda.ipynb`.
- `certification`: reservada para etapas de certificación/QA (uso en avances posteriores).

## Estado del proyecto

- [x] V1.0.0 — Estructura base de carpetas
- [x] V1.0.1 — Carga de datos y EDA completo (corrección post-review)

## Hallazgos clave del EDA (V1.0.1)

- Fuerte desbalance de clases en la variable objetivo `Pago_atiempo` (~95% / ~5%).
- Posible **data leakage** en la variable `puntaje` (correlación de 0.92 con el target) — pendiente de verificación con el equipo de datos antes de usarla como feature.
- Los nulos de `tendencia_ingresos` y `promedio_ingresos_datacredito` coinciden casi en su totalidad en las mismas filas, sugiriendo un origen común (falla en la consulta externa a DataCrédito).
- Se detectaron y corrigieron valores inválidos en `tendencia_ingresos` (números filtrados dentro de una columna categórica).
- Se detectaron 150 registros con `edad_cliente` fuera de un rango razonable (hasta 123 años), documentado como regla de validación pendiente.
