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
└── readme.md

```

## Cómo levantar el entorno

1. Cloná el repositorio:

git clone https://github.com/ezequielirianni/mlops_pipeline.git
cd mlops_pipeline


2. Ejecutá el script de setup (Windows):

.\set_up.bat

   Esto crea el entorno virtual `creditos-mlops-venv`, instala las dependencias de `requirements.txt` y registra el kernel de Jupyter como `creditos-mlops-venv Python ETL`.

3. Activá el entorno virtual manualmente si hace falta:

.\creditos-mlops-venv\Scripts\Activate.ps1


4. Abrí los notebooks en Jupyter Lab o VS Code, seleccionando el kernel `creditos-mlops-venv Python ETL`.

## Flujo de trabajo (branches)

- `main`: rama estable. Solo recibe cambios vía Pull Request aprobado.
- `developer`: rama de desarrollo activo.
- `certification`: reservada para etapas de certificación/QA (uso en avances posteriores).

## Estado del proyecto

- [x] V1.0.0 — Estructura base de carpetas
- [ ] V1.0.1 — Carga de datos y EDA (en curso)