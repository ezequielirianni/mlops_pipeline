# Imagen de la API de predicción de pago de créditos (FastAPI + Uvicorn)

# 1. Imagen base liviana (slim). Se usa Python 3.12 y no 3.11 porque las
#    versiones fijadas de numpy (2.5.3) requieren Python >= 3.12, y la API debe
#    cargar el modelo con las mismas versiones con las que se entrenó.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2. Carpeta interna donde vive la aplicación
WORKDIR /app

# 3. Dependencias antes que el código: si solo cambia el código, Docker
#    reutiliza la capa de instalación. Se copia el requirements mínimo de la
#    API con el nombre requirements.txt (además, find_repo_root() usa ese
#    archivo para ubicar la raíz del proyecto).
COPY requirements-api.txt requirements.txt

# 4. Instalación sin guardar la caché de pip (imagen más liviana)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Archivos que necesita la API: su código, el módulo de feature engineering
#    que importa, el modelo entrenado y el dataset (con el que recalcula al
#    iniciar los umbrales de acotamiento del entrenamiento)
COPY src/model_deploy.py src/ft_engineering.py ./src/
COPY mejor_modelo.pkl Base_de_datos.xlsx ./

# 6. Puerto en el que escucha Uvicorn dentro del contenedor
EXPOSE 8000

# 7. Uvicorn como proceso principal del contenedor
CMD ["uvicorn", "model_deploy:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
