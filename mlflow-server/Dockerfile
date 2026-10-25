FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet

RUN python -m pip install --no-cache-dir --disable-pip-version-check mlflow==3.17.2

EXPOSE 5000

# Same mount path as the robotics app so run artifact URIs resolve in both containers.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "file:/mnt/mlflow/mlruns"]
