FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet \
    MLFLOW_ALLOW_FILE_STORE=true \
    ROBOTICS_DEMO_HOST=0.0.0.0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

COPY app.py mlops.py experiment_tracking.py observability.py operations.py pipelines.py retrain_trigger.py robot_data.py video_privacy.py ./
COPY static ./static
COPY tools/results/ai4i_random_forest_sweep.json ./tools/results/ai4i_random_forest_sweep.json

EXPOSE 8765
CMD ["python", "app.py"]
