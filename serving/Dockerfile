FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY serving/requirements.txt /app/serving/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 --retries 10 -r /app/serving/requirements.txt

COPY models/lightgbm_tuned_v2.joblib /app/models/lightgbm_tuned_v2.joblib
COPY models/isotonic_calibrator_v2.joblib /app/models/isotonic_calibrator_v2.joblib
COPY models/serving_constants_v2.json /app/models/serving_constants_v2.json
COPY serving/ /app/serving/

ENV MODEL_PATH=/app/models/lightgbm_tuned_v2.joblib \
    CALIBRATOR_PATH=/app/models/isotonic_calibrator_v2.joblib \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

WORKDIR /app/serving

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=40s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
