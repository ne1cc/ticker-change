FROM python:3.11-slim
WORKDIR /app

# libgomp1 provides libgomp.so.1 (GNU OpenMP), which LightGBM's compiled
# extension dynamically links against at import time. Without it,
# `from lightgbm import LGBMClassifier` raises OSError and ml.py silently
# falls back to sklearn's HistGradientBoostingClassifier -- LightGBM has
# never actually run in this image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install build deps and pip packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "3"]
