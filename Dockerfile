FROM python:3.11-slim
WORKDIR /app

# Install build deps and pip packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

ENV PORT=8000
EXPOSE 8000

# --timeout 120: /api/institutional's cold path runs a 100-permutation
# block-bootstrap null test, which can exceed gunicorn's 30s default on a
# shared-vCPU machine and get the worker killed before it writes its cache.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
