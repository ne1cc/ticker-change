FROM python:3.12-slim
WORKDIR /app

# Install build deps and pip packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

ENV PORT=8000
EXPOSE 8000

# --timeout 300: slow cold paths must finish before gunicorn kills the worker
# (which resets the connection and nginx returns 502). Two paths need it:
# /api/institutional's 100-permutation bootstrap test, and /ai-summary's
# comprehensive LLM report (multi-minute on a cold cache; each failed
# provider in the fallback chain can burn 90s before the next is tried).
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "300"]
