FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV & Web diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Run with Gunicorn production server utilizing ASGI Uvicorn workers
CMD ["gunicorn", "-c", "gunicorn_conf.py", "main:app"]
