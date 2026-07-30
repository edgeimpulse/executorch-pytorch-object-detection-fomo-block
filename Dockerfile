# Use a slim Python base. CPU-only training keeps the image small; switch to an
# nvidia/cuda base image if you need GPU acceleration.
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal. Add build tooling here only if a wheel needs it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py train.py ./

# Edge Impulse calls the entrypoint with --data-directory, --out-directory,
# --info-file and any custom parameters from parameters.json.
ENTRYPOINT ["python3", "train.py"]
