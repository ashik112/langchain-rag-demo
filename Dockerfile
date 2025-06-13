#── builder stage ─────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update \
 && apt-get install -y \
      build-essential \
      libmagic1 \
      libjpeg-dev \
      zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy everything (including web/, assets/, faiss_index/, etc.)
COPY . .

#── final stage ────────────────────────────────────────────────────────
FROM python:3.11-slim AS app
WORKDIR /app

RUN apt-get update \
 && apt-get install -y \
      libmagic1 \
      libjpeg-dev \
      zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# bring in only what we need
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

EXPOSE 5000
CMD ["python", "web_server.py"]
