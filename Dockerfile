FROM ghcr.io/astral-sh/uv:0.11.2 AS uv

FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl ca-certificates git openssl libssl-dev \
    libjpeg-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev \
    tcl8.6-dev tk8.6-dev python3-tk \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

RUN python -m pip install --no-cache-dir --upgrade pip

# Copy dependency files first for caching
COPY pyproject.toml uv.lock* ./

# ---- create non-root user ----
RUN useradd -m -u 10001 appuser

# Install deps (do this as root, then chown site-packages not needed; just chown /app)
COPY . .
RUN test -f uv.lock
RUN uv pip install --system -e .

RUN mkdir -p results logs .checkpoint \
  && chown -R appuser:appuser /app

ENV ARC_URL_BASE="https://arcprize.org"
ENV PYTHONUNBUFFERED=1

# Switch to non-root for runtime (this is the important part)
USER appuser

CMD ["python", "main.py"]
