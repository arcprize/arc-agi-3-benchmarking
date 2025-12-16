# Dockerfile for ARC-AGI-3 Benchmarking CLI
# Supports building and running cli/main.py (main.py) with all prerequisites

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management (optional, but recommended)
RUN pip install --no-cache-dir uv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock* ./

# Copy the entire project
COPY . .

# Install Python dependencies using uv (or fallback to pip)
# uv.lock is optional - if it doesn't exist, pip will be used
RUN if [ -f uv.lock ]; then \
        uv pip install --system -e .; \
    else \
        pip install --no-cache-dir -e .; \
    fi

# Create directories for results, checkpoints, and logs
RUN mkdir -p results logs .checkpoint

ENV ARC_URL_BASE="https://three.arcprize.org"

# CLI argument environment variables (all can be set via env vars)
ENV GAME_ID=""
ENV CONFIG=""
ENV CHECKPOINT=""
ENV SAVE_RESULTS_DIR=""
ENV OVERWRITE_RESULTS="false"
ENV MAX_ACTIONS="40"
ENV RETRY_ATTEMPTS="3"
ENV RETRIES="3"
ENV NUM_PLAYS="1"
ENV SHOW_IMAGES="false"
ENV USE_VISION="true"
ENV MEMORY_LIMIT=""
ENV CHECKPOINT_FREQUENCY="1"
ENV CLOSE_ON_EXIT="false"
ENV LOG_LEVEL="INFO"
ENV VERBOSE="false"
ENV LIST_CHECKPOINTS="false"
ENV CLOSE_SCORECARD=""

# Default command runs the main CLI
CMD ["python", "main.py"]

