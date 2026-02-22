FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.5.8

WORKDIR /app

# Copy everything (source needed for hatchling to build the local package)
COPY . .

# Install dependencies
RUN uv sync --no-dev --frozen --no-cache

CMD uv run uvicorn pdfz.server:app --host 0.0.0.0 --port $PORT
