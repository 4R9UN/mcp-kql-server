# Build Stage
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build from pyproject.toml so package metadata stays in one place.
COPY pyproject.toml README.md LICENSE ./
COPY mcp_kql_server ./mcp_kql_server

RUN python -m pip install --upgrade pip build
RUN python -m build --wheel --outdir /dist

# Runtime Stage
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd -r mcpuser && useradd -r -g mcpuser mcpuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /dist /dist
RUN python -m pip install --no-cache-dir /dist/*.whl

RUN chown -R mcpuser:mcpuser /app /dist

USER mcpuser

ENV PORT=8000
EXPOSE 8000

CMD ["python", "-m", "mcp_kql_server"]
