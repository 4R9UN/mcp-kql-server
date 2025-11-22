# Build Stage
FROM python:3.11-slim as builder

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Create wheels for dependencies
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Runtime Stage
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user
RUN groupadd -r mcpuser && useradd -r -g mcpuser mcpuser

# Install runtime dependencies (curl for healthchecks if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels

# Install dependencies
RUN pip install --no-cache /wheels/*

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R mcpuser:mcpuser /app

# Switch to non-root user
USER mcpuser

# Expose port (Container Apps defaults to 80 or 8080, fastmcp uses SSE usually on a configurable port)
# We'll assume standard MCP port or configure via env var
ENV PORT=8000
EXPOSE 8000

# Run the server
# Assuming the entry point is via the 'mcp-kql-server' command installed by pip (if setup.py exists)
# OR running the module directly.
# Based on project structure, we run the module.
CMD ["python", "-m", "mcp_kql_server.main"]
