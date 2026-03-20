# Dockerfile for ai-ml-cheatsheet
# Production-ready Streamlit application

FROM python:3.10-slim

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
# This layer is cached - only rebuilds if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code with correct ownership
# This runs after dependencies, so code changes don't trigger full rebuild
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 7862

# Health check - Docker will know if app is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7862/_stcore/health || exit 1

# Run Streamlit
# Port is set in .streamlit/config.toml (7862)
# --server.address=0.0.0.0: Listen on all interfaces (required for Docker)
# --server.headless=true: Don't try to open browser
# --server.fileWatcherType=none: Disable file watching (production mode)
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]