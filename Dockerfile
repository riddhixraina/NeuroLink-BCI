# Dockerfile for NeuroLink-BCI Backend
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY backend/ ./backend/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p logs uploads static

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run the application
CMD ["gunicorn", "--worker-class", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "backend.app:app"]
