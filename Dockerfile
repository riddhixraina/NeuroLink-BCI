# Railway-specific Dockerfile for lightweight deployment
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy lightweight requirements
COPY backend/requirements-railway.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the minimal backend
COPY backend/app_minimal.py ./app.py

# Create necessary directories
RUN mkdir -p logs

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=5000

# Expose port
EXPOSE 5000

# Run the lightweight application
# Run the application
CMD ["python", "app.py"]
