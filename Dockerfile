# Use Python 3.13 slim image for smaller size
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create directories for outputs
RUN mkdir -p data models f1_cache

# Create empty checkpoint file if needed
RUN echo '{"successful_sessions": [], "failed_sessions": []}' > data/checkpoint.json

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command - run training pipeline
CMD ["python", "src/f1_prediction_flow.py", "--mode", "training"]
