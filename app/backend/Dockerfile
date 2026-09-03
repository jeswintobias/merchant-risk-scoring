FROM python:3.12-slim

WORKDIR /app

# Install build tools for compiling ML packages (xgboost, lightgbm) if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# We copy it from the root directory context
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary source directories
COPY app/backend /app/app/backend
COPY src /app/src
COPY models/final /app/models/final
COPY models/artifacts /app/models/artifacts
COPY outputs /app/outputs

# Set python path so src module can be resolved
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
