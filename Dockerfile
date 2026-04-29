# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Install requirements (LightGBM and shap can be heavy)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Production Runner
# ==========================================
FROM python:3.10-slim as runner

# LightGBM requires libgomp1 on Debian-based slim systems
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables to optimize Python execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copy API source code and feature engineering module
# The features module is required because joblib unpickles the
# preprocessing pipeline which references features.transformers.DomainFeatureGenerator
COPY src/api/ /app/src/api/
COPY src/features/ /app/src/features/

# Copy MLflow tracking database and model artifacts (Our "Free" Registry)
COPY mlflow.db /app/mlflow.db
COPY mlruns/ /app/mlruns/
COPY models/ /app/models/

# Expose the API port
EXPOSE 8000

# Create a non-root user for security (Best Practice)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Start the FastAPI application via Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]