FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure
COPY common/ ./common/
COPY environment/ ./environment/
COPY agents/ ./agents/
COPY solvers/ ./solvers/
COPY config/ ./config/
COPY replica_office1.ply .

# Default command (overridden by compose)
CMD ["python", "environment/ScannerService.py"]
