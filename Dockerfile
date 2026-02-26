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

# Copy the application code
COPY VirtualScanner.py .
COPY ScannerService.py .
COPY MobileNode.py .
COPY SimpleICP.py .
COPY AlignmentSolver.py .
COPY labModel.obj .
COPY labModel.mtl .

# Default command (can be overridden in docker-compose)
CMD ["python", "ScannerService.py"]
