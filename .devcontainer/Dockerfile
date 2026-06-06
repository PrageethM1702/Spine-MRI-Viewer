FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-annex \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements from project root (build context is ..)
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Expose port for Marimo
EXPOSE 2718

# Default command
CMD ["marimo", "run", "spine_app.py", "--host", "0.0.0.0", "--port", "2718"]
