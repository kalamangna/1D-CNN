# Use Python 3.12 slim image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Command to run the application using Gunicorn
RUN pip install gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
