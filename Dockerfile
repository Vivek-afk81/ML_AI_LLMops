
# Use a lightweight Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy dependencies first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# FastAPI port
EXPOSE 8000

# Start RecallR API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
