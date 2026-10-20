FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by OpenCV and git for TensorLayerX
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment
ENV TL_BACKEND=torch
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
