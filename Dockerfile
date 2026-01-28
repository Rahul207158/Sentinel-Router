# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python Dependencies
# We list them explicitly to keep image small
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    litellm \
    onnxruntime \
    numpy \
    transformers \
    torch --extra-index-url https://download.pytorch.org/whl/cpu 
    # Note: Torch is only needed if transformers tokenizer depends on it, 
    # usually we can skip it for pure ONNX, but distilbert tokenizer might want it.
    # To be safe for V1, we include the CPU version.

# Copy your code
COPY . /app

# Expose the port
EXPOSE 8000

# Run the server
CMD ["python", "server.py"]