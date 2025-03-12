# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv
RUN apt update && apt install git -y

# Copy the package and dependency files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY README.md ./

# Install dependencies using uv sync
RUN uv sync
RUN uv pip install flash-attn --no-build-isolation

# Command to run the server
# Using --host 0.0.0.0 to make it accessible outside container
# Default validation time of 5 seconds, can be overridden with docker run command
# uv run python src/zeroband/inference.py
ENTRYPOINT ["uv", "run", "python", "src/zeroband/inference.py"]
CMD ["@", "configs/inference/debug.yml"] 
