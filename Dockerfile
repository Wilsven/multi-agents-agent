# Use an official Python 3.12 runtime as a parent image
FROM python:3.12.9-slim

# Set environment variables consistently
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT="/usr/local/" \
    # Ensure scripts know where the app root is
    PYTHONPATH="/app"

# Install uv using the recommended multi-stage copy
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy the rest of the application code
COPY . /app

# Set the working directory in the container
WORKDIR /app

# Copy only the dependency definition files first
COPY pyproject.toml uv.lock* ./

# Install dependencies
# This relies on pyproject.toml and uv.lock
RUN uv sync --no-dev --frozen --no-cache

# Make scripts executable
RUN chmod +x scripts/*

# Make port 8000 available (adjust if your API uses a different port)
EXPOSE 8001

# Define the default command to run the application
CMD ["uvicorn", "app.main:agent_app", "--host", "0.0.0.0", "--port", "8001"]
