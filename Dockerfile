# Use the official Python 3.10 image as a parent image
FROM python:3.10-slim

# Set environment variables to reduce Python package issues and ensure output is sent straight to the terminal without buffering it first
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory to /app
WORKDIR /app

# Copy main.py to /app
COPY main.py .

# Copy secrets.json to /app
COPY secrets.json .

# Install any needed packages specified in requirements.txt
# Assuming pyspacer is available on PyPI and has a requirements.txt file
# If pyspacer is not on PyPI, you might need to copy it into the container and install it manually
RUN pip install --no-cache-dir pyspacer

# Run main.py when the container launches
CMD ["python", "main.py"]

