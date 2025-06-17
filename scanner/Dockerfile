FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install xgboost pandas numpy google-cloud-storage

# Create working directory
RUN mkdir /app
WORKDIR /app

# Copy training code
COPY train.py .

# Run training
ENTRYPOINT ["python", "train.py"]
