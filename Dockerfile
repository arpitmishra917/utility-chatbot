# Use official Python image
FROM python:3.10-slim

# Use the exact package names that worked in your manual test
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY requirement.txt .
# COPY .env .
COPY src/ src/

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirement.txt

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Start both FastAPI and Streamlit
CMD ["bash", "-c", "uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 & streamlit run src/frontend/streamlit.py --server.port 8501 & wait"]