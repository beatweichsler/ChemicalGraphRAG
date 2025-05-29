# /cura-app/Dockerfile

# --- Stage 1: Use a stable 'slim' (Debian-based) Python image ---
# Using 3.11 or 3.12 is recommended for best package compatibility.
FROM python:3.13-slim

# --- Stage 2: Set up the environment ---
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- Stage 3: Install system dependencies using apt-get (The Debian way) ---
# This is the correct command for a 'slim' image.
RUN apt-get update && apt-get install -y swig build-essential && rm -rf /var/lib/apt/lists/*

# --- Stage 4: Install Python dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 5: Copy your application and data ---
COPY . .

# --- Stage 6: Expose the port ---
EXPOSE 8501

# --- Stage 7: Define the run command ---
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]