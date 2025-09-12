FROM python:3.11-slim

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY app.py .
EXPOSE 8501
# Lancer Streamlit en exposant sur 0.0.0.0 pour accepter les connexions externes
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
