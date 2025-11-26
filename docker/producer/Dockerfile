FROM python:3.10

WORKDIR /app

# Copia requirements primero (mejor para cache)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia TODO el código del producer
COPY . .

CMD ["python", "producer.py"]
