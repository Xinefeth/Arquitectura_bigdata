FROM apache/spark:3.4.2

USER root
WORKDIR /opt/spark/work-dir

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app
COPY streaming.py train_spark.py /app/
