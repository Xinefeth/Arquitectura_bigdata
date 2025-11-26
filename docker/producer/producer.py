import base64
from hdfs import InsecureClient
from kafka import KafkaProducer

HDFS_URL = "http://namenode:9870"
HDFS_PATH = "/neumonia-data/dataset"
KAFKA_TOPIC = "images-stream"
KAFKA_BROKER = "kafka-broker:9092"

client = InsecureClient(HDFS_URL, user="root")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: v,  # bytes tal cual
)

print("🚀 Producer iniciado. Leyendo imágenes desde HDFS...")

def send_images_from_dir(path):
    items = client.list(path)

    for item in items:
        full_path = f"{path}/{item}"

        if client.status(full_path)['type'] == 'DIRECTORY':
            send_images_from_dir(full_path)
        else:
            with client.read(full_path) as reader:
                img_bytes = reader.read()

            # ⬅︎ AQUÍ: key = nombre/ruta de archivo
            producer.send(
                KAFKA_TOPIC,
                key=full_path.encode("utf-8"),
                value=img_bytes,
            )
            producer.flush()

            print(f"📤 Imagen enviada a Kafka → {full_path}")

if __name__ == "__main__":
    try:
        send_images_from_dir(HDFS_PATH)
        print("✅ Envío completo, cerrando producer.")
    except Exception as e:
        print(f"❌ Error en producer: {e}")
    finally:
        producer.close()
