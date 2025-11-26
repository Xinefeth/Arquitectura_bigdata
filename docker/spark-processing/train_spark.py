import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, Model

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when
from sklearn.model_selection import train_test_split

# ========= CONFIG =========
HDFS_IMAGES = "hdfs://namenode:9000/neumonia-data"

LOCAL_MODEL_DIR = "/models"
HDFS_MODEL_DIR = "/models"          # ruta en HDFS
MODEL_NAME = "efficientnetv2b0_neumonia.keras"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 2

# 🔹 Límite de imágenes para no reventar memoria
MAX_IMAGES = 500   # BAJAMOS A 500 PARA PROBAR SEGURO


# ========= UTILIDADES =========
def decode_and_resize(image_bytes: bytes) -> np.ndarray:
    """Decodifica JPEG, pasa a escala de grises, redimensiona y normaliza [0,1]."""
    img = tf.io.decode_jpeg(image_bytes, channels=1)  # (H, W, 1)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()  # (128, 128, 1)


def build_model() -> Model:
    """Construye EfficientNetV2B0 para clasificación binaria (neumonía / normal)."""
    base = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )

    # Congelamos el backbone para transfer learning básico
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input_gray")
    # Replicamos el canal gris a 3 canales para encajar con pesos ImageNet
    x = layers.Concatenate(axis=-1)([inputs, inputs, inputs])  # (H, W, 3)
    x = base(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="efficientnetv2b0_neumonia")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ========= MAIN =========
def main():
    spark = (
        SparkSession.builder
        .appName("train-efficientnetv2b0-neumonia")
        # Apagar vectorized reader por si acaso
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        # 🔹 aumentar tamaño máximo de resultado al driver
        .config("spark.driver.maxResultSize", "2g")
        # 🔹 bajar número de particiones para no tener 190 tasks
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.default.parallelism", "20")
        .getOrCreate()
    )

    print("=== ENTRENAMIENTO CNN (EfficientNetV2B0) ===")
    print("Leyendo imágenes desde HDFS:", HDFS_IMAGES)

    # 1) Leer imágenes directamente desde HDFS
    df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpeg")
        .option("recursiveFileLookup", "true")
        .load(HDFS_IMAGES)
    )
    # df tiene: path, modificationTime, length, content (bytes)

    # 2) Sacar etiqueta desde el path
    df = (
        df
        .withColumn("label_str", regexp_extract(col("path"), r"/(NORMAL|PNEUMONIA)/", 1))
        .withColumn("label", when(col("label_str") == "PNEUMONIA", 1).otherwise(0))
        .select("content", "label")
    )

    total_imgs = df.count()
    print("Total de imágenes encontradas:", total_imgs)

    # 🔹 limitar el número de imágenes para no exceder memoria/driver
    if total_imgs > MAX_IMAGES:
        print(
            f"⚠️ Dataset grande ({total_imgs}). "
            f"Se usarán solo {MAX_IMAGES} imágenes para el entrenamiento."
        )
        # limit + coalesce para que no cree demasiadas particiones
        df = df.limit(MAX_IMAGES).coalesce(10)
    else:
        print("Usando todas las imágenes para entrenamiento.")
        df = df.coalesce(10)

    # 3) Traer a NumPy para entrenar con TensorFlow
    print("⌛ Collect al driver (con límite / pocas particiones)...")
    rows = df.collect()

    X_list = []
    y_list = []

    for row in rows:
        # content viene como bytearray -> lo convertimos a bytes para TensorFlow
        img_bytes = bytes(row["content"])
        label = float(row["label"])

        img_arr = decode_and_resize(img_bytes)  # (128, 128, 1)
        X_list.append(img_arr)
        y_list.append(label)

    X = np.stack(X_list, axis=0).astype("float32")  # (N, 128, 128, 1)
    y = np.array(y_list, dtype="float32")           # (N,)

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)

    # 4) Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # 5) Construir modelo y entrenar
    model = build_model()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # 6) Guardar modelo en el contenedor
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    local_model_path = os.path.join(LOCAL_MODEL_DIR, MODEL_NAME)
    model.save(local_model_path)
    print("✅ Modelo guardado en:", local_model_path)

    try:
        from hdfs import InsecureClient

        client = InsecureClient("http://namenode:9870", user="root")
        with open(local_model_path, "rb") as f:
            client.write(
                f"{HDFS_MODEL_DIR}/{MODEL_NAME}",
                f,
                overwrite=True,
            )
        print(
            "✅ Modelo subido a HDFS en:",
            f"{HDFS_MODEL_DIR}/{MODEL_NAME}",
        )
    except Exception as e:
        print("⚠️ No se pudo subir a HDFS, pero el modelo local está OK:", e)

    spark.stop()
    print("=== ENTRENAMIENTO TERMINADO ===")


if __name__ == "__main__":
    main()
