# streaming.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when

# Rutas en HDFS
HDFS_INPUT = "hdfs://namenode:9000/neumonia-data"
HDFS_OUTPUT = "hdfs://namenode:9000/Preprocesamiento"

IMG_SIZE = 128  # solo informativo aquí


def main():
    spark = (
        SparkSession.builder
        .appName("image-preprocessing-cnn")
        .getOrCreate()
    )

    print("=== INICIANDO PREPROCESAMIENTO DE IMÁGENES ===")
    print("INPUT HDFS: ", HDFS_INPUT)
    print("OUTPUT HDFS:", HDFS_OUTPUT)
    print("IMG_SIZE:   ", f"{IMG_SIZE}x{IMG_SIZE}")

    # 1) Leer imágenes desde HDFS como binario
    df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpeg")
        .option("recursiveFileLookup", "true")
        .load(HDFS_INPUT)
    )
    # columnas: path, modificationTime, length, content (binary)

    # 2) Extraer metadata y label
    df_meta = (
        df
        # nombre de archivo
        .withColumn("filename", regexp_extract(col("path"), r"([^/]+)$", 1))
        # etiqueta textual desde el path
        .withColumn("label_str", regexp_extract(col("path"), r"/(NORMAL|PNEUMONIA)/", 1))
        # etiqueta numérica
        .withColumn(
            "label",
            when(col("label_str") == "PNEUMONIA", 1).otherwise(0)
        )
        # nos quedamos con lo útil
        .select(
            col("path").alias("hdfs_path"),
            "filename",
            "label_str",
            "label",
            "length",
            "content"  # bytes de la imagen (importante para entrenar luego)
        )
    )

    total = df_meta.count()
    print("TOTAL IMÁGENES LEÍDAS:", total)

    # 3) Guardar en HDFS como Parquet
    (
        df_meta
        .repartition(6)
        .write
        .mode("overwrite")
        .parquet(HDFS_OUTPUT)
    )

    print("=== PREPROCESAMIENTO COMPLETADO Y GUARDADO EN HDFS ===")
    spark.stop()


if __name__ == "__main__":
    main()
