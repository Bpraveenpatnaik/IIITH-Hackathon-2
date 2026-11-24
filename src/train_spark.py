"""Train the same kind of model with PySpark MLlib, to show the pipeline at scale.

Run: python src/train_spark.py   (needs Java 8+ and pyspark)
"""
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from data import DATA_PATH, load_data

TARGET = "SalePrice"
SEED = 42


def main():
    load_data()  # makes sure data/train.csv exists
    spark = SparkSession.builder.master("local[*]").appName("house-prices").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True, nullValue="NA")

    # numeric features only; the target must NOT be one of the inputs
    numeric = [c for c, t in df.dtypes if t in ("int", "double") and c not in ("Id", TARGET)]
    df = df.select(*[F.col(c).cast("double") for c in numeric],
                   F.log1p(F.col(TARGET)).alias("label"))

    pipeline = Pipeline(stages=[
        Imputer(inputCols=numeric, outputCols=numeric, strategy="median"),
        VectorAssembler(inputCols=numeric, outputCol="raw_features"),
        StandardScaler(inputCol="raw_features", outputCol="features"),
        LinearRegression(maxIter=100, regParam=0.01, elasticNetParam=0.5),
    ])

    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    model = pipeline.fit(train)
    pred = model.transform(test)

    rmse = RegressionEvaluator(metricName="rmse").evaluate(pred)
    r2 = RegressionEvaluator(metricName="r2").evaluate(pred)
    dollars = pred.select(F.sqrt(F.avg((F.expm1("prediction") - F.expm1("label")) ** 2))).first()[0]
    print(f"Spark elastic-net regression on {len(numeric)} numeric features")
    print(f"  test RMSE (log price): {rmse:.4f}")
    print(f"  test RMSE ($):         {dollars:,.0f}")
    print(f"  test R² (log price):   {r2:.3f}")
    spark.stop()


if __name__ == "__main__":
    main()
