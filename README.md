# pyspark_dbscan
An Implementation of DBSCAN on PySpark

```python
import dbscan
from sklearn.datasets import make_blobs
from pyspark.sql import types as T, SparkSession
from scipy.spatial import distance

spark = SparkSession \
        .builder \
        .appName("DBSCAN") \
        .config("spark.jars.packages", "graphframes:graphframes:0.7.0-spark2.3-s_2.11") \
        .config('spark.driver.host', '127.0.0.1') \
        .getOrCreate()
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=5)
data = [(i, [float(item) for item in X[i]]) for i in range(X.shape[0])]
schema = T.StructType([T.StructField("id", T.IntegerType(), False),
                               T.StructField("value", T.ArrayType(T.FloatType()), False)])
#please repartition appropriately                            
df = spark.createDataFrame(data, schema=schema).repartition(10)
df_clusters = dbscan.process(spark, df, .2, 10, distance.euclidean, 2, "checkpoint")
```
