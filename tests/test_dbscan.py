import unittest
import dbscan
from pyspark.sql import types as T, SparkSession
from scipy.spatial import distance
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np


def dist(x, y):
    return distance.euclidean(x, y)


def get_spark():
    return SparkSession \
        .builder \
        .appName("DBSCAN") \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.1-spark2.4-s_2.12") \
        .config('spark.driver.host', '127.0.0.1') \
        .getOrCreate()


class DBScanTestCase(unittest.TestCase):
    def setUp(self):
        self.spark = get_spark()

    def tearDown(self):
        self.spark.stop()

    def test_generated_blobs(self):
        centers = [[1, 1], [-1, -1], [1, -1]]
        # with following data operations with sklearn dbscan 750*749/2 = 280875 for spark  149716(.2) 217624(0.3)
        X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=5)
        db = DBSCAN(eps=0.2, min_samples=10).fit(X)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        labels_spark = np.zeros_like(db.labels_)
        labels_spark[:] = -1

        data = [(i, [float(item) for item in X[i]]) for i in range(X.shape[0])]
        schema = T.StructType([T.StructField("id", T.IntegerType(), False),
                               T.StructField("value", T.ArrayType(T.FloatType()), False)])
        df = self.spark.createDataFrame(data, schema=schema)
        df_clusters = dbscan.process(self.spark, df, .2, 10, dist, 2, "checkpoint")
        out = df_clusters.distinct().collect()
        for item in out:
            labels_spark[item.point] = item.component
        n_clusters_spark_ = len(set(labels_spark)) - (1 if -1 in labels else 0)
        n_noise_spark_ = list(labels_spark).count(-1)
        self.assertEqual(n_clusters_, n_clusters_spark_)
        self.assertEqual(n_noise_, n_noise_spark_)

    def test_generated_rings(self):
        num_samples = 500
        # make a simple unit circle
        theta = np.linspace(0, 2 * np.pi, num_samples)
        X1 = np.random.rand(num_samples, 2) + np.transpose([0.5 * np.cos(theta), 0.5 * np.sin(theta) ])
        X2 = np.random.rand(num_samples, 2) + np.transpose([5 * np.cos(theta), 5 * np.sin(theta) ])
        X = np.concatenate([X1, X2])
        db = DBSCAN(eps=0.3, min_samples=5).fit(X)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        labels_spark = np.zeros_like(db.labels_)
        labels_spark[:] = -1

        data = [(i, [float(item) for item in X[i]]) for i in range(X.shape[0])]
        schema = T.StructType([T.StructField("id", T.IntegerType(), False),
                               T.StructField("value", T.ArrayType(T.FloatType()), False)])
        df = self.spark.createDataFrame(data, schema=schema)
        df_clusters = dbscan.process(self.spark, df, .3, 5, dist, 2, "checkpoint")
        out = df_clusters.distinct().collect()
        for item in out:
            labels_spark[item.point] = item.component
        n_clusters_spark_ = len(set(labels_spark)) - (1 if -1 in labels else 0)
        n_noise_spark_ = list(labels_spark).count(-1)
        self.assertEqual(n_clusters_, n_clusters_spark_)
        self.assertEqual(n_noise_, n_noise_spark_)


if __name__ == '__main__':
    unittest.main()
