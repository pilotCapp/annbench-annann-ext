import os
import pickle

from sklearn.neighbors import NearestNeighbors
from .base import BaseANN
import annoy
import tensorflow as tf
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
import numpy as np
from pathlib import Path
from hydra.utils import to_absolute_path
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from .hnsw import HnswANN


class ANNANN_BASE(BaseANN):
    def __init__(self):
        self.name = "annann_base"
        self.index = {}
        self.isdynamic=False

        self.n_sub_clusters = 50000

        self.ef_search = 100

        self.path = None

        self.cluster_algorithm = MiniBatchKMeans(n_clusters=self.n_sub_clusters)

        self.hnsw = HnswANN()  # use HNSW for cluster lookup

        self.hnsw.set_index_param({"ef_construction": self.ef_search * 2, "M": 16})

        self.train_size = 1

        self.query_runs = 0
        self.query_time_1 = 0
        self.query_time_2 = 0
        self.query_time_3 = 0

        self.query_time_3_1 = 0
        self.query_time_3_2 = 0
        self.query_time_3_3 = 0

        self.query_time_3_3_1 = 0
        self.query_time_3_3_2 = 0
        self.query_time_3_3_3 = 0

    def set_index_param(self, param):
        self.isdynamic = param["Dynamic"]

    def train(self, vecs):
        pass

    def has_train(self):
        return False

    def add(self, vecs):

        if self.isdynamic:
            self.cluster_algorithm.fit(vecs[: int(len(vecs) / 2)])
        else:
            self.cluster_algorithm.fit(vecs)

        cluster_assignments = self.cluster_algorithm.predict(vecs)
        centroids = self.cluster_algorithm.cluster_centers_

        self.index = {
            i: [[], []] for i in range(len(centroids))
        }  # TODO change to ssd read/write

        for k, (vec, cluster_index) in enumerate(zip(vecs, cluster_assignments)):
            self.index[cluster_index][0].append(k)
            self.index[cluster_index][1].append(vec)

        self.centroids = np.array(centroids)
        print("cluster centroids length", len(self.centroids))
        self.hnsw.add(self.centroids)

        print("index built with length", len(self.index))
        print(
            "first cluster in index has length",
            len(self.index[0]),
        )

    def query(self, vecs, topk, param):
        time_log = time.time()

        self.query_time_1 += time.time() - time_log  # Time for encoding queries
        time_log = time.time()

        results = []

        closest_sub_clusters_all = self.hnsw.query(
            vecs,
            param["cluster_search"],  # number of closest to find
            {"ef": self.ef_search},
        )

        self.query_time_2 += time.time() - time_log  # Time for hnsw cluster search
        time_log = time.time()
        time_log_3 = time.time()

        for original_query, closest_sub_clusters in zip(vecs, closest_sub_clusters_all):

            cluster_entries = [[], []]

            self.query_time_3_1 += (
                time.time() - time_log_3
            )  # Time for retrieving and decoding closest clusters
            time_log_3 = time.time()

            self.query_time_3_2 += (
                time.time() - time_log_3
            )  # Time for  finding closest subclusters
            time_log_3 = time.time()

            # Retrieve cluster entries corresponding to the closest centroid
            for i in closest_sub_clusters:
                cluster_entries[0].extend(self.index[i][0])
                cluster_entries[1].extend(self.index[i][1])

            indices_list = cluster_entries[0]
            vectors_list = cluster_entries[1]

            # Compute distances from the original query to all vectors in the closest clusters
            # Using efficient NumPy broadcasting
            distances = np.linalg.norm(vectors_list - original_query, axis=1)

            # Get indices of the top-k nearest neighbors
            topk_indices = np.argsort(distances)[:topk]

            # Append the indices of the top-k nearest neighbors to the results
            results.append([indices_list[i] for i in topk_indices])

            self.query_time_3_3 += (
                time.time() - time_log_3
            )  # Time for computing distances and sorting
            time_log_3 = time.time()

        self.query_time_3 += time.time() - time_log
        return results

    def write(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(path, "centroids.pkl"), "wb") as f:
            pickle.dump(self.centroids, f)
        self.hnsw.write(os.path.join(path, "hnsw"))

    def read(self, path, D):
        print("Reading index")
        with open(os.path.join(path, "index.pkl"), "rb") as f:
            self.index = pickle.load(f)
        with open(os.path.join(path, "centroids.pkl"), "rb") as f:
            self.centroids = pickle.load(f)

        self.hnsw.read(
            os.path.join(
                path,
                "hnsw",
            ),
            128, #only for sift dataset
        )

    def stringify_index_param(self, param):
        """Convert index parameters to a string representation."""
        return f"train_size_{self.train_size}_clusters_{self.n_sub_clusters}_dynamic_{self.isdynamic}"

# query time 1%: 1.4446113043461131e-05
# query time 2%: 0.1920812646390015
# query time 3%: 0.8079042892479551

# query time 3_1%: 0.004796367236940948

# query time 3_2%: 0.0009527712761709851

# query time 3_3%: 0.9911641588943708
