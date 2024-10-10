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


def soft_cluster_assignments(z, cluster_centers, alpha=1.0):
    q = 1.0 / (
        1.0
        + (
            tf.reduce_sum(
                tf.square(tf.expand_dims(z, axis=1) - cluster_centers), axis=2
            )
            / alpha
        )
    )
    q = q ** ((alpha + 1.0) / 2.0)
    q = q / tf.reduce_sum(q, axis=1, keepdims=True)
    return q


def target_distribution(q):
    weight = q**2 / tf.reduce_sum(q, axis=0)
    p = tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, axis=1))
    return p


def dec_loss(encoder, cluster_centers, alpha=1.0):
    mse = tf.keras.losses.MeanSquaredError()

    def loss_function(y_true, y_pred):
        # Reconstruction loss (mean squared error)
        reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)

        # Get the latent space from the autoencoder (encoder output)
        z = encoder(y_true)

        # Compute soft assignments q
        q = soft_cluster_assignments(z, cluster_centers, alpha=alpha)

        # Compute the target distribution p
        p = target_distribution(q)

        # Clustering loss (KL divergence between p and q)
        clustering_loss = tf.reduce_mean(tf.keras.losses.KLDivergence()(p, q))

        # Total loss = reconstruction loss + clustering loss
        total_loss = reconstruction_loss + clustering_loss

        return total_loss

    return loss_function


class ANNANN(BaseANN):
    def __init__(self):
        self.index = None
        self.nn_layers = 5
        self.n_clusters = 10000
        self.encoding_dim = None
        self.autoencoder = None
        self.encoder = None
        self.cluster_algorithm = MiniBatchKMeans(
            n_clusters=self.n_clusters, max_iter=300
        )
        self.train_size = 0.2

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
        self.encoding_dim = param["input_param"]

    def train(self, vecs, path):
        model_path = os.path.join(path, "model.keras")

        self.normalizer = MinMaxScaler()

        self.normalizer.fit(vecs)
        normalized_vecs = self.normalizer.transform(vecs)

        if os.path.exists(model_path):
            print("model already exists")
            self.autoencoder = tf.keras.models.load_model(model_path)
            self.encoder = tf.keras.models.Model(
                inputs=self.autoencoder.input,
                outputs=self.autoencoder.layers[-2].output,
            )
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            normalized_vecs = normalized_vecs[
                : int(len(normalized_vecs) * self.train_size)
            ]  # train_size is the fraction of the data the model should be trained on, to test how the model performs on new data
            # take only the first n_size elements
            input_dim = normalized_vecs.shape[1]
            input_vec = tf.keras.layers.Input(shape=(input_dim,))
            print("input dim is", input_dim)

            # Calculate the sizes of the encoder layers
            # This will create a list of sizes decreasing from input_dim to encoding_dim
            encoder_layer_sizes = [
                int(
                    input_dim
                    - i / (self.nn_layers) * int(input_dim * self.encoding_dim)
                )
                for i in range(1, self.nn_layers + 1)
            ]
            print(encoder_layer_sizes)

            # Build the encoder
            encoded = input_vec
            for units in encoder_layer_sizes:
                encoded = tf.keras.layers.Dense(units, activation="relu")(encoded)

            # Calculate the sizes of the decoder layers
            # This will create a list of sizes increasing from encoding_dim back to input_dim
            decoder_layer_sizes = encoder_layer_sizes[
                :-1
            ]  # Reverse the encoder sizes except the last one
            decoder_layer_sizes.append(input_dim)  # Add the output dimension

            # Build the decoder
            decoded = encoded
            for units in decoder_layer_sizes:
                decoded = tf.keras.layers.Dense(units, activation="relu")(decoded)

            # Use 'sigmoid' activation in the final layer if your input data is normalized between 0 and 1
            decoded = tf.keras.layers.Dense(input_dim, activation="sigmoid")(decoded)

            # Define the autoencoder model
            self.autoencoder = tf.keras.models.Model(input_vec, decoded)
            # Define the encoder model
            self.encoder = tf.keras.models.Model(input_vec, encoded)
            self.autoencoder.compile(optimizer="adam", loss="mean_squared_error")
            # Define early stopping
            early_stopping_initial = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=5, restore_best_weights=True
            )

            # Add early stopping as a callback
            self.autoencoder.fit(
                normalized_vecs,
                normalized_vecs,
                epochs=100,
                batch_size=64,
                callbacks=[early_stopping_initial],
            )
            predictions = self.encoder.predict(normalized_vecs)

            initial_cluster_centers = self.cluster_algorithm.fit(
                predictions
            ).cluster_centers_

            early_stopping_final = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=3, restore_best_weights=True
            )

            self.autoencoder.compile(
                optimizer="adam",
                loss=dec_loss(self.encoder, initial_cluster_centers),
            )
            self.autoencoder.fit(
                normalized_vecs,
                normalized_vecs,
                epochs=25,
                batch_size=64,
                callbacks=[early_stopping_final],
            )
            self.autoencoder.compile(optimizer="adam", loss="mean_squared_error")
            self.autoencoder.save(model_path)
            print("model saved")

    def has_train(self):
        return True

    def add(self, vecs):
        normalized_vecs = self.normalizer.transform(vecs)
        encoded_vecs = self.encode(normalized_vecs).numpy()

        self.cluster_algorithm.fit(encoded_vecs)
        cluster_assignments = self.cluster_algorithm.predict(encoded_vecs)
        self.centroids = self.cluster_algorithm.cluster_centers_

        self.index = {}  # TODO change to ssd read/write

        for k, (vector, cluster) in enumerate(zip(vecs, cluster_assignments)):
            if cluster not in self.index:
                self.index[cluster] = [[], []]
            self.index[cluster][0].append(k)
            self.index[cluster][1].append(vector)

        # for a, index in enumerate(cluster_assignments):
        #     if a not in self.index:
        #         self.index[a] = []
        #     self.index[a].append(vecs[index])
        print("index built with length", len(self.index))

    def query(self, vecs, topk, param):
        time_log = time.time()

        normalized_vecs = self.normalizer.transform(vecs)
        self.query_runs += 1
        # Encode the query vectors

        encoded_queries = self.encode(normalized_vecs).numpy()

        self.query_time_1 += time.time() - time_log
        time_log = time.time()
        # Ensure centroids are NumPy arrays
        centroids = np.array(self.centroids)

        # Compute squared norms of encoded_queries and centroids
        queries_norm_squared = np.sum(np.square(encoded_queries), axis=1).reshape(
            -1, 1
        )  # Shape: (num_queries, 1)
        centroids_norm_squared = np.sum(np.square(centroids), axis=1).reshape(
            1, -1
        )  # Shape: (1, num_centroids)

        # Compute dot product between queries and centroids
        dot_product = np.dot(
            encoded_queries, centroids.T
        )  # Shape: (num_queries, num_centroids)

        # Compute distances using the efficient formula
        distances_to_centroids = np.sqrt(
            queries_norm_squared - 2 * dot_product + centroids_norm_squared
        )

        # Find the closest centroid for each query
        closest_centroid_indices = np.argsort(distances_to_centroids, axis=1)

        results = []
        self.query_time_2 += time.time() - time_log
        time_log = time.time()
        for original_query, closest_centroid_idx in zip(vecs, closest_centroid_indices):

            time_log_3 = time.time()
            # Retrieve cluster entries corresponding to the closest centroid
            cluster_entries = [[], []]
            cluster_number = max(
                int(param.get("cluster_num", "default") * self.n_clusters), 1
            )
            for i in closest_centroid_idx[:cluster_number]:
                cluster_entries[0].extend(self.index[i][0])
                cluster_entries[1].extend(self.index[i][1])

            if len(cluster_entries) < topk:
                print("not enough entries in cluster")

            self.query_time_3_1 += time.time() - time_log_3
            time_log_3 = time.time()

            indices_list = cluster_entries[0]
            vectors_list = cluster_entries[1]

            self.query_time_3_2 += time.time() - time_log_3
            time_log_3 = time.time()
            time_log_3_3 = time.time()

            # Compute distances from the original query to all vectors in the closest clusters
            # Using efficient NumPy broadcasting
            distances = np.linalg.norm(vectors_list - original_query, axis=1)

            self.query_time_3_3_1 += time.time() - time_log_3_3
            time_log_3_3 = time.time()

            # Get indices of the top-k nearest neighbors
            topk_indices = np.argsort(distances)[:topk]

            self.query_time_3_3_2 += time.time() - time_log_3_3
            time_log_3_3 = time.time()

            # Append the indices of the top-k nearest neighbors to the results
            results.append([indices_list[i] for i in topk_indices])

            self.query_time_3_3_3 = time.time() - time_log_3_3
            self.query_time_3_3 += time.time() - time_log_3

        self.query_time_3 += time.time() - time_log
        return results

    @tf.function
    def encode(self, vecs):
        return self.encoder(vecs, training=False)

    def write(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)

    def read(self, path, D):
        print("storing path")
        self.path = path
        with open(os.path.join(path, "index.pkl"), "rb") as f:
            self.index = pickle.load(f)

    def stringify_index_param(self, param):
        """Convert index parameters to a string representation."""
        return f"train_size_{param.get('input_param', 'default')}_encodingdim_{self.encoding_dim}_clusters_{self.n_clusters}"


# query time 1%: 0.014361327789875485
# query time 2%: 0.0010875552723763689
# query time 3%: 0.9845511169377481

# query time 3_1%: 0.015918930155873506
# query time 3_2%: 0.7160475324813171
# query time 3_3%: 0.26657019335323495
