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


@tf.function
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


@tf.function
def target_distribution(q):
    weight = q**2 / tf.reduce_sum(q, axis=0)
    p = tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, axis=1))
    return p


def dec_loss(encoder, cluster_centers, alpha=1.0):
    mse = tf.keras.losses.MeanSquaredError()

    @tf.function
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
        self.input_dim = 128  # only for sift
        self.name = "annann"
        self.index = None
        self.nn_layers = 2
        self.n_main_clusters = 100
        self.n_sub_clusters = 10

        self.encoding_dim = None
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.path = None
        self.cluster_algorithm = self.cluster_algorithm_small = KMeans(
            n_clusters=self.n_clusters
        )
        self.cluster_algorithm_small = KMeans(n_clusters=self.n_clusters_small)

        self.hnsw = HnswANN()  # use HNSW for cluster lookup
        self.hnsw.set_index_param({"ef_construction": 300, "M": 15})
        self.hnsw_small = {}

        self.normalizer = MinMaxScaler()

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
        self.encoding_dim = param["input_param"]

    def train(self, vecs, path):
        self.path = path
        model_path = os.path.join(path, "model.keras")

        self.normalizer.fit(vecs)
        normalized_vecs = self.normalizer.transform(vecs)

        if os.path.exists(model_path):
            print("model already exists")
            self.autoencoder = tf.keras.models.load_model(model_path)
            self.encoder = tf.keras.models.Model(
                inputs=self.autoencoder.input,
                outputs=self.autoencoder.layers[self.nn_layers].output,
            )
            self.decoder = tf.keras.models.Model(
                inputs=self.autoencoder.layers[self.nn_layers + 1].input,
                outputs=self.autoencoder.output,
            )
            self.normalizer = joblib.load(os.path.join(path, "normalizer.pkl"))
            print(
                "encoder retrieved with input/output shape",
                self.encoder.input_shape,
                self.encoder.output_shape,
            )
            print(
                "decoder retrieved with input/output shape",
                self.decoder.input_shape,
                self.decoder.output_shape,
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
            steps = (input_dim - int(input_dim * self.encoding_dim)) / (
                max(self.nn_layers - 1, 1)
            )
            encoder_layer_sizes = [
                int(input_dim - i * steps) for i in range(0, self.nn_layers)
            ]

            print("encoder layer sizes", encoder_layer_sizes)

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
            self.decoder = tf.keras.models.Model(
                inputs=self.autoencoder.layers[self.nn_layers + 1].input,
                outputs=self.autoencoder.output,
            )
            self.autoencoder.compile(optimizer="adam", loss="mean_squared_error")
            # Define early stopping
            early_stopping_initial = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=3, restore_best_weights=True
            )
            print("encoder output shape:", self.encoder.output_shape)

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
                epochs=15,
                batch_size=64,
                callbacks=[early_stopping_final],
            )
            self.autoencoder.compile(optimizer="adam", loss="mean_squared_error")
            self.autoencoder.save(model_path)
            joblib.dump(self.normalizer, os.path.join(path, "normalizer.pkl"))
            print("model saved")

    def has_train(self):
        return not os.path.exists(os.path.join(self.path, "model.keras"))

    def add(self, vecs):
        normalized_vecs = self.normalizer.transform(vecs)
        encoded_vecs = self.encode(normalized_vecs).numpy()

        self.cluster_algorithm.fit(encoded_vecs)
        cluster_assignments = self.cluster_algorithm.predict(encoded_vecs)
        self.centroids = self.cluster_algorithm.cluster_centers_

        self.hnsw.add(self.centroids)

        self.index = {}  # TODO change to ssd read/write

        for k, (vector, main_cluster_index) in enumerate(
            zip(vecs, cluster_assignments)
        ):
            if main_cluster_index not in self.index:
                self.index[main_cluster_index] = [[], []]
            self.index[main_cluster_index][0].append(k)
            self.index[main_cluster_index][1].append(vector)

        for cluster_index in self.index.keys():
            if len(self.index[cluster_index][1]) < 10:
                raise ("too few instances in main_cluster, fix this future kris")

            self.cluster_algorithm_small.fit(self.index[cluster_index][1])
            cluster_assignments_small = self.cluster_algorithm_small.predict(
                self.index[cluster_index][1]
            )
            cluster_small_centroids = self.cluster_algorithm_small.cluster_centers_

            if cluster_index not in self.hnsw_small:
                self.hnsw_small[cluster_index] = HnswANN()

            self.hnsw_small[cluster_index].set_index_param(
                {"ef_construction": 10, "M": 5}
            )
            self.hnsw_small[cluster_index].add(cluster_small_centroids)

            temp_index = {}

            for vector_index, vector, small_cluster_assignment in zip(
                self.index[cluster_index][0],
                self.index[cluster_index][1],
                cluster_assignments_small,
            ):
                if small_cluster_assignment not in temp_index:
                    temp_index[small_cluster_assignment] = [[], []]

                temp_index[small_cluster_assignment][0].append(vector_index)
                temp_index[small_cluster_assignment][1].append(vector)
            self.index[cluster_index] = temp_index

        # for a, index in enumerate(cluster_assignments):
        #     if a not in self.index:
        #         self.index[a] = []
        #     self.index[a].append(vecs[index])
        print("index built with length", len(self.index))
        print("with sub cluster length,", len(self.index[0]))

    def query(self, vecs, topk, param):
        time_log = time.time()

        normalized_vecs = self.normalizer.transform(vecs)
        self.query_runs += 1
        # Encode the query vectors

        encoded_queries = self.encode(normalized_vecs).numpy()

        self.query_time_1 += time.time() - time_log  # Time for encoding queries
        time_log = time.time()
        # Ensure centroids are NumPy arrays
        centroids = np.array(self.centroids)

        closest_main_centroid_indices = self.hnsw.query(
            encoded_queries,
            param["branch_num"],
            {"ef": 100},  # finding the closest main clusters with branch param
        )

        results = []
        self.query_time_2 += time.time() - time_log  # Time for finding closest clusters
        time_log = time.time()
        for original_query, closest_centroid_idx in zip(
            vecs, closest_main_centroid_indices
        ):
            time_log_3 = time.time()
            # no need for decoding anymore...
            self.query_time_3_1 += (
                time.time() - time_log_3
            )  # Time for decoding centroids, not feasable
            time_log_3 = time.time()

            cluster_entries = [[], []]

            for main_cluster in closest_centroid_idx:
                closest_sub_clusters = self.hnsw_small[main_cluster].query(
                    original_query,
                    param["branch_num"],
                    {
                        "ef": 100
                    },  # should be param["branch_num"], 10 for testing accuracy of subdomain
                )
                self.query_time_3_2 += (
                    time.time() - time_log_3
                )  # Time for retrieving and sorting distances
                time_log_3 = time.time()

                # Retrieve cluster entries corresponding to the closest centroid
                for i in closest_sub_clusters[0]:
                    cluster_entries[0].extend(self.index[main_cluster][i][0])
                    cluster_entries[1].extend(self.index[main_cluster][i][1])

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

        self.query_time_3 += time.time() - time_log
        return results

    @tf.function
    def encode(self, vecs):
        return self.encoder(vecs, training=False)

    @tf.function
    def decode(self, vecs):
        return self.decoder(vecs, training=False)

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
        self.autoencoder = tf.keras.models.load_model(os.path.join(path, "model.keras"))
        self.encoder = tf.keras.models.Model(
            inputs=self.autoencoder.input,
            outputs=self.autoencoder.layers[self.nn_layers].output,
        )
        self.decoder = tf.keras.models.Model(
            inputs=self.autoencoder.layers[self.nn_layers + 1].input,
            outputs=self.autoencoder.output,
        )
        print(
            "encoder retrieved with input/output shape",
            self.encoder.input_shape,
            self.encoder.output_shape,
        )
        print(
            "decoder retrieved with input/output shape",
            self.decoder.input_shape,
            self.decoder.output_shape,
        )
        normalizer = joblib.load(os.path.join(path, "normalizer.pkl"))
        print("reading encoding dim", self.encoding_dim)

        self.hnsw.read(
            os.path.join(
                path,
                "hnsw",
            ),
            int(self.input_dim * self.encoding_dim),
        )

    def stringify_index_param(self, param):
        """Convert index parameters to a string representation."""
        print(self.encoding_dim)
        return f"train_size_{self.train_size}_encodingdim_{self.encoding_dim}_clusters_{self.n_clusters}_layers_{self.nn_layers}"


# query time 1%: 0.014361327789875485
# query time 2%: 0.0010875552723763689
# query time 3%: 0.9845511169377481

# query time 3_1%: 0.015918930155873506
# query time 3_2%: 0.7160475324813171
# query time 3_3%: 0.26657019335323495
