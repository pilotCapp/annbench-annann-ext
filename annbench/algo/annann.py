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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

    @tf.function
    def encode(encoder, vecs):
        return encoder(vecs, training=False)

    @tf.function(jit_compile=True)
    def loss_function(y_true, y_pred):
        # Reconstruction loss (mean squared error)
        reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)

        # Get the latent space from the autoencoder (encoder output)
        z = encode(encoder, y_true)

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
        self.index = {}
        self.nn_layers = 4
        self.isdynamic = False
        self.test_size = 0.1

        self.n_sub_clusters = 50000

        self.ef_search = 100

        self.encoding_dim = 0.25
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.path = None

        # Parameters
        max_iter = 100
        batch_size = 10000  # Adjust this based on memory and stability
        max_no_improvement = 10  # Stop if no improvement

        self.cluster_algorithm = MiniBatchKMeans(
            n_clusters=self.n_sub_clusters,
            max_iter=max_iter,
            batch_size=batch_size,
            max_no_improvement=max_no_improvement,
            init="k-means++",
            random_state=42,
            verbose=1,
        )
        # self.cluster_algorithm = KMeans(n_clusters=self.n_sub_clusters)

        self.hnsw = HnswANN()  # use HNSW for cluster lookup
        self.hnsw.set_index_param({"ef_construction": self.ef_search * 2, "M": 16})

        self.normalizer = MinMaxScaler()

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

    def train(self, vecs, path):
        self.path = path
        model_path = os.path.join(path, "model.keras")

        self.normalizer.fit(vecs)
        normalized_vecs = self.normalizer.transform(vecs)

        if self.isdynamic:
            normalized_vecs = normalized_vecs[: int(len(normalized_vecs) / 2)]

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

            self.encode(np.zeros((1, self.input_dim)))
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            vecs_train, vecs_test = train_test_split(
                normalized_vecs, test_size=self.test_size, random_state=42
            )

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
                monitor="val_loss", patience=3, restore_best_weights=True
            )
            print("encoder output shape:", self.encoder.output_shape)

            # Add early stopping as a callback
            self.autoencoder.fit(
                x=vecs_train,
                y=vecs_train,
                validation_data=(vecs_test, vecs_test),
                epochs=100,
                batch_size=64,
                callbacks=[early_stopping_initial],
            )

            print("predicting embedded vectors")
            predictions = self.encode(
                normalized_vecs
            ).numpy()  # predicint sub_cluster positions

            print("creating cluster centroids")
            sub_cluster_centers = self.cluster_algorithm.fit(
                predictions
            ).cluster_centers_

            # sub_cluster_centers_embedded = self.encoder.predict(sub_cluster_centers)

            early_stopping_final = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True
            )

            self.autoencoder.compile(
                optimizer="adam",
                loss=dec_loss(self.encoder, sub_cluster_centers),
            )
            self.autoencoder.fit(
                x=vecs_train,
                y=vecs_train,
                validation_data=(vecs_test, vecs_test),
                epochs=20,
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

        if self.isdynamic:
            self.cluster_algorithm.fit(encoded_vecs[: int(len(encoded_vecs) / 2)])
        else:
            self.cluster_algorithm.fit(encoded_vecs)

        cluster_assignments = self.cluster_algorithm.predict(encoded_vecs)
        centroids = self.cluster_algorithm.cluster_centers_

        # Compute the number of entries in each cluster
        counts = np.bincount(cluster_assignments)
        clusters = np.arange(len(counts))

        counts_zero = np.sum(counts == 0)
        counts_five = np.sum(counts <= 5)
        print(
            f"amount clusters zero:{counts_zero}, amount clusters five or less:{counts_five}"
        )

        # Sort the clusters by counts in descending order
        sorted_indices = np.argsort(counts)[::-1]  # Use [::-1] for descending order
        sorted_clusters = clusters[sorted_indices]
        sorted_counts = counts[sorted_indices]

        # Plot the distribution using Matplotlib
        plt.figure(figsize=(12, 6))

        # Use sequential x positions
        x_positions = range(len(sorted_counts))

        plt.bar(x_positions, sorted_counts, color="skyblue")
        plt.xlabel("Cluster Index (sorted)", fontsize=14)
        plt.ylabel("Number of Entries", fontsize=14)
        plt.title("Distribution of Number of Entries per Cluster", fontsize=16)

        # Set the x-ticks to be at the sequential positions with labels as sorted cluster indices
        plt.xticks(
            x_positions, labels=sorted_clusters, rotation=45
        )  # Rotate labels if necessary

        # **Add a horizontal line at y=20**
        plt.axhline(
            y=20, color="red", linestyle="--", linewidth=2, label="Threshold = 20"
        )

        # Optional: Add a legend to describe the horizontal line
        plt.legend()

        plt.tight_layout()

        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{self.path}/cluster_spread_{self.isdynamic}.png")

        self.index = {
            i: [[], []] for i in range(len(centroids))
        }  # TODO change to ssd read/write

        for k, (vec, cluster_index) in enumerate(zip(vecs, cluster_assignments)):
            if cluster_index not in self.index:
                self.index[cluster_index] = [[], []]
            self.index[cluster_index][0].append(k)
            self.index[cluster_index][1].append(vec)

        self.centroids = np.array(centroids)
        self.centroids_decoded = self.normalizer.inverse_transform(
            self.decode(self.centroids).numpy()
        )

        print("cluster centroids length", len(self.centroids))
        self.hnsw.add(self.centroids)

        print("index built with length", len(self.index))
        print(
            "first cluster in index has length",
            len(self.index[0]),
        )

    def query(self, vecs, topk, param):
        time_log = time.time()
        # vecs = tf.convert_to_tensor(vecs)
        normalized_vecs = self.normalizer.transform(vecs)
        encoded_queries = self.encode(normalized_vecs).numpy()
        self.query_runs += 1
        # Encode the query vectors

        self.query_time_1 += time.time() - time_log  # Time for encoding queries
        time_log = time.time()

        results = []

        closest_sub_clusters_all = self.hnsw.query(
            encoded_queries,
            param["cluster_search"] * 10,  # number of closest to find
            {"ef": self.ef_search},
        )

        self.query_time_2 += time.time() - time_log  # Time for hnsw cluster search
        time_log = time.time()
        time_log_3 = time.time()

        for original_query, closest_sub_clusters in zip(vecs, closest_sub_clusters_all):

            cluster_entries = [[], []]

            closest_sub_clusters_decoded = [
                self.centroids_decoded[i] for i in closest_sub_clusters
            ]

            self.query_time_3_1 += (
                time.time() - time_log_3
            )  # Time for retrieving and decoding closest clusters
            time_log_3 = time.time()

            closest_sub_distances = np.linalg.norm(
                closest_sub_clusters_decoded - original_query, axis=1
            )

            closest_sub_clusters_sorted = np.argsort(closest_sub_distances)[
                : param["cluster_search"]
            ]

            self.query_time_3_2 += (
                time.time() - time_log_3
            )  # Time for  finding closest subclusters
            time_log_3 = time.time()

            # Retrieve cluster entries corresponding to the closest centroid
            for i in closest_sub_clusters_sorted:
                cluster_entries[0].extend(self.index[closest_sub_clusters[i]][0])
                cluster_entries[1].extend(self.index[closest_sub_clusters[i]][1])

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

    @tf.function(reduce_retracing=True)
    def encode(self, vecs):
        return self.encoder(vecs, training=False)

    @tf.function(reduce_retracing=True)
    def decode(self, vecs):
        return self.decoder(vecs, training=False)

    def write(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)

        with open(os.path.join(path, "centroids.pkl"), "wb") as f:
            pickle.dump(self.centroids, f)
        with open(os.path.join(path, "centroids_decoded.pkl"), "wb") as f:
            pickle.dump(self.centroids_decoded, f)

        self.hnsw.write(os.path.join(path, "hnsw"))

    def read(self, path, D):
        print("Reading index")
        with open(os.path.join(path, "index.pkl"), "rb") as f:
            self.index = pickle.load(f)
        with open(os.path.join(path, "centroids.pkl"), "rb") as f:
            self.centroids = pickle.load(f)
        with open(os.path.join(path, "centroids_decoded.pkl"), "rb") as f:
            self.centroids_decoded = pickle.load(f)
        self.autoencoder = tf.keras.models.load_model(os.path.join(path, "model.keras"))
        self.encoder = tf.keras.models.Model(
            inputs=self.autoencoder.input,
            outputs=self.autoencoder.layers[self.nn_layers].output,
        )
        self.encode(np.zeros((1, self.input_dim)))
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
        return f"test_size_{self.test_size}_encodingdim_{self.encoding_dim}_clusters_{self.n_sub_clusters}_layers_{self.nn_layers}_dynamic_{self.isdynamic}"


# query time 1%: 0.07688888739751326
# query time 2%: 0.11602366853797023
# query time 3%: 0.8070874440645165

# query time 3_1%: 0.13722583539672958

# query time 3_2%: 0.28620613149015445

# query time 3_3%: 0.5731103718875663
