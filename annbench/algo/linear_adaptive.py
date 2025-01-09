import os
import pickle

from .base import BaseANN
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers


import numpy as np
from pathlib import Path
from hydra.utils import to_absolute_path
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split


from .faiss_cpu import LinearANN


def contrastive_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


# Custom training step to handle pairs of vectors
@tf.function
def train_step(encoder, optimizer, x1, x2, y):
    with tf.GradientTape() as tape:
        encoded_x1 = encoder(x1, training=True)
        encoded_x2 = encoder(x2, training=True)
        distances = K.sqrt(K.sum(K.square(encoded_x1 - encoded_x2), axis=1))
        loss = contrastive_loss(y, distances)
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    return loss


class Linear_Adaptive(BaseANN):
    def __init__(self):
        self.index = None
        self.index_norms = None
        self.ANN = LinearANN()
        self.ANN_2 = LinearANN()

        self.full_dim = 128
        self.partition_dim = 64

        self.query_runs = 0

        self.query_time_1 = 0
        self.query_time_1_1 = 0
        self.query_time_1_2 = 0
        self.query_time_1_3 = 0

        self.query_time_2 = 0
        self.query_time_2_1 = 0
        self.query_time_2_2 = 0
        self.query_time_2_3 = 0

        self.query_time_3 = 0
        self.query_time_3_1 = 0
        self.query_time_3_2 = 0
        self.query_time_3_3 = 0

        self.partitions = 2

        self.train_size = 0.9
        self.isadaptive = True
        self.nn_layers = 2

        self.path = None
        self.encoder = None
        self.normalizer = StandardScaler()

    def set_index_param(self, param):
        self.isadaptive = param["is_adaptive"]

    def train(self, vecs, path):
        self.path = path
        model_path = os.path.join(path, "model.keras")

        if os.path.exists(model_path):
            print("model already exists")
            self.encoder = tf.keras.models.load_model(model_path)

            self.normalizer = joblib.load(os.path.join(path, "normalizer.pkl"))
            print(
                "encoder retrieved with input/output shape",
                self.encoder.input_shape,
                self.encoder.output_shape,
            )
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            self.normalizer.fit(vecs)
            normalized_vecs = self.normalizer.transform(vecs)

            vecs_train, vecs_test = train_test_split(
                normalized_vecs, test_size=(1 - self.train_size), random_state=42
            )

            # take only the first n_size elements
            self.full_dim = normalized_vecs.shape[1]
            self.partition_dim = int(self.full_dim / self.partitions)
            input_vec = Input(shape=(self.full_dim,))
            print("input dim is", self.full_dim)

            # Calculate the sizes of the encoder layers
            # This will create a list of sizes decreasing from input_dim to encoding_dim

            # Build the encoder
            encoded = input_vec
            for layer in range(self.nn_layers):
                encoded = Dense(self.full_dim, activation="relu")(encoded)

            # Use 'sigmoid' activation in the final layer if your input data is normalized between 0 and 1
            encoded = Dense(self.full_dim, activation="relu")(encoded)

            # Define the encoder model
            self.encoder = tf.keras.Model(input_vec, encoded)

            print("making pairs")

            pairs = np.array(
                [
                    (vecs_train[i], vecs_train[j])
                    for i in range(len(vecs_train))
                    for j in np.random.choice(len(vecs_train), 50)
                ]
            )

            np.random.shuffle(pairs)

            x1 = pairs[:, 0]
            x2 = pairs[:, 1]
            y = np.linalg.norm(x1 - x2, axis=1)

            epochs = 100
            batch_size = 64
            optimizer = tf.keras.optimizers.Adam()

            # Training loop
            for epoch in range(epochs):
                epoch_loss = 0
                for i in range(0, len(x1), batch_size):
                    x1_batch = x1[i : i + batch_size]
                    x2_batch = x2[i : i + batch_size]
                    y_batch = y[i : i + batch_size]
                    loss = train_step(
                        self.encoder, optimizer, x1_batch, x2_batch, y_batch
                    )
                    epoch_loss += loss
                    # print(f"Batch {i//batch_size+1}/{len(x1)//batch_size}, Loss: {loss}")

                print(
                    f"Epoch {epoch+1}/{epochs}, loss: {epoch_loss/(len(x1)/batch_size):.4f}"
                )

            self.encoder.save(model_path)
            joblib.dump(self.normalizer, os.path.join(path, "normalizer.pkl"))
            print("model saved")

    def has_train(self):
        print("checking if requires training")
        print(self.isadaptive)
        return (
            not os.path.exists(
                os.path.join(self.stringify_index_param({}), "model.keras")
            )
        ) and self.isadaptive

    def add(self, vecs):

        if self.isadaptive:
            self.index = self.encoder.predict(self.normalizer.transform(vecs))
        else:
            self.index = vecs

        self.ANN.add(self.index[:, : self.partition_dim])
        # self.index_norms_1 = np.sum(self.index[:, : self.partition_dim] ** 2, axis=1)
        self.index_norms_2 = np.sum(self.index[:, self.partition_dim :] ** 2, axis=1)

    # ## Brute force search using numpy, slower but no index generation required for each query vector during pruning

    def query(self, vecs, topk=1, param=None):
        """
        Finds the top-k nearest neighbors for each query vector in vecs.

        Args:
            vecs (np.ndarray): 2D array where each row is a query vector.
            topk (int): Number of nearest neighbors to return.

        Returns:
            np.ndarray: 2D array of shape (len(vecs), topk) containing the indices of the top-k nearest neighbors for each query.
        """
        # Compute pairwise distances between query vectors and index
        # Using broadcasting for efficient computation

        time_log = time.time()
        time_log_1 = time.time()

        if self.isadaptive:
            normalized_vecs = self.normalizer.transform(vecs)
            encoded_vecs = self.encode(normalized_vecs).numpy()
            vecs = encoded_vecs

        self.query_time_1_1 += (
            time.time() - time_log_1
        )  # Time for initial distance calculation
        time_log_1 = time.time()

        self.query_time_1 += (
            time.time() - time_log
        )  # Time for initial distance calculation
        time_log = time.time()
        time_log_2 = time.time()

        # vecs_norms = np.sum(vecs[:, : self.partition_dim] ** 2, axis=1)

        # distances_1 = (
        #     self.index_norms_1[None, :]
        #     - 2 * (vecs[:, :self.partition_dim] @ self.index[:, :self.partition_dim].T)
        #     + vecs_norms[:, None]
        # )

        # self.query_time_1_1 += (
        #     time.time() - time_log_1
        # )  # Time for initial distance calculation
        # time_log_1 = time.time()

        # candidates = np.argpartition(distances_1, topk * 10, axis=1)[:, : topk * 10]

        candidates, distances_1 = self.ANN.query(
            vecs[:, : self.partition_dim], topk * 100, param, ret_distances=True
        )

        self.query_time_2_2 += (
            time.time() - time_log_2
        )  # Time for initial distance calculation
        time_log_2 = time.time()

        self.query_time_2 += (
            time.time() - time_log
        )  # Time for initial distance calculation
        time_log = time.time()

        # Get the indices of the candidates

        # Alternatively, if you want the actual candidate vectors from self.index
        # candidate_vectors = self.index[candidates[1]]

        candidate_vectors = self.index[
            candidates, self.partition_dim :
        ]  # Shape: (num_queries, num_candidates, split_dim)
        candidate_norms = self.index_norms_2[
            candidates
        ]  # Shape: (num_queries, num_candidates)

        # Extract the second half of the query vectors
        query_vectors = vecs[:, self.partition_dim :]  # Shape: (num_queries, split_dim)

        # Compute dot products: (num_queries, num_candidates)
        dot_products = np.einsum("ij,ikj->ik", query_vectors, candidate_vectors)

        # Compute query norms: (num_queries, 1), then broadcast to (num_queries, num_candidates)
        query_norms = np.sum(query_vectors**2, axis=1, keepdims=True)

        # Compute final distances: (num_queries, num_candidates)
        final_distances = (
            candidate_norms  # Precomputed norms for candidates
            - 2 * dot_products  # Dot products
            + query_norms  # Query norms
        )

        # Step 3: Combine distances from both halves
        combined_distances = distances_1 + final_distances

        # Step 4: Select top-k based on combined distances
        topk_indices_within_candidates = np.argpartition(
            combined_distances, topk, axis=1
        )[:, :topk]

        topk_indices = candidates[
            np.arange(vecs.shape[0])[:, None], topk_indices_within_candidates
        ]

        # Step 5: Sort top-k indices based on final combined distances
        sorted_order = np.argsort(
            combined_distances[
                np.arange(vecs.shape[0])[:, None],
                topk_indices_within_candidates,
            ],
            axis=1,
        )
        topk_indices = topk_indices[
            np.arange(topk_indices.shape[0])[:, None], sorted_order
        ]

        self.query_time_3 += (
            time.time() - time_log
        )  # Time for top-k selection and sorting
        time_log = time.time()

        return topk_indices

    @tf.function(reduce_retracing=True)
    def encode(self, vecs):
        return self.encoder(vecs, training=False)

    def write(self, path):
        print("Writing index")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(path, "index_norms.pkl"), "wb") as f:
            pickle.dump(self.index_norms_2, f)
        self.ANN.write(os.path.join(path, "faiss_index.bin"))

    def read(self, path, D):
        print("Reading index")
        with open(os.path.join(path, "index.pkl"), "rb") as f:
            self.index = pickle.load(f)
        self.encoder = tf.keras.models.load_model(os.path.join(path, "model.keras"))
        self.ANN.read(os.path.join(path, "faiss_index.bin"), D)
        self.normalizer = joblib.load(os.path.join(path, "normalizer.pkl"))
        self.index_norms_2 = pickle.load(
            open(os.path.join(path, "index_norms.pkl"), "rb")
        )

    def time_log(self):

        time_sum = self.query_time_1 + self.query_time_2 + self.query_time_3

        print(f"query time 1%: {self.query_time_1/time_sum}")
        print(f"\nquery time 1_1%: {self.query_time_1_1/self.query_time_1}")
        print(f"\nquery time 1_2%: {self.query_time_1_2/self.query_time_1}")
        print(f"\nquery time 1_3%: {self.query_time_1_3/self.query_time_1}")

        print(f"query time 2%: {self.query_time_2/time_sum}")
        print(f"\nquery time 2_1%: {self.query_time_2_1/self.query_time_2}")
        print(f"\nquery time 2_2%: {self.query_time_2_2/self.query_time_2}")
        print(f"\nquery time 2_3%: {self.query_time_2_3/self.query_time_2}")

        print(f"query time 3%: {self.query_time_3/time_sum}")
        print(f"\nquery time 3_1%: {self.query_time_3_1/self.query_time_3}")
        print(f"\nquery time 3_2%: {self.query_time_3_2/self.query_time_3}")
        print(f"\nquery time 3_3%: {self.query_time_3_3/self.query_time_3}")

    def stringify_index_param(self, param):
        """Convert index parameters to a string representation."""
        return f"train_size_{self.train_size}_layers_{self.nn_layers}_is_adaptive_{self.isadaptive}"
