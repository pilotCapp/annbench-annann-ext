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

def KNN_CONTRASTIVE_LOSS(K, alpha=1.0, beta=1.0, margin=1.0):
    @tf.function
    def loss_function(y_true, y_pred):
        z = y_pred  # Encoded vectors

        # Compute pairwise distances in the original space
        distances_x = tf.norm(
            tf.expand_dims(y_true, axis=1) - tf.expand_dims(y_true, axis=0), axis=-1
        )  # Shape: (batch_size, batch_size)

        # Exclude self-distances by setting diagonals to a large value
        batch_size = tf.shape(y_true)[0]
        K_adjusted = tf.minimum(K, batch_size - 2)  # Ensure at least one negative
        mask = tf.eye(batch_size, dtype=tf.bool)
        distances_x = tf.where(mask, tf.constant(1e10, dtype=tf.float32), distances_x)

        # Get indices of K nearest neighbors in the original space
        _, knn_indices_x = tf.math.top_k(
            -distances_x, k=K_adjusted
        )  # Shape: (batch_size, K_adjusted)

        # Gather K nearest neighbors in latent space
        knn_z = tf.gather(
            z, knn_indices_x, batch_dims=0
        )  # Shape: (batch_size, K_adjusted, latent_dim)

        # Compute distances to K nearest neighbors in latent space
        distances_z = tf.norm(
            tf.expand_dims(z, axis=1) - knn_z, axis=-1
        )  # Shape: (batch_size, K_adjusted)

        # KNN loss: minimize distances to neighbors
        knn_loss = tf.reduce_mean(distances_z)

        # Contrastive loss: maximize distances to non-neighbors

        # Create a mask for valid negatives (not self and not neighbors)
        # First, create a (batch_size, batch_size) mask
        neighbor_mask = tf.reduce_any(
            tf.equal(
                tf.expand_dims(tf.range(batch_size), axis=-1),
                knn_indices_x[:, tf.newaxis, :],
            ),
            axis=-1,
        )  # Shape: (batch_size, batch_size)
        invalid_mask = tf.logical_or(
            mask, neighbor_mask
        )  # Shape: (batch_size, batch_size)
        valid_negatives_mask = tf.logical_not(
            invalid_mask
        )  # Shape: (batch_size, batch_size)

        # For each sample, get the indices of valid negatives
        # Replace invalid negatives with -1 to ignore them
        neg_indices = tf.where(
            valid_negatives_mask, tf.expand_dims(tf.range(batch_size), axis=0), -1
        )
        # However, tf.where will flatten, so instead, use masked indices per sample

        # To sample negatives, we can use masking and gather
        # Compute the number of possible negatives per sample
        num_possible_negatives = tf.reduce_sum(
            tf.cast(valid_negatives_mask, tf.int32), axis=1
        )
        # Set the number of negatives to sample per sample
        num_negatives = tf.minimum(K_adjusted, tf.reduce_min(num_possible_negatives))

        # To sample negatives uniformly, shuffle the valid indices and take the first num_negatives
        # Create a tensor of valid indices per sample
        def sample_negatives(args):
            idx, mask = args
            valid_indices = tf.boolean_mask(tf.range(batch_size), mask)
            shuffled = tf.random.shuffle(valid_indices)
            sampled = shuffled[:num_negatives]
            # Pad if not enough samples
            padding = num_negatives - tf.shape(sampled)[0]
            sampled = tf.cond(
                padding > 0,
                lambda: tf.concat([sampled, tf.fill([padding], -1)], axis=0),
                lambda: sampled,
            )
            return sampled

        sampled_neg_indices = tf.map_fn(
            sample_negatives,
            (tf.range(batch_size), valid_negatives_mask),
            dtype=tf.int32,
        )  # Shape: (batch_size, num_negatives)

        # Replace -1 indices with 0 to prevent errors in gather (they will be ignored later)
        sampled_neg_indices = tf.maximum(sampled_neg_indices, 0)

        # Gather negative samples
        neg_z = tf.gather(
            z, sampled_neg_indices, batch_dims=0
        )  # Shape: (batch_size, num_negatives, latent_dim)

        # Compute distances to negative samples
        distances_neg = tf.norm(
            z[:, tf.newaxis, :] - neg_z, axis=-1
        )  # Shape: (batch_size, num_negatives)

        # If any sampled_neg_indices were -1, set their distances to 0 to ignore them in loss
        mask_neg = tf.cast(
            sampled_neg_indices >= 0, tf.float32
        )  # Shape: (batch_size, num_negatives)
        distances_neg = distances_neg * mask_neg  # Ignore invalid negatives

        # Apply hinge loss: max(0, margin - distance)
        hinge_losses = tf.maximum(
            0.0, margin - distances_neg
        )  # Shape: (batch_size, num_negatives)

        # Compute contrastive loss, ignoring padded negatives
        contrastive_loss = tf.reduce_sum(hinge_losses) / tf.reduce_sum(mask_neg + 1e-8)

        # Total loss
        total_loss = alpha * knn_loss + beta * contrastive_loss

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

        self.ef_search = 100

        self.ef_construction = None
        self.M = None

        self.encoding_dim = 0.5
        self.encoder = None
        self.path = None

        self.hnsw = HnswANN()  # use HNSW for cluster lookup

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
        self.ef_construction = param["ef_construction"]
        self.M = param["M"]

        self.hnsw.set_index_param(
            {"ef_construction": self.ef_construction * 2, "M": self.M}
        )

    def train(self, vecs, path):
        self.path = path
        model_path = os.path.join(path, "model.keras")

        self.normalizer.fit(vecs)
        normalized_vecs = self.normalizer.transform(vecs)

        if self.isdynamic:
            normalized_vecs = normalized_vecs[: int(len(normalized_vecs) / 2)]

        if os.path.exists(model_path):
            print("model already exists")
            self.encoder = tf.keras.models.load_model(model_path)

            self.normalizer = joblib.load(os.path.join(path, "normalizer.pkl"))
            print(
                "encoder retrieved with input/output shape",
                self.encoder.input_shape,
                self.encoder.output_shape,
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

            # Define the encoder model
            self.encoder = tf.keras.models.Model(input_vec, encoded)

            ## KNN_LOSS

            # Define Early Stopping Callback
            early_stopping_final = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            )

            # Create tf.data.Datasets with drop_remainder=True
            batch_size = 2500

            # Training Dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((vecs_train, vecs_train))
            train_dataset = train_dataset.shuffle(buffer_size=1024)
            train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
            train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

            # Validation Dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((vecs_test, vecs_test))
            val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
            val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

            # Instantiate the loss function
            loss_fn = KNN_CONTRASTIVE_LOSS(K=1)

            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, clipnorm=1.0)
            self.encoder.compile(optimizer=optimizer, loss=loss_fn)

            # Train the Model
            self.encoder.fit(
                train_dataset,
                batch_size= tf.shape(vecs_train)[0].numpy(),
                epochs=50,
                #validation_data=val_dataset,
                callbacks=[early_stopping_final],
            )

            self.encoder.compile(optimizer="adam", loss="mean_squared_error")
            self.encoder.save(model_path)
            joblib.dump(self.normalizer, os.path.join(path, "normalizer.pkl"))
            print("model saved")

    def has_train(self):
        return not os.path.exists(os.path.join(self.path, "model.keras"))

    def add(self, vecs):
        normalized_vecs = self.normalizer.transform(vecs)
        encoded_vecs = self.encode(normalized_vecs).numpy()
        self.hnsw.add(encoded_vecs)

    def query(self, vecs, topk, param):
        # vecs = tf.convert_to_tensor(vecs)
        normalized_vecs = self.normalizer.transform(vecs)
        encoded_queries = self.encode(normalized_vecs).numpy()
        # Encode the query vectors

        results = self.hnsw.query(
            encoded_queries,
            topk,  # number k to find
            param,
        )

        return results

    @tf.function(reduce_retracing=True)
    def encode(self, vecs):
        return self.encoder(vecs, training=False)

    # @tf.function(reduce_retracing=True)
    # def decode(self, vecs):
    #     return self.decoder(vecs, training=False)

    def write(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        self.hnsw.write(os.path.join(path, "hnsw"))

    def read(self, path, D):
        print("Reading index")
        with open(os.path.join(path, "index.pkl"), "rb") as f:
            self.index = pickle.load(f)
        self.encoder = tf.keras.models.load_model(os.path.join(path, "model.keras"))
        self.encode(np.zeros((1, self.input_dim)))
        print(
            "encoder retrieved with input/output shape",
            self.encoder.input_shape,
            self.encoder.output_shape,
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
        return f"test_size_{self.test_size}_encodingdim_{self.encoding_dim}_layers_{self.nn_layers}_dynamic_{self.isdynamic}"


# query time 1%: 0.07688888739751326
# query time 2%: 0.11602366853797023
# query time 3%: 0.8070874440645165

# query time 3_1%: 0.13722583539672958

# query time 3_2%: 0.28620613149015445

# query time 3_3%: 0.5731103718875663
