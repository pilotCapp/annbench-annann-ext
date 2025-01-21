import os
import pickle

from .base import BaseANN
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

import multiprocessing

num_threads = multiprocessing.cpu_count()
print(num_threads)

import faiss
from .faiss_cpu import LinearANN
faiss.omp_set_num_threads(32)  # Default to the max available threads

from .hnsw import HnswANN


print("TF num physical CPUs:", tf.config.experimental.list_physical_devices('CPU'))
print("TF num logical CPUs:", tf.config.threading.get_intra_op_parallelism_threads())
print("TF num inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())

tf.config.threading.set_intra_op_parallelism_threads(32)  # Number of threads for individual ops
tf.config.threading.set_inter_op_parallelism_threads(16)  # Number of threads for parallel ops

print("TF num physical CPUs:", tf.config.experimental.list_physical_devices('CPU'))
print("TF num logical CPUs:", tf.config.threading.get_intra_op_parallelism_threads())
print("TF num inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())


tf.config.optimizer.set_jit(True)  # Enable XLA


import numpy as np
from pathlib import Path
from hydra.utils import to_absolute_path
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split


@tf.function
def contrastive_loss(y_true, d_pred, margin=1.0):
    # Reshape y_true to match d_pred shape
    y_true = tf.reshape(y_true, tf.shape(d_pred))  # Match shape to d_pred

    y_true = tf.reshape(y_true, tf.shape(d_pred))
    positive_loss = y_true * tf.square(d_pred)
    negative_loss = (1 - y_true) * tf.square(tf.maximum(0.0, margin - d_pred))
    return tf.reduce_mean(positive_loss + negative_loss)

@tf.function
def triplet_margin_loss(y_true, y_pred, margin=0.01):
    """
    Triplet loss that enforces the positive distance is smaller by at least 'margin'
    compared to each negative distance.

    Args:
        y_true: Not actually used (we rely on the structure of y_pred),
                but Keras requires this signature.
        y_pred: Shape (batch_size, K). 
                - y_pred[:, 0] is the anchor->positive distance
                - y_pred[:, 1:] are anchor->negative distances
        margin: The enforced margin between positive and negative distances.

    Returns:
        Scalar loss (tf.Tensor) which is the average margin violation across all negatives.
    """
    # The first column is anchor->positive distance
    pos_dist = y_pred[:, 0:1]  # shape: (batch_size, 1)

    # The remaining columns are anchor->negative distances
    neg_dist = y_pred[:, 5:]   # shape: (batch_size, K-1)

    # Compute hinge-like margin loss per negative:
    # loss = max(0, margin + pos_dist - neg_dist)
    # If pos_dist + margin < neg_dist, no loss; else penalize the violation.
    violations = tf.nn.relu(margin + pos_dist - neg_dist)

    # Reduce over the negative dimension, then average over the batch
    return tf.reduce_mean(violations)


@tf.function
def distance_layer(tensors):
    emb1, emb2, emb3 = tensors
    
    return tf.norm(emb1 - emb3 +5e-9, axis=1, keepdims=True) + tf.norm(emb2 - emb3 +5e-9, axis=1, keepdims=True)
    #(tf.norm(emb1 - emb2, axis=1, keepdims=True), tf.reduce_sum(emb1 * emb2, axis=1, keepdims=True))

@tf.function
def distribution_layer_old(tensors):
    anchor, emb1, emb2, emb3, emb4, emb5, emb6,emb7, emb8, emb9, emb10 = tensors

    distances = [
        tf.norm(anchor - emb1, axis=1, keepdims=True),
        tf.norm(anchor - emb2, axis=1, keepdims=True),
        tf.norm(anchor - emb3, axis=1, keepdims=True),
        tf.norm(anchor - emb4, axis=1, keepdims=True),
        tf.norm(anchor - emb5, axis=1, keepdims=True),
        tf.norm(anchor - emb6, axis=1, keepdims=True),
        tf.norm(anchor - emb7, axis=1, keepdims=True),
        tf.norm(anchor - emb8, axis=1, keepdims=True),
        tf.norm(anchor - emb9, axis=1, keepdims=True),
        tf.norm(anchor - emb10, axis=1, keepdims=True)
    ]
    
    return tf.concat(distances, axis=1)  # shape: (batch_size, 10)

@tf.function
def distribution_layer(tensors):
    # Assume tensors is a list: [anchor, emb1, emb2, ..., embN]
    anchor = tensors[0]
    distances = []
    # Iterate over each embedding in the list (excluding the first anchor)
    for emb in tensors[1:]:
        # Compute distance between anchor and the current embedding
        dist = tf.norm(anchor - emb, axis=1, keepdims=True)
        distances.append(dist)
    # Concatenate all distances: shape will be (batch_size, number_of_embeddings)
    return tf.concat(distances, axis=1)

    


@tf.function
def weighted_mse_loss(y_true, y_pred):
    # Example weight: inverse proportional to true distance (add epsilon to avoid division by zero)
    weights = 1 / (tf.norm(y_true)+1e-8)
    return tf.reduce_mean(weights * tf.square(y_pred - y_true))

@tf.function
def ordering_loss(y_true, y_pred, margin=0):
    """
    y_pred: predicted distances shape (batch_size, 10)
    We assume that y_true is not used here because the loss enforces ordering
    among the predictions directly. 
    """
    # Compute differences between successive distances
    # We want each distance to be less than the next one
    diffs = y_pred[:, 1:] - y_pred[:, :-1]  # shape: (batch_size, 9)
    
    # Penalize if any successive difference is negative (i.e., out of order)
    violations = tf.nn.relu(margin - diffs)  # margin can enforce a gap if desired
    
    # Mean penalty across all violations
    return tf.reduce_mean(violations)

@tf.function
def weighted_ordering_loss(y_true, y_pred, margin=0):
    """
    Applies higher penalties to ordering errors at the top ranks.
    """
    # Compute successive differences
    diffs = y_pred[:, 1:] - y_pred[:, :-1]  # shape: (batch_size, 9)
    
    # Define weights that decrease for lower priority ranks (e.g., more weight on first differences)
    # For instance, weight[0] > weight[1] > ... > weight[8]
    weights = tf.constant([10, 9,8, 7, 6, 5, 4, 3,2], dtype=y_pred.dtype)
    
    # Expand weights to match batch size and differences shape
    weights = tf.reshape(weights, (1, -1))  # shape: (1, 9)
    weights = tf.broadcast_to(weights, tf.shape(diffs))
    
    # Penalize ordering violations with weighted relu
    violations = tf.nn.relu(margin - diffs) * weights
    
    return tf.reduce_mean(violations)


@tf.function
def top1_ranking_loss(y_true, y_pred, margin=0):
    """
    Enforces that the first distance is smaller than all others by at least margin.
    y_pred shape: (batch_size, 10)
    """
    # Extract the first predicted distance
    first_distance = y_pred[:, 0:1]  # shape: (batch_size, 1)
    other_distances = y_pred[:, 1:]  # shape: (batch_size, 9)
    
    # Compute margin differences: first_distance + margin < each of the other distances
    violations = tf.nn.relu(margin + first_distance - other_distances)
    
    # Take mean violation across all pairs for each sample, then across batch
    return tf.reduce_mean(violations)





class Linear_Adaptive(BaseANN):
    def __init__(self):
        self.index = None
        self.index_norms = None
        self.ANN = LinearANN()

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
        self.is_adaptive = True
        self.nn_layers = 2

        self.path = None
        self.encoder = None
        self.normalizer = StandardScaler()

        self.pair_ann = HnswANN()  # use HNSW for cluster lookup
        self.pair_ann.set_index_param({"ef_construction": 300, "M": 32})


    def set_index_param(self, param):
        self.is_adaptive = param["is_adaptive"]

    def train(self, vecs, path):
        self.path = path
        model_path = os.path.join(path, "../model.keras")

        batch_size = 512
        epochs = 100
        n_pairs_closest = 10
        n_pairs_random = 20

        if os.path.exists(model_path):
            print("model already exists")
            self.encoder = tf.keras.models.load_model(model_path)

            self.normalizer = joblib.load(os.path.join(path, "../normalizer.pkl"))
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

            # Build the encoder model
            encoded = input_vec
            for layer in range(self.nn_layers):
                encoded = Dense(self.full_dim, activation="linear")(encoded) #linear activation for simple conversion, relu seems to overfit more
            
            # Linear as output activation
            encoded = Dense(self.full_dim, activation="linear")(encoded)

            # Define the encoder model as a shared encoder for
            shared_encoder = tf.keras.Model(input_vec, encoded)

            #Generate inputs and embeddings for final models
            inputs = [Input(shape=(self.full_dim,), name=f"input_vec_{i}") for i in range(1, n_pairs_closest+2)]
            embeddings = [shared_encoder(inp) for inp in inputs]

            #Models utilizing shared encoder for distance and distribution calculation
            distribution_pre = Lambda(distribution_layer)(embeddings)
            distribution_post = Lambda(distribution_layer)(embeddings[:5])

            distribution_model_pre = tf.keras.Model(inputs=inputs, outputs=distribution_pre)
            distribution_model_post = tf.keras.Model(inputs=inputs[:5], outputs=distribution_post)

            
            ##MAKING TRAINING DATA FROM K NEAREST NEIGHBOURS
            self.pair_ann.add(normalized_vecs)
            closest_train= self.pair_ann.query(vecs_train,n_pairs_closest+1,param ={"ef": 150})
            closest_test= self.pair_ann.query(vecs_test,n_pairs_closest+1,param ={"ef": 150})
            
            #closest_train and closest_test contain each query point as target, because the query is also in the existing data

            # Closest entries training instance
            indices_train_closest = closest_train
            indices_test_closest = closest_test

            #Generate training data
            x_train_closest = normalized_vecs[indices_train_closest]
            anchor = x_train_closest[:, 0:1, :]
            neighbors = x_train_closest[:, 1:, :]
            diff = neighbors - anchor  
            
            y_train_closest = np.linalg.norm(diff, axis=2, keepdims=True).astype(np.float32)
            list_of_inputs = [x_train_closest[:, i, :] for i in range(x_train_closest.shape[1])]

            #Generate validation data
            indices_test = indices_test_closest
            x_test_closest = normalized_vecs[indices_test]
            list_of_validation = [x_test_closest[:, i, :] for i in range(x_test_closest.shape[1])]

            anchor_test = x_test_closest[:, 0:1, :]
            neighbors_test = x_test_closest[:, 1:, :]
            diff_test = neighbors_test - anchor_test  
            
            y_test_closest = np.linalg.norm(diff_test, axis=2, keepdims=True).astype(np.float32)


            ##TRAINING
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        
            #compile the two models, one for the n_pairs_closest entries and one for the n_pairs_closest/2 closest entries (currently 10)
            distribution_model_pre.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
            distribution_model_post.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")

            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        
            distribution_model_pre.fit(list_of_inputs, y_train_closest, validation_data=(list_of_validation, y_test_closest),epochs=400,
            batch_size=batch_size, callbacks=[early_stopping],)
            
            distribution_model_post.fit(list_of_inputs[:5], y_train_closest[:,:4], validation_data=(list_of_validation[:5], y_test_closest[:,:4]),epochs=200,
            batch_size=batch_size, callbacks=[early_stopping],)

            #Save encoder to shared space for adaptive and non-adaptive 
            self.encoder = shared_encoder
            self.encoder.save(model_path)
            joblib.dump(self.normalizer, os.path.join(path, "../normalizer.pkl"))
            print("model saved")
            time.sleep(2) #duplicated training instance bug fix
            
    def has_train(self):
        print("checking if requires training")
        print(self.is_adaptive)
        return (
            not os.path.exists(
                os.path.join(self.stringify_index_param({}), "model.keras")
            )
        )

    def add(self, vecs):

        self.index = self.encoder.predict(self.normalizer.transform(vecs))
        if self.is_adaptive:
            self.ANN.add(self.index[:, : self.partition_dim])
        else:
            self.ANN.add(self.index)
        # self.index_norms_1 = np.sum(self.index[:, : self.partition_dim] ** 2, axis=1)
        self.index_norms_2 = np.sum(self.index[:, self.partition_dim :] ** 2, axis=1)

    # ## Brute force search using numpy, slower but no index generation required for each query vector during pruning

    def query(self, vecs, topk, param=None):
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

        normalized_vecs = self.normalizer.transform(vecs)
        encoded_vecs = self.encode(normalized_vecs).numpy()
        vecs = encoded_vecs

        if not self.is_adaptive:
            topk_indices = self.ANN.query(
            vecs, topk, param, ret_distances=False
            )
            return topk_indices

        
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
            vecs[:, : self.partition_dim], 1000, param, ret_distances=True
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
        self.normalizer = joblib.load(os.path.join(path, "../normalizer.pkl"))
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
        return f"train_size_{self.train_size}_layers_{self.nn_layers}_is_adaptive_{self.is_adaptive}"