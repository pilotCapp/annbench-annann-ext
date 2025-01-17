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
def distance_layer(tensors):
    emb1, emb2, emb3 = tensors
    
    return tf.norm(emb1 - emb3 +5e-9, axis=1, keepdims=True) + tf.norm(emb2 - emb3 +5e-9, axis=1, keepdims=True)
    #(tf.norm(emb1 - emb2, axis=1, keepdims=True), tf.reduce_sum(emb1 * emb2, axis=1, keepdims=True))

@tf.function
def weighted_mse_loss(y_true, y_pred):
    # Example weight: inverse proportional to true distance (add epsilon to avoid division by zero)
    weights = 1 / (tf.norm(y_true)+1e-8)
    return tf.reduce_mean(weights * tf.square(y_pred - y_true))





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

            # Build the encoder
            encoded = input_vec
            for layer in range(self.nn_layers):
                encoded = Dense(self.full_dim, activation="linear")(encoded)

            # Use 'sigmoid' activation in the final layer if your input data is normalized between 0 and 1
            encoded = Dense(self.full_dim, activation="linear")(encoded)

            # Define the encoder model
            shared_encoder = tf.keras.Model(input_vec, encoded)


            input_vec_1 = Input(shape=(self.full_dim,))
            input_vec_2 = Input(shape=(self.full_dim,))
            input_vec_3 = Input(shape=(self.full_dim,))
            

            emb_1 = shared_encoder(input_vec_1)
            emb_2 = shared_encoder(input_vec_2)
            emb_3 = shared_encoder(input_vec_3)

            distance = Lambda(distance_layer)([emb_1, emb_2, emb_3])
            distance_model = tf.keras.Model(inputs=[input_vec_1, input_vec_2, input_vec_3], outputs=distance)

            optimizer = tf.keras.optimizers.Adam()
            
            distance_model.compile(optimizer=optimizer, loss = weighted_mse_loss)


            batch_size = 512
            epochs = 100
            n_pairs_closest = 101
            n_pairs_random =21

            ##MAKING PAIRS
            self.pair_ann.add(vecs_train)
            closest= self.pair_ann.query(vecs_train,n_pairs_closest,param ={"ef": 150})
            
            closest=closest[:,1:]
                        


            # Closest entries training instance
            indices_train_closest = closest
            indices_train_random = np.random.choice(len(vecs_train), (len(vecs_train), n_pairs_random))

            # Shuffle the appended array
            x1_train_closest = np.repeat(vecs_train, n_pairs_closest-1, axis=0)
            x2_train_closest = vecs_train[indices_train_closest.flatten()]
            x3_train_closest = (x2_train_closest+x1_train_closest)/2
            y_train_closest=np.linalg.norm(x1_train_closest - x2_train_closest, axis=1, keepdims=True).astype(np.float32)
            
            x1_train_random = np.repeat(vecs_train, n_pairs_random, axis=0)
            x2_train_random = vecs_train[indices_train_random.flatten()]
            y_train_random = np.linalg.norm(x1_train_random - x2_train_random, axis=1, keepdims=True).astype(np.float32)

            x1_train = np.concatenate((x1_train_closest, x1_train_random), axis=0)
            x2_train = np.concatenate((x2_train_closest, x2_train_random), axis=0)
            x3_train = (x2_train+x1_train)/2
            
            y_train = np.concatenate((y_train_closest, y_train_random), axis=0)

            print(x1_train.shape,x2_train.shape,y_train.shape)

            # Shuffle the combined pairs together
            indices = np.arange(x1_train.shape[0])
            np.random.shuffle(indices)
            x1_train = x1_train[indices]
            x2_train = x2_train[indices]
            x3_train = x3_train[indices]
            y_train= y_train[indices]
            
            #distance_train = np.linalg.norm(x1_train - x2_train, axis=1)
            #dot_product_train = np.sum(x1_train * x2_train, axis=1)
            #y_train = distance_train#(distance_train, dot_product_train)
             


            #validation test instance
            indices_test = np.random.choice(len(vecs_test), (len(vecs_test), n_pairs_random))
            x1_test = np.repeat(vecs_test, n_pairs_random, axis=0)
            x2_test = vecs_test[indices_test.flatten()]

            x3_test = (x2_test+x1_test)/2

            
            
            distance_test = np.linalg.norm(x1_test - x2_test, axis=1)
            dot_product_test = np.sum(x1_test * x2_test, axis=1)
            y_test = np.linalg.norm(x1_test - x2_test, axis=1, keepdims=True).astype(np.float32)
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
            


            distance_model.fit([x1_train, x2_train, x3_train], y_train,validation_data=([x1_test, x2_test, x3_test], y_test), epochs=epochs,
            batch_size=batch_size, callbacks=[early_stopping],)

            distance_model.compile(optimizer=tf.keras.optimizers.Adam(), loss = weighted_mse_loss)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=8, restore_best_weights=True
            )

            distance_model.fit([x1_train_closest, x2_train_closest, x3_train_closest], y_train_closest, epochs=epochs,
            batch_size=batch_size, callbacks=[early_stopping],)

            self.encoder = shared_encoder
            self.encoder.save(model_path)
            joblib.dump(self.normalizer, os.path.join(path, "../normalizer.pkl"))
            print("model saved")

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
            vecs[:, : self.partition_dim], topk * 1000, param, ret_distances=True
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