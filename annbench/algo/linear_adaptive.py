import os
import pickle
from .base import BaseANN
import numpy as np
import time

from .faiss_cpu import LinearANN


class Linear_Adaptive(BaseANN):
    def __init__(self):
        self.index = None
        self.index_norms = None
        self.ANN = LinearANN()
        self.ANN_2 = LinearANN()

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

        self.partitions = 1
        self.partition_dim = 64

    def set_index_param(self, param):
        pass
        # self.partitions = param[
        #     "partitions"
        # ]  # Number of vector partitions, 1 is default for KNN brute force

    def train(self, vecs):
        pass

    def has_train(self):
        return False

    def add(self, vecs):
        self.index = vecs
        self.index_norms_1 = np.sum(self.index[:, : self.partition_dim] ** 2, axis=1)
        self.ANN.add(vecs[:, : self.partition_dim])

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

        vecs_norms = np.sum(vecs[:, : self.partition_dim] ** 2, axis=1)

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
            vecs[:, : self.partition_dim], topk * 100, param
        )

        self.query_time_1_2 += (
            time.time() - time_log_1
        )  # Time for initial distance calculation
        time_log_1 = time.time()

        # Get the indices of the candidates

        self.query_time_1 += (
            time.time() - time_log
        )  # Time for initial distance calculation
        time_log = time.time()

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

        self.query_time_2 += (
            time.time() - time_log
        )  # Time for second half distance calculation
        time_log = time.time()

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
                np.arange(vecs.shape[0])[:, None], topk_indices_within_candidates
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

    ## Using faiss KNN, faster but requires index generation for each query vector during pruning
    # def query(self, vecs, topk, param=None):
    #     time_log = time.time()

    #     # Initial candidate search
    #     candidates, distances = self.ANN.query(
    #         vecs[:, : int(128 / self.partitions)], topk * 100, param
    #     )

    #     unique_candidates = np.unique(candidates.flatten())
    #     print(len(unique_candidates))

    #     self.query_time_1 += time.time() - time_log  # Time for initial candidate search
    #     time_log = time.time()

    #     # Add unique candidates to ANN_2
    #     self.ANN_2.add(self.index[unique_candidates, int(128 / self.partitions) :])

    #     self.query_time_2 += (
    #         time.time() - time_log
    #     )  # Time for candidate index generation
    #     time_log = time.time()

    #     # Final search
    #     finalists, final_distances = self.ANN_2.query(
    #         vecs[:, int(128 / self.partitions) :], topk * 10, param
    #     )
    #     print(finalists[-5:])

    #     # Calculate combined distances
    #     print(distances.shape)
    #     print(final_distances.shape)
    #     combined_distances = distances[:,finalists] + final_distances

    #     # Get top-k winners
    #     winners = np.argsort(combined_distances, axis=1)[:, :topk]

    #     self.query_time_3 += time.time() - time_log  # Time for final search
    #     time_log = time.time()

    #     return candidates[winners]

    def write(self, path):
        self.ANN.write(path)

    def read(self, path, D):
        self.ANN.read(path, D)

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
        return self.ANN.stringify_index_param(param)
