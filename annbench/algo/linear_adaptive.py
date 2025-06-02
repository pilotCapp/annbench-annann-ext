import os
import pickle

from .base import BaseANN
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
import time



import multiprocessing
import faiss
from .faiss_cpu import LinearANN
faiss.omp_set_num_threads(32)  # Default to the max available threads

print("TF num physical CPUs:", tf.config.experimental.list_physical_devices('CPU'))
print("TF num logical CPUs:", tf.config.threading.get_intra_op_parallelism_threads())
print("TF num inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())

tf.config.threading.set_intra_op_parallelism_threads(32)  # Number of threads for individual ops
tf.config.threading.set_inter_op_parallelism_threads(16)  # Number of threads for parallel ops

print("TF num physical CPUs:", tf.config.experimental.list_physical_devices('CPU'))
print("TF num logical CPUs:", tf.config.threading.get_intra_op_parallelism_threads())
print("TF num inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())


tf.config.optimizer.set_jit(True)  # Enable XLA


from .hnsw import HnswANN
import numpy as np
from pathlib import Path
from hydra.utils import to_absolute_path
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import joblib
from sklearn.model_selection import train_test_split


@tf.function
def batch_hard_triplet_loss(y_true, y_pred, margin=1):
    anchor, positive, negative = y_pred[:, 0, :], y_pred[:, 1, :], y_pred[:, 2, :]
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    # Finn de harde negative eksemplene
    hard_negatives = tf.reduce_max(neg_dist)
    
    loss = tf.reduce_mean(tf.maximum(pos_dist + margin - hard_negatives, 0.0))
    return loss

@tf.function
def triplet_loss_old(y_true, y_pred, margin=1):
    """
    Triplet loss function.

    Args:
        margin (float): Margin for triplet loss.

    Returns:
        function: A loss function.
    """

    anchor = y_pred[:, 0, :]   # Shape: (batch_size, embedding_dim)
    positive = y_pred[:, 1, :] # Shape: (batch_size, embedding_dim)
    negative = y_pred[:, 2, :] # Shape: (batch_size, embedding_dim)

    # Compute squared L2 distances
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # Compute triplet loss
    basic_loss =  pos_dist + y_true - neg_dist
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))

    return loss

@tf.function
def triplet_loss(y_true, y_pred, partition_dims=None, bias=0):
    """
    Triplet loss function computed over multiple partitions of the embedding space.
    
    For each partition defined in partition_dims (e.g. [d1, d2, ..., full_dim]),
    the loss is computed using the first k dimensions (0:k) and then scaled by
    k/full_dim. The final loss is the sum of the scaled losses.

    Args:
        y_true: Unused here, but you can use it for margin if needed.
        y_pred: Tensor of shape (batch_size, 3, embedding_dim) containing
                the anchor, positive, and negative embeddings.
        margin (float): Margin for the triplet loss.
        partition_dims (list or tensor of ints): A list of increasing dimension indices.
            For example, if the embedding is 128 dimensions, you might use
            [12, 32, 64, 128]. The loss computed using the first 12 dims will be scaled
            by 12/128, the loss computed with the first 32 dims will be scaled by 32/128, etc.
    
    Returns:
        loss (tf.Tensor): A scalar tensor containing the combined triplet loss.
    """
    # Unpack the triplets.
    anchor   = y_pred[:, 0, :]  # Shape: (batch_size, embedding_dim)
    positive = y_pred[:, 1, :]  # Shape: (batch_size, embedding_dim)
    negative = y_pred[:, 2, :]  # Shape: (batch_size, embedding_dim)
    
    # Determine the full dimensionality (as float for scaling)
    full_dim = tf.cast(tf.shape(anchor)[1], tf.float32)
    
    # If no partition_dims is provided, use the full embedding.
    if partition_dims is None:
        partition_dims = [tf.shape(anchor)[1]]
    else:
        # Ensure partition_dims is a tensor.
        partition_dims = tf.convert_to_tensor(partition_dims, dtype=tf.int32)
    
    total_loss = 0.0

    # Loop over each partition. For each partition, use the first k dimensions.
    for k in partition_dims:
        # Convert k to int32 and float32 as needed.
        k_int   = tf.cast(k, tf.int32)
        k_float = tf.cast(k, tf.float32)
        
        # Extract the relevant partition from each embedding.
        anchor_part   = anchor[:, :k_int]
        positive_part = positive[:, :k_int]
        negative_part = negative[:, :k_int]
        
        # Compute squared L2 distances for the partition.
        pos_dist = tf.reduce_sum(tf.square(anchor_part - positive_part), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor_part - negative_part), axis=1)
        
        # Compute the basic triplet loss for this partition.
        # (You could also use y_true here if you intended to use dynamic margins.)
        basic_loss = pos_dist + y_true - neg_dist
        
        # Apply hinge so that we only penalize when (pos_dist + margin > neg_dist).
        loss_part = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        
        # Scale the loss by the fraction of dimensions used.
        scale = k_float / full_dim
        
        total_loss += scale * loss_part*bias + (1-bias)*loss_part

    return total_loss


@tf.function
def mat_adaptive_loss_old(y_true, y_pred, bias=0.0):
    print(y_pred.shape, y_true.shape)
         
    # Now y_pred has shape (batch_size, 2, num_embeddings)
    distances_first_half = y_pred[:, 0, :]
    distances_full = y_pred[:, 1, :]
    return tf.reduce_mean(tf.square(distances_full - y_true[:,:,0]))

@tf.function
def mat_adaptive_loss(y_true, y_pred, bias=0.0):
    """
    Compute a weighted sum of two MSE losses based on a bias parameter.

    Args:
        y_true: Ground truth tensor.
        y_pred: A list or tuple of predictions. 
                For example, [pred1, pred2] if you have two prediction branches.
        bias: A float value affecting the weighting between the two losses.

    Returns:
        A scalar tensor representing the combined loss.
    """
    distances_first_half = y_pred[:, 0, :]
    distances_full = y_pred[:, 1, :]
    
    # Compute individual MSE losses
    loss1 = tf.reduce_mean(tf.square(distances_first_half - y_true[:,:,0]))
    loss2 = tf.reduce_mean(tf.square(distances_full - y_true[:,:,0]))

    # Combine the losses with weights adjusted by bias
    combined_loss = (0.5 + bias) * loss1 + (1.0 - bias) * loss2
    return loss2
    


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
        self.search_k = 5000

        self.partitions = 2
        self.full_dim = 128
        self.partition_dims = [64]
        self.dimensional_weights = 0

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

        self.path = None
        self.encoder = None
        


    def set_index_param(self, param, ndim=0):
        self.is_adaptive = param["is_adaptive"]
        pd = param.get("partition_dims", "16,128")
        if isinstance(pd, str):
            self.partition_dims = [int(x.strip()) for x in pd.split(",")]
        else:
            self.partition_dims = list(pd)
        self.search_k = param["search_k"]

        if ndim>0:
            self.full_dim = ndim

        # # Create n+1 equally spaced numbers between 0 and 1.
        # x = np.linspace(0, 1, self.partitions+1)
        # power=1.5
        # # Apply a power transformation to create a nonlinear spacing.
        # # (You can adjust `power` to control how fast slice sizes grow.)
        # boundaries = np.round(self.full_dim * x**power).astype(int)
        # # Ensure the first and last boundaries are 0 and total.
        # boundaries[0] = 0
        # boundaries[-1] = self.full_dim
        # # Build a list of slice (start, stop) tuples.
        # slices = [boundaries[i] for i in range(1,self.partitions+1)]
        # self.partition_dims = slices
        # #[16,45,83), np.int64(128)]
        print("partition_dims is ",self.partition_dims)


    def train(self, vecs, path):
        pass
            
    def has_train(self):
        print("checking if requires training")
        return (
            False
        )

    def add(self, vecs):
        self.index = vecs
            
        if self.is_adaptive:
            self.ANN.add(self.index[:, : self.partition_dims[0]])
        else:
            self.ANN.add(self.index)
        # self.index_norms_1 = np.sum(self.index[:, : self.partition_dim] ** 2, axis=1)
        self.index_norms = np.sum(self.index[:, self.partition_dims[0] :] ** 2, axis=1)

    # ## Brute force search using numpy, slower but no index generation required for each query vector during pruning

    def query(self, vecs, topk, param=None):
        """
        Finds the top-k nearest neighbors for each query vector in vecs,
        using an initial faiss query on the first partition followed by
        iterative pruning on each additional partition in self.partition_dims.
    
        Args:
            vecs (np.ndarray): 2D array where each row is a query vector.
            topk (int): Number of nearest neighbors to return.
            param (dict, optional): Dictionary of parameters. Expected keys:
                - "search_k": candidate pool size for the faiss query.
                - "prune_factor": factor to multiply topk for candidate pruning.
                  (Default is 10 if not provided.)
        
        Returns:
            np.ndarray: 2D array of shape (len(vecs), topk) with the indices
            of the top-k nearest neighbors for each query.
        """        # Set candidate pool sizes and prune factor (fallback defaults)
        #search_k = int(self.partition_dims[0]/self.partition_dims[-1]*len(self.index))

        
        # (Optional) timing logs
        t_start = time.time()
    
        # Normalize and encode the query vectors (your encoding step)
    
        # If not in adaptive mode, use a plain ANN query and return.
        if not self.is_adaptive:
            return self.ANN.query(vecs, topk, param=None, ret_distances=False)
    
        # Assume self.partition_dims is a list of increasing boundaries.
        # For example, if total dimension is 128:
        #    self.partition_dims = [d1, d2, 128]
        partition_dims = self.partition_dims
        num_queries = vecs.shape[0]
    
        # --- Stage 1: Initial Candidate Generation via faiss on first partition ---
        # Use the first partition (dimensions 0:partition_dims[0]) for the ANN query.
        candidates, distances = self.ANN.query(
            vecs[:, :partition_dims[0]], self.search_k, param=None, ret_distances=True
        )
        # 'candidates' is assumed to be of shape (num_queries, search_k)
        # and 'distances' of the same shape.
    
        # --- Stage 2: Iterative Refinement over subsequent partitions ---
        # For each additional partition (i=1,2,...), update the cumulative distance
        # and prune the candidate set.
        for i in range(1, len(partition_dims)):
            if partition_dims[i]>param["partition_stop"]:
                break
            # Define the slice for the current partition:
            start = partition_dims[i - 1]
            stop = partition_dims[i]
    
            # Extract the portion of the query vectors for the current partition.
            query_part = vecs[:, start:stop]  # shape: (num_queries, slice_dim)
    
            # Retrieve the candidate vectors for this partition.
            # Using advanced indexing, the result has shape:
            # (num_queries, current_candidate_count, slice_dim)
            candidate_vectors = self.index[candidates, start:stop]
    
            # Get (or compute) the norms for the candidate vectors in this partition.
            # If you have precomputed norms for each partition stored as a list (with one entry per partition),
            # you can do:
            #
            #candidate_norms = self.index_norms_list[i][candidates]
            #
            # Otherwise, compute on the fly:
            candidate_norms = np.sum(candidate_vectors**2, axis=2)  # shape: (num_queries, num_candidates)
    
            # Compute dot products between the query partition and candidate vectors.
            dot_products = np.einsum("ij,ikj->ik", query_part, candidate_vectors)
            # Compute norms for the query part.
            query_norms = np.sum(query_part**2, axis=1, keepdims=True)
    
            # Compute the squared Euclidean distance for this partition:
            partition_distances = candidate_norms - 2 * dot_products + query_norms
    
            # Update the cumulative distances.
            distances = distances + partition_distances
    
            # --- Prune the candidate set ---
            # Keep only the top (topk * prune_factor) candidates for each query.
            new_pool_size = int((stop-start)*self.partition_dims[-1]*len(candidates))
            k = min(new_pool_size, distances.shape[1] - 1)
            top_candidate_idxs = np.argpartition(distances, k, axis=1)[:, :new_pool_size]
            # Use np.take_along_axis to keep only the selected candidates and their distances.
            candidates = np.take_along_axis(candidates, top_candidate_idxs, axis=1)
            distances = np.take_along_axis(distances, top_candidate_idxs, axis=1)
    
        # --- Stage 3: Final Selection ---
        # From the final candidate set, choose the topk candidates based on the cumulative distance.
        topk_candidate_idxs = np.argpartition(distances, topk, axis=1)[:, :topk]
        final_candidates = np.take_along_axis(candidates, topk_candidate_idxs, axis=1)
        # Sort these final candidates according to their distances.
        sorted_order = np.argsort(np.take_along_axis(distances, topk_candidate_idxs, axis=1), axis=1)
        final_candidates = np.take_along_axis(final_candidates, sorted_order, axis=1)
    
        # (Optional) log query time if desired:
        # self.query_time_total += time.time() - t_start
    
        return final_candidates

    def write(self, path):
        print("Writing index")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(path, "index_norms.pkl"), "wb") as f:
            pickle.dump(self.index_norms, f)
        self.ANN.write(os.path.join(path, "faiss_index.bin"))

    def read(self, path, D):
        print("Reading index")
        with open(os.path.join(path, "index.pkl"), "rb") as f:
            self.index = pickle.load(f)
        self.ANN.read(os.path.join(path, "faiss_index.bin"), D)
        self.index_norms = pickle.load(
            open(os.path.join(path, "index_norms.pkl"), "rb")
        )
        self.full_dim = self.index.shape[1]


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
        return f"is_adaptive_{self.is_adaptive}_partitions_{self.partitions}"
