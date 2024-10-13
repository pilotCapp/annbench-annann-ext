from .base import BaseANN
import hnswlib


class HnswANN(BaseANN):
    def __init__(self):
        self.ef_construction, self.M, self.index = None, None, None

    def set_index_param(self, param):
        self.ef_construction = param["ef_construction"]
        self.M = param["M"]

    def has_train(self):
        return False

    def add(self, vecs):
        N, D = vecs.shape
        self.index = hnswlib.Index(space="l2", dim=D)
        self.index.init_index(
            max_elements=N, ef_construction=self.ef_construction, M=self.M
        )
        self.index.add_items(data=vecs)

    def query(self, vecs, topk, param):
        self.index.set_num_threads(1)
        self.index.set_ef(ef=param["ef"])
        labels, _ = self.index.knn_query(data=vecs, k=topk)
        return labels

    def write(self, path):
        self.index.save_index(path_to_index=path)

    def read(self, path, D):
        # self.ef_construction and self.M are not set after reading
        self.index = hnswlib.Index(space="l2", dim=D)
        self.index.load_index(path_to_index=path)

    def stringify_index_param(self, param):
        return f"efc{param['ef_construction']}_M{param['M']}.bin"
