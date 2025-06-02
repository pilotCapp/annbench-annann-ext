from .base import BaseDataset
from ..util import ivecs_read, fvecs_read

from urllib.request import urlretrieve
import tarfile


class Paq1m(BaseDataset):
    def __init__(self, path):
        super().__init__(path=path)

    def download(self):
        pass

    def vecs_train(self):
        vec_path = self.path / "paq/paq_learn.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))

    def vecs_base(self):
        vec_path = self.path / "paq/paq_base.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))

    def vecs_query(self):
        vec_path = self.path / "paq/paq_query.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))

    def groundtruth(self):
        vec_path = self.path / "paq/paq_groundtruth.ivecs"
        assert vec_path.exists()
        return ivecs_read(fname=str(vec_path))
