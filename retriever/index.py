import os
import logging
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from tqdm import tqdm, trange

logger = logging.getLogger()

FAISSINDEX_DICT = {
    "inner_product": faiss.IndexFlatIP,
    "l2": faiss.IndexFlatL2,
}

class Indexer(object):

    def __init__(self, vector_sz, metric="inner_product", n_subquantizers=0, n_bits=8):
        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = FAISSINDEX_DICT[metric](vector_sz)
        self.index_id_to_db_id = np.empty((0), dtype=np.int64)

    def index_data(self, ids, embeddings):

        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        
        logger.info(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size=1024, verbose: bool=True) -> List[Tuple[List[object], List[float]]]:
        query_vectors = query_vectors.astype('float32')
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        # for k in tqdm(range(nbatch)):
        if verbose:
            progress_bar = trange(nbatch, desc="KNN Search")
        for k in range(nbatch):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
            if verbose:
                progress_bar.update(1)
        return result

    def serialize(self, dir_path):
        # index_file = dir_path / 'index.faiss'
        # meta_file = dir_path / 'index_meta.dpr'
        index_file = os.path.join(dir_path, "index.faiss")
        meta_file = os.path.join(dir_path, "index_meta.faiss")
        logger.info(f'Serializing index to {index_file}, meta data to {meta_file}')

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        # index_file = dir_path / 'index.faiss'
        # meta_file = dir_path / 'index_meta.dpr'
        index_file = os.path.join(dir_path, "index.faiss")
        meta_file = os.path.join(dir_path, "index_meta.faiss")
        logger.info(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
        logger.info('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        new_ids = np.array(db_ids, dtype=np.int64)
        self.index_id_to_db_id = np.concatenate((self.index_id_to_db_id, new_ids), axis=0)


# examples 

def build_entity_embedding_index():

    embedding_size = 768 
    indexer = Indexer(embedding_size)

    index_path = "/nfs/common/data/usml/index"
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    
    all_entity_embeddings = np.load("/nfs/common/data/usml/sapbert_all_entity_embeddings.npy")
    all_entity_ids = list(range(len(all_entity_embeddings)))
    buffer_size = 50000
    for i in trange((len(all_entity_ids)-1) // buffer_size + 1):
        batch_entity_ids = all_entity_ids[i*buffer_size: (i+1)*buffer_size]
        batch_entity_embeddings = all_entity_embeddings[i*buffer_size: (i+1)*buffer_size]
        indexer.index_data(batch_entity_ids, batch_entity_embeddings)
    
    indexer.serialize(index_path)
    print(f"Successfully save index to {index_path}!")
    