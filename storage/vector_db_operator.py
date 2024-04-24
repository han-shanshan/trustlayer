import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

"""
Reference: https://medium.com/loopio-tech/how-to-use-faiss-to-build-your-first-similarity-search-bf0f708aa772
llama index
"""


class SingletonSentenceTransformer:
    _instance = None

    @staticmethod
    def get_instance():
        if SingletonSentenceTransformer._instance is None:
            SingletonSentenceTransformer._instance = SentenceTransformer("paraphrase-mpnet-base-v2")
        return SingletonSentenceTransformer._instance


class VectorDBOperator():
    def __init__(self):
        self.encoder = SingletonSentenceTransformer.get_instance()

    def store_data_to_vector_db(self, data_list, idx_name):
        import faiss
        vectors = self.generate_vectors(data_list)
        vector_dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        faiss.write_index(index, idx_name)
        del index

    def generate_vectors(self, data_list):  # todo: try more encoding methods
        return self.encoder.encode(data_list)

    def search(self, search_text, idx_name, k=10):  # index.ntotal
        import numpy as np
        search_vector = self.encoder.encode(search_text)
        _vector = np.array([search_vector])
        faiss.normalize_L2(_vector)
        index = faiss.read_index(idx_name)
        distances, ann = index.search(_vector, k=k)  # search for all nearest neighbours
        print(f"distance = {distances}")
        print(f"ann = {ann}")
        return pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
