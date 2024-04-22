import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

"""
Reference: https://medium.com/loopio-tech/how-to-use-faiss-to-build-your-first-similarity-search-bf0f708aa772
llama index
"""


class VectorDBOperator():
    def __init__(self):
        self.encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

    def store_data_to_db(self, data_list, idx_name):
        import faiss
        vectors = self.generate_vectors(data_list)
        # vectors = self.encoder.encode(data_list)
        vector_dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        faiss.write_index(index, idx_name)
        del index

    def generate_vectors(self, data_list):
        # """ Generate embeddings for each text input using BERT """
        # encoded_input = self.tokenizer(data_list, padding=True, truncation=True, return_tensors='pt')
        # with torch.no_grad():
        #     model_output = self.model(**encoded_input)
        # # Use the average of the last hidden states as the sentence embedding
        # vectors = model_output.last_hidden_state.mean(dim=1).numpy()
        # return vectors
        return self.encoder.encode(data_list)

    def search(self, search_text, idx_name):
        import numpy as np
        search_vector = self.encoder.encode(search_text)
        _vector = np.array([search_vector])
        faiss.normalize_L2(_vector)
        index = faiss.read_index(idx_name)
        distances, ann = index.search(_vector, k=index.ntotal)  # search for all nearest neighbours
        results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
        print(results)

