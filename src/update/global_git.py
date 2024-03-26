
import logging
import numpy as np
from elv_client_py import ElvClient
from sklearn.cluster import KMeans

from src.index.faiss import Index
from src.update.builder import IndexBuilder
from src.embedding import TextEncoder
from src.utils import timeit

class GlobalGitBuilder(IndexBuilder):
    def __init__(self, encoder: TextEncoder, K: int=100):
        super().__init__(encoder)
        self.K = K

    def _build(self, content_id: str, index: Index, client: ElvClient):
        update_state = self.running_tasks[content_id]
        field_configs = self._get_field_configs(content_id, client)
        num_docs = self._get_num_docs(content_id, client)
        # Unpack the entire index with a "select all" query
        with timeit("unpacking index"):
            all_docs = client.search(object_id=content_id, query={"filters": "has_field:id", "max_total": num_docs, "limit": num_docs, "display_fields": ["all"]})['results']
        logging.info("Embedding documents and adding to index.")
        object_embeddings = []
        for i, doc in enumerate(all_docs):
            if update_state.stop_event.is_set():
                logging.info("Stopping indexing.")
                update_state.status = "stopped"
                return
            uid = f"{doc['hash']}{doc['prefix']}"
            field_embeddings = {fname: text for fname, text in doc['fields'].items() if fname[2:] in field_configs and field_configs[fname[2:]]['type'] == 'text'}
            embeddings = self.encoder.encode(field_embeddings)
            if 'f_object' in embeddings:
                object_embeddings.extend(e for e in embeddings['f_object'])
            for field, values in embeddings.items():
                if field != 'f_object':
                    index.add(field, uid, values)
            update_state.progress = (i+1) / len(all_docs)
        # get k means from object embeddings
        if object_embeddings:
            kmeans = KMeans(n_clusters=self.K, n_init=10)
            object_embeddings = np.array(object_embeddings)
            kmeans.fit(object_embeddings)
        # go through all docs and for each object embedding, find the closest cluster and add that cluster to the index
        for i, doc in enumerate(all_docs):
            uid = f"{doc['hash']}{doc['prefix']}"
            field_embeddings = {fname: text for fname, text in doc['fields'].items() if fname[2:] in field_configs and field_configs[fname[2:]]['type'] == 'text'}
            embeddings = self.encoder.encode(field_embeddings)
            if 'f_object' in embeddings:
                for e in embeddings['f_object']:
                    nearest, _ = self.find_nearest_neighbor(kmeans.cluster_centers_, e)
                    index.add('f_object', uid, np.expand_dims(nearest, 0))
        with timeit("committing index"):
            index.commit()
        update_state.status = "complete"
        logging.info("Indexing complete.")
    
    def set_k(self, K: int) -> None:
        self.K = K

    def find_nearest_neighbor(self, vectors, query_vector):
        # Calculate the Euclidean distances between the query vector and all other vectors
        distances = np.linalg.norm(vectors - query_vector, axis=1)
        
        # Find the index of the smallest distance
        nearest_index = np.argmin(distances)
        
        # Return the nearest vector and its index
        return vectors[nearest_index], nearest_index
