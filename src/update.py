
from typing import Dict, Callable
from elv_client_py import ElvClient
import os
import shutil
import logging
import threading
from sklearn.cluster import KMeans
import numpy as np

import src.config as config
from src.index import Index
from src.embedding import TextEncoder
from src.utils import timeit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndexBuilder():
    class UpdateStatus():
        def __init__(self, status: str, progress: float):
            self.status = status
            self.progress = progress
            self.stop_event = threading.Event()
            self.error = None
    
    # Args:
    #   encoder: Used for embedding text fields
    #   preprocess: Function to preprocess documents before indexing
    def __init__(self, encoder: TextEncoder):
        self.encoder = encoder
        self.running_tasks = {}
        self.ended_tasks = {}
        self.lock = threading.Lock()

    def stop(self, qid: str, status: str="stopped") -> UpdateStatus:
        with self.lock:
            if qid not in self.running_tasks:
                return None
            task = self.running_tasks[qid]
            task.stop_event.set()
            shutil.rmtree(os.path.join(config.TMP_PATH, qid), ignore_errors=True)
            task.status = status
            self.ended_tasks[qid] = task
            del self.running_tasks[qid]
            return task

    def get_status(self, content_id: str) -> UpdateStatus:
        with self.lock:
            if content_id in self.running_tasks:
                return self.running_tasks[content_id]
            elif content_id in self.ended_tasks:
                return self.ended_tasks[content_id]
            else:
                return None

    """
    Args:
        content_id: index object id to base vector index off of

    Saves a vector index for the given content_id which can be searched.
    """
    def build(self, content_id: str, index: Index, client: ElvClient) -> None:
        with self.lock:
            if content_id in self.running_tasks:
                logging.error(f"Indexing already in progress for {content_id}.")
                return
            update_state = self.UpdateStatus('running', 0)
            self.running_tasks[content_id] = update_state
        try:
            self._build(content_id, index, client)
            self.stop(content_id, status="finished")
        except Exception as e:
            with self.lock:
                task = self.running_tasks[content_id]
                task.status = 'error'
                task.error = str(e.with_traceback(None))
                self.ended_tasks[content_id] = task
                del self.running_tasks[content_id]
        
    def _build(self, content_id: str, index: Index, client: ElvClient):
        update_state = self.running_tasks[content_id]
        field_configs = self._get_field_configs(content_id, client)
        num_docs = self._get_num_docs(content_id, client)
        # Unpack the entire index with a "select all" query
        with timeit("unpacking index"):
            all_docs = client.search(object_id=content_id, query={"filters": "has_field:id", "max_total": num_docs, "limit": num_docs, "display_fields": ["all"]})['results']
        logging.info("Embedding documents and adding to index.")
        for i, doc in enumerate(all_docs):
            if update_state.stop_event.is_set():
                logging.info("Stopping indexing.")
                update_state.status = "stopped"
                return
            uid = f"{doc['hash']}{doc['prefix']}"
            field_embeddings = {fname: text for fname, text in doc['fields'].items() if fname[2:] in field_configs and field_configs[fname[2:]]['type'] == 'text'}
            embeddings = self.encoder.encode(field_embeddings)
            for field, values in embeddings.items():
                index.add(field, uid, values)
            update_state.progress = (i+1) / len(all_docs)
        with timeit("committing index"):
            index.commit()
        update_state.status = "complete"
        logging.info("Indexing complete.")
    
    def cleanup(self) -> None:
        with self.lock:
            for qid in self.running_tasks:
                self.running_tasks[qid].stop_event.set()
                del self.running_tasks[qid]
            # TODO: this global path might eventually be shared so probably should parameterize it
            shutil.rmtree(config.TMP_PATH, ignore_errors=True)

    def _get_field_configs(self, content_id: str, client: ElvClient) -> Dict[str, Dict[str, float]]:
        res = client.content_object_metadata(object_id=content_id, metadata_subtree='indexer/config/indexer/arguments/fields')
        return res

    # Retrieves the total number of documents in the index
    def _get_num_docs(self, content_id: str, client: ElvClient) -> int:
        res = client.content_object_metadata(object_id=content_id, metadata_subtree='indexer/stats/document/total')
        return int(res)
    
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
