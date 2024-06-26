
from typing import Dict
from elv_client_py import ElvClient
import os
import shutil
import logging
import threading

import src.config as config
from src.index.faiss import Index
from src.embedding.abstract import DocEncoder
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
    def __init__(self, encoder: DocEncoder):
        self.encoder = encoder
        self.running_tasks = {}
        self.ended_tasks = {}
        self.lock = threading.Lock()

    def stop(self, hash: str, status: str="stopped") -> UpdateStatus:
        with self.lock:
            if hash not in self.running_tasks:
                return None
            task = self.running_tasks[hash]
            task.stop_event.set()
            shutil.rmtree(os.path.join(config.TMP_PATH, hash), ignore_errors=True)
            task.status = status
            self.ended_tasks[hash] = task
            del self.running_tasks[hash]
            return task

    def get_status(self, hash: str) -> UpdateStatus:
        with self.lock:
            if hash in self.running_tasks:
                return self.running_tasks[hash]
            elif hash in self.ended_tasks:
                return self.ended_tasks[hash]
            else:
                return None

    """
    Args:
        hash: version hash of index object to base vector index off of

    Saves a vector index for the given content_id which can be searched.
    """
    def build(self, hash: str, index: Index, client: ElvClient) -> None:
        with self.lock:
            if hash in self.running_tasks:
                logging.error(f"Indexing already in progress for {hash}.")
                return
            update_state = self.UpdateStatus('running', 0)
            self.running_tasks[hash] = update_state
        try:
            self._build(hash, index, client)
            self.stop(hash, status="finished")
        except Exception as e:
            with self.lock:
                task = self.running_tasks[hash]
                task.status = 'error'
                task.error = str(e.with_traceback(None))
                self.ended_tasks[hash] = task
                del self.running_tasks[hash]
        
    def _build(self, hash: str, index: Index, client: ElvClient):
        update_state = self.running_tasks[hash]
        field_configs = self._get_field_configs(hash, client)
        num_docs = self._get_num_docs(hash, client)
        # Unpack the entire index with a "select all" query
        with timeit("unpacking index"):
            all_docs = client.search(version_hash=hash, query={"filters": "has_field:id", "max_total": num_docs, "limit": num_docs, "display_fields": ["all"]})['results']
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
            logging.info("Performing clean up...")
            for hash in self.running_tasks:
                self.running_tasks[hash].stop_event.set()
                del self.running_tasks[hash]
            # TODO: this global path might eventually be shared so probably should parameterize it
            shutil.rmtree(config.TMP_PATH, ignore_errors=True)

    def _get_field_configs(self, hash: str, client: ElvClient) -> Dict[str, Dict[str, float]]:
        res = client.content_object_metadata(version_hash=hash, metadata_subtree='indexer/config/indexer/arguments/fields')
        return res

    # Retrieves the total number of documents in the index
    def _get_num_docs(self, hash: str, client: ElvClient) -> int:
        res = client.content_object_metadata(version_hash=hash, metadata_subtree='indexer/stats/document/total')
        return int(res)
