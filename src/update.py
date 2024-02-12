
from typing import Dict
from elv_client_py import ElvClient
import os
import shutil
import logging
import threading
import numpy as np

import src.config as config
from src.index import Index
from src.embedding import TextEncoder
from src.utils import timeit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TODO: ensure thread safety for this class
class IndexBuilder():
    class UpdateStatus():
        def __init__(self, status: str, progress: float):
            self.status = status
            self.progress = progress
            self.stop_event = threading.Event()
            self.lock = threading.Lock()
            self.error = None
            
    def __init__(self, encoder: TextEncoder):
        self.encoder = encoder
        self.tasks = {}

    def stop(self, qid: str) -> UpdateStatus:
        if qid not in self.tasks:
            return None
        with self.tasks[qid].lock:
            self.tasks[qid].stop_event.set()
            shutil.rmtree(os.path.join(config.TMP_PATH, qid), ignore_errors=True)
            self.tasks[qid].status = 'stopped'
        return self.tasks[qid]

    def get_status(self, content_id: str) -> UpdateStatus:
        if content_id not in self.tasks:
            return None
        return self.tasks[content_id]

    """
    Args:
        content_id: index object id to base vector index off of

    Saves a vector index for the given content_id which can be searched.
    """
    def build(self, content_id, index: Index, client: ElvClient) -> None:
        try:
            self._build(content_id, index, client)
        except Exception as e:
            with self.tasks[content_id].lock:
                self.tasks[content_id].status = 'error'
                self.tasks[content_id].error = str(e)
        
    def _build(self, content_id: str, index: Index, client: ElvClient) -> None:
        # TODO: ensure thread safety
        update_state = self.UpdateStatus('running', 0)
        self.tasks[content_id] = update_state
        field_configs = self._get_field_configs(content_id, client)
        num_docs = self._get_num_docs(content_id, client)
        # Unpack the entire index with a "select all" query
        with timeit("unpacking index"):
            all_docs = client.search(object_id=content_id, query={"filters": "has_field:id", "max_total": num_docs, "limit": num_docs, "display_fields": ["all"]})['results']
        logging.info("Embedding documents and adding to index.")
        for i, doc in enumerate(all_docs): 
            if update_state.stop_event.is_set():
                logging.info("Stopping indexing.")
                with update_state.lock:
                    update_state.status = "stopped"
                return
            uid = f"{doc['hash']}{doc['prefix']}"
            for field, fvalues in doc['fields'].items():
                if not field.startswith('f_') or field_configs[field[2:]]['type'] != 'text':
                    continue
                embeddings = self.encoder(fvalues)
                index.add(field, uid, embeddings)
            update_state.progress = (i+1) / len(all_docs)
        with timeit("committing index"):
            index.commit()
        with update_state.lock:
            update_state.status = "complete"
        logging.info("Indexing complete.")

    def _get_field_configs(self, content_id: str, client: ElvClient) -> Dict[str, Dict[str, float]]:
        res = client.content_object_metadata(object_id=content_id, metadata_subtree='indexer/config/indexer/arguments/fields')
        return res

    # Retrieves the total number of documents in the index
    def _get_num_docs(self, content_id: str, client: ElvClient) -> int:
        res = client.content_object_metadata(object_id=content_id, metadata_subtree='indexer/stats/document/total')
        return int(res)
    
    def cleanup(self) -> None:
        for qid in self.tasks:
            with self.tasks[qid].lock:
                self.tasks[qid].exit_signal.set()
            del self.tasks[qid]
        # TODO: this global path might eventually be shared so probably should parameterize it
        shutil.rmtree(config.TMP_PATH, ignore_errors=True)
