import logging
import time
import json
from elv_client_py import ElvClient
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class timeit:
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        logging.info(f'{self.message}')
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logging.info(f'Finished {self.message}...')
        logging.info(f"Elapsed time: {self.interval:.4f} seconds")

class LRUSearchCache:
    def __init__(self, client: ElvClient):
        self.client = client
    
    def search(self, object_id: str, query: dict) -> dict:
        query = json.dumps(query, sort_keys=True)
        return self._cached_search(object_id, query)
    
    def __getattr__(self, name):
        return getattr(self.client, name)
    
    @lru_cache(maxsize=None)
    def _cached_search(self, object_id: str, query: str) -> dict:
        return self.client.search(object_id=object_id, query=json.loads(query))
