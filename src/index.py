import faiss
import torch
from typing import List, Dict, Any, Type
import os
import dill
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import shutil

from src.index_interface import Index
from src.classes import TextDocument, ScorerFactory, Result

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaissIndex(Index):
    def __init__(self, path: str, encoder: str, scoring: ScorerFactory, index_type: Type==faiss.IndexFlatL2):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.encoder = encoder
        # maps field -> faiss index
        self.indices = {}
        # maps field -> list of uids (essentially map from idx -> uid)
        self.id_map = {}
        # maps uid -> text data
        self.text_data: Dict[str, TextDocument] = {}
        self.scoring = scoring
        self.index_type = index_type
        os.makedirs(os.path.join(path, 'indices'), exist_ok=True)
        os.makedirs(os.path.join(path, 'vector_data'), exist_ok=True)
        self.is_setup = False

    @staticmethod
    def from_path(path: str) -> Index:
        pkl_path = os.path.join(path, 'index.pkl')
        with open(pkl_path, 'rb') as f:
            index = dill.load(f)
        index.setup()
        return index

    """
    Args:
        field: Field under which to add the text
        uid: Unique identifier of the document
        text: Text to add to the index
    Side Effects:
        Adds the text to the index so that it can be searched
    """
    def add(self, field: str, uid: str, text: List[str]) -> None:
        if not self.is_setup:
            self.setup()
        embeddings = self.encoder_model.encode(text, show_progress_bar=False)
        if field not in self.id_map:
            self.id_map[field] = []
            self.indices[field] = self.index_type(self.shape)
        self.indices[field].add(embeddings)
        for _ in embeddings:
            self.id_map[field].append(uid)
        if uid not in self.text_data:
            self.text_data[uid] = {}
        self.text_data[uid][field] = text
        if not os.path.exists(os.path.join(self.path, 'vector_data', uid_to_path(uid))):
            os.makedirs(os.path.join(self.path, 'vector_data', uid_to_path(uid)))
        with open(os.path.join(self.path, 'vector_data', uid_to_path(uid), f'{field}.pkl'), 'wb') as f:
            torch.save(embeddings, f)
        self.text_data[uid][field] = text

    """
    Args:
        query: Text query to search the index
        fields: fields to search on. 
            Default: all fields
        k1: Number of results from each field to retrieve in retrieval step
        k2: Final number of results to present after re-ranking
    Returns:
        List of uids of the top k2 results
    """
    def search(self, query: str, fields: List[str]=None, k1: int=500, k2: int=10) -> List[Result]:
        if not self.is_setup:
            self.setup()
        query_embed = self.encoder_model.encode([query])
        assert len(fields) > 0
        assert all(field in self.indices for field in fields), f"Index is missing some of the requested fields: searching={fields}, available={self.indices.keys()}"
        uids = []
        for field in fields:
            _, ids = self.indices[field].search(query_embed, k1)
            ids = ids.tolist()[0]
            ids = sorted(ids)
            uids.extend(self.id_map[field][i] for i in ids)
        result = self._sort_results(query, query_embed, uids, k2)
        return result
    
    def _sort_results(self, query: str, query_embed: torch.Tensor, uids: List[str], k2: int) -> List[str]:
        scorer = self.scoring(query)
        scores = []
        for uid in uids:
            scores.append(scorer(query_embed, self.get_vector_data(uid)))
        return [(score, uid) for score, uid in sorted(zip(scores, uids), reverse=True)][0:k2]
    
    def get_fields(self) -> List[str]:
        return list(self.indices.keys())

    def get_repr(self, uid: str) -> Dict[str, Any]:
        return self.text_data[uid]

    # Retrieves all vector data from disk 
    def get_vector_data(self, uid: str) -> Dict[str, np.ndarray]:
        if not self.is_setup:
            self.setup()
        path = os.path.join(self.path, 'vector_data', uid_to_path(uid))
        if not os.path.exists(path):
            raise ValueError(f"Vector data for {uid} does not exist")
        data = {}
        for field_data in os.listdir(path):
            field = field_data[0:-4]
            with open(os.path.join(path, field_data), 'rb') as f:
                data[field] = torch.load(f)
        return data
    
    # Backs up index to disk. Index will need to be setup again before it can be used.
    def commit(self) -> None:
        if not self.is_setup:
            raise ValueError("Cannot commit index that has not been setup")
        # hide this attribute so it doesn't get saved
        self.encoder_model = None
        for field, index in self.indices.items():
            faiss.write_index(index, os.path.join(self.path, 'indices', f'{field}.index'))
            self.indices[field] = None
        with open(os.path.join(self.path, 'index.pkl'), 'wb') as f:
            dill.dump(self, f)
        self.is_setup = False

    def setup(self):
        for field in self.indices:
            self.indices[field] = faiss.read_index(os.path.join(self.path, 'indices', f'{field}.index'))
        self.encoder_model = SentenceTransformer(self.encoder)
        self.encoder_model = self.encoder_model.cuda()
        self.shape = self.encoder_model.get_sentence_embedding_dimension()
        self.is_setup = True

    def move(self, new_path: str) -> None:
        shutil.move(self.path, new_path)
        self.path = new_path

def uid_to_path(uid: str) -> str:
    return uid.replace('/', '_')