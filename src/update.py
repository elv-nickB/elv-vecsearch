
from typing import Dict
from elv_client_py import ElvClient
import os
import faiss
import shutil

import src.config as config
from src.index import FaissIndex
from src import scoring

"""
Args:
    content_id: index object id to base vector index off of
    auth_token: authentication

Saves a vector index for the given content_id which can be searched.
"""
def build_index(content_id: str, auth_token: str) -> None:
    client = ElvClient.from_configuration_url(config.CONFIG_URL, auth_token)
    field_configs = _get_field_configs(client, content_id)
    num_docs = _get_num_docs(client, content_id)
    all_docs = client.search(object_id=content_id, query={"terms": "has_field:id", "max_total": num_docs, "limit": num_docs, "display_fields": ["all"]})['results']
    scoring_fact = scoring.get_term_weight_scoring_factory(content_id, client)
    index_path = os.path.join(config.INDEX_PATH, content_id)
    shutil.rmtree(index_path, ignore_errors=True)
    os.mkdir(index_path)
    index = FaissIndex(path=index_path,
                       encoder=config.SBERT_MODEL, 
                       scoring=scoring_fact,
                       index_type=faiss.IndexFlatIP)
    for doc in all_docs[150000:160000]:
        uid = f"{doc['hash']}{doc['prefix']}"
        for field, fvalues in doc['fields'].items():
            if not field.startswith('f_') or field_configs[field[2:]]['type'] != 'text':
                continue
            index.add(field, uid, fvalues)
    index.commit()
    
def _get_field_configs(client: ElvClient, content_id: str) -> Dict[str, Dict[str, float]]:
    res = client.content_object_metadata(object_id=content_id, metadata_subtree='indexer/config/indexer/arguments/fields')
    return res

# Retrieves the total number of documents in the index
def _get_num_docs(client: ElvClient, content_id: str) -> int:
    res = client.content_object_metadata(object_id=content_id, metadata_subtree='indexer/stats/document/total')
    return int(res)
