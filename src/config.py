import os
import faiss

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = os.path.join(SRC_PATH, '..')
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
SBERT_MODEL = 'all-mpnet-base-v2'

CONFIG_URL = 'https://main.net955305.contentfabric.io/config'

TMP_PATH = os.path.join(DATA_PATH, 'tmp')
if not os.path.exists(TMP_PATH):
    os.makedirs(TMP_PATH)
INDEX_PATH = os.path.join(DATA_PATH, 'indices')
if not os.path.exists(INDEX_PATH):
    os.makedirs(INDEX_PATH)

IndexConstructor = lambda: faiss.IndexFlatL2(768)
