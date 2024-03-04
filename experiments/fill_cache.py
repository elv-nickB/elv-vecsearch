
from src.update import IndexBuilder
from elv_client_py import ElvClient
from src import config

import argparse 
import os
from src.index import FaissIndex
import tempfile
from src.embedding import VideoTagEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=str, help="Authorization token", required=True)
    parser.add_argument("--qid", type=str, help="Index id", required=True)
    args = parser.parse_args()
    client = ElvClient.from_configuration_url(config.CONFIG_URL, args.auth)
    tmp_path = tempfile.mkdtemp(dir=config.TMP_PATH)
    index = FaissIndex(tmp_path, config.IndexConstructor)
    encoder = VideoTagEncoder(config.SBERT_MODEL)
    index_builder = IndexBuilder(encoder)  

    index_builder.build(args.qid, index, client)
    save_path = os.path.join(config.DATA_PATH, 'experiments', 'embedding_cache')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    encoder.save_cache(os.path.join(save_path, args.qid))