from flask import Flask, request, Response
import argparse
import json
import os
import threading
import atexit
import shutil
from elv_client_py import ElvClient
from flask_cors import CORS
import tempfile
import logging
from typing import Tuple

from src.search.simple import SimpleSearcher 
from src.ranking.simple import SimpleRanker
from src.ranking.scorers import get_semantic_scorer
from src.query_processing.simple import SimpleProcessor
from src.update.builder import IndexBuilder
from src.format import SearchArgs
from src.embedding.object_clean import ObjectCleanEncoder
from src.embedding.utils import load_encoder_with_cache
from src.index.faiss import FaissIndex
from src import config
from src.utils import to_hash

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_server():
    server = Flask(__name__)
    text_encoder = load_encoder_with_cache(config.SBERT_MODEL)
    encoder = ObjectCleanEncoder(text_encoder)
    index_builder = IndexBuilder(encoder)
    searchers = {}

    def _is_indexed(hash: str) -> bool:
        return hash in os.listdir(config.INDEX_PATH)
        
    # Validates permission, converts qhot to hash, and creates an ElvClient instance based on auth token
    def _get_client_and_hash(qhot: str, auth: str) -> Tuple[str, ElvClient]:
        client = ElvClient.from_configuration_url(config.CONFIG_URL, auth)
        try:
            hash = to_hash(qhot, client)
        except:
            return None, None
        return hash, client

    @server.route('/q/<qhot>/rep/search')
    def handle_search(qhot: str) -> Response:
        return _search(qhot)
    
    @server.route('/qlibs/<qlib>/q/<qhot>/rep/search')
    def handle_search_with_qlib(qlib: str, qhot: str) -> Response:
        # ignore qlib 
        return _search(qhot)

    def _search(qhot: str) -> Response:
        hash, client = _get_client_and_hash(qhot, request.args.get('authorization'))
        if hash is None:
            return Response(response=json.dumps({'error': f'Unauthorized, qhot={qhot}'}), status=401, mimetype='application/json')

        if not _is_indexed(hash):
            status = index_builder.get_status(hash)
            if status is not None:
                return Response(response=json.dumps({'error': f'Index update has not completed for hash={hash}, status={status.status}'}), status=400, mimetype='application/json')
            return Response(response=json.dumps({'error': f'Index has not been built for hash={hash}'}), status=400, mimetype='application/json')
        
        args = request.args.to_dict()
        try:
            args = SearchArgs().load(args)
        except ValueError as e:
            return Response(response=json.dumps({'error': str(e)}), status=400, mimetype='application/json')
        
        if hash not in searchers:
            index = FaissIndex.from_path(os.path.join(config.INDEX_PATH, hash))
            processor = SimpleProcessor(client, text_encoder)
            ranker = SimpleRanker(index, get_semantic_scorer(0.0, 0.0))
            searcher = SimpleSearcher(hash, client, processor, index, ranker)
            searchers[hash] = searcher
        searcher = searchers[hash]

        if "search_fields" not in args:
            args["search_fields"] = searcher.index.get_fields()

        res = searcher.search(args)
       
        return Response(response=json.dumps(res), status=200, mimetype='application/json')

    @server.route('/q/<qhot>/search_update')
    def handle_search_update(qhot: str) -> Response:
        return _update(qhot)
    
    @server.route('/qlibs/<qlib>/q/<qhot>/rep/search')
    def handle_update_with_qlib(qlib: str, qhot: str) -> Response:
        # ignore qlib 
        return _update(qhot)

    def _update(qhot: str) -> Response:
        hash, client = _get_client_and_hash(qhot, request.args.get('authorization'))
        if hash is None:
            return Response(response=json.dumps({'error': f'Unauthorized, qhot={qhot}'}), status=401, mimetype='application/json')
        
        args = request.args
        auth = args.get('authorization')
        client = ElvClient.from_configuration_url(config.CONFIG_URL, auth)
        hash = to_hash(qhot, client)

        status = index_builder.get_status(hash)
        if status and status.status == 'running':
            return Response(response=json.dumps({'error': f'Indexing already in progress, hash={hash}, status={status.status}, progress={status.progress}'}), status=400, mimetype='application/json')
        
        threading.Thread(target=_update_job, args=(hash, client)).start()
        return Response(response=json.dumps({'success': True}), status=200, mimetype='application/json')
    
    def _update_job(hash: str, client: ElvClient) -> None:
        tmp_path = tempfile.mkdtemp(dir=config.TMP_PATH)
        index = FaissIndex(tmp_path, config.IndexConstructor)
        index_builder.build(hash, index, client)
        if index_builder.get_status(hash).status == 'finished':
            if os.path.exists(os.path.join(config.INDEX_PATH, hash)):
                logging.warning(f'Index already exists for hash={hash}, overwriting')
                shutil.rmtree(os.path.join(config.INDEX_PATH, hash))
            index.set_path(os.path.join(config.INDEX_PATH, hash))

    @server.route('/q/<qhot>/update_status')
    def handle_status(qhot: str) -> Response:
        return _status(qhot)
    
    @server.route('/qlibs/<qlib>/q/<qhot>/rep/update_status')
    def handle_status_with_qlib(qlib: str, qhot: str) -> Response:
        return _status(qhot)

    def _status(qhot: str) -> Response:
        hash, client = _get_client_and_hash(qhot, request.args.get('authorization'))                               
        if hash is None:
            return Response(response=json.dumps({'error': f'Unauthorized, qhot={qhot}'}), status=401, mimetype='application/json')
        
        status = index_builder.get_status(hash)
        if status:
            return Response(response=json.dumps({'status': status.status, 'progress': status.progress, 'error': status.error}), status=200, mimetype='application/json')
        else:
            return Response(response=json.dumps({'error': f'No index build has been initiated for hash={hash}'}), status=400, mimetype='application/json')
    
    @server.route('/q/<qhot>/stop_update')
    def handle_stop(qhot: str) -> Response:
        return _stop(qhot)
    
    @server.route('/qlibs/<qlib>/q/<qhot>/rep/stop_update')
    def handle_stop_with_qlib(qlib: str, qhot: str) -> Response:
        return _stop(qhot)

    def _stop(qhot: str) -> Response:
        hash, _ = _get_client_and_hash(qhot, request.args.get('authorization'))
        if hash is None:
            return Response(response=json.dumps({'error': f'Unauthorized, qhot={qhot}'}), status=401, mimetype='application/json')
        
        status = index_builder.stop(hash)
        if status is None:
            return Response(response=json.dumps({'error': f'No index build has been initiated for hash={hash}'}), status=400, mimetype='application/json')
        return Response(response=json.dumps({'status': status.status, 'progress': status.progress, 'error': status.error}), status=200, mimetype='application/json')

    # register cleanup on exit
    atexit.register(index_builder.cleanup)

    CORS(server)
    return server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8085)
    args = parser.parse_args()
    server = get_server()
    server.run(port=args.port)  