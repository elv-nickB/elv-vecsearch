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

from src.search.simple import SimpleSearcher 
from src.ranking.simple import SimpleRanker
from src.ranking.scorers import get_semantic_scorer
from src.query_processing.simple import SimpleProcessor
from src.update.builder import IndexBuilder
from src.format import SearchArgs
from src.embedding.chunk import ChunkEncoder
from src.embedding.utils import load_encoder_with_cache
from src.index.faiss import FaissIndex
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_server():
    server = Flask(__name__)
    text_encoder = load_encoder_with_cache(config.SBERT_MODEL)
    encoder = ChunkEncoder(text_encoder, 5, 2)
    index_builder = IndexBuilder(encoder)
    searchers = {}

    def _update(qid: str, auth: str) -> None:
        client = ElvClient.from_configuration_url(config.CONFIG_URL, auth)
        tmp_path = tempfile.mkdtemp(dir=config.TMP_PATH)
        index = FaissIndex(tmp_path, config.IndexConstructor)
        index_builder.build(qid, index, client)
        if index_builder.get_status(qid).status == 'finished':
            if os.path.exists(os.path.join(config.INDEX_PATH, qid)):
                logging.warning(f'Index already exists for qid={qid}, overwriting')
                shutil.rmtree(os.path.join(config.INDEX_PATH, qid))
            index.set_path(os.path.join(config.INDEX_PATH, qid))

    def _is_indexed(qid: str) -> bool:
        return qid in os.listdir(config.INDEX_PATH)
        
    def _check_access(qid: str, auth: str) -> bool:
        client = ElvClient.from_configuration_url(config.CONFIG_URL, auth)
        try:
            client.content_object(object_id=qid)
        except:
            return False
        return True

    @server.route('/q/<qid>/search')
    def handle_search(qid: str) -> Response:
        if not _check_access(qid, request.args.get('auth')):
            return Response(response=json.dumps({'error': f'Unauthorized, qid={qid}'}), status=401, mimetype='application/json')

        if not _is_indexed(qid):
            status = index_builder.get_status(qid)
            if status is not None:
                return Response(response=json.dumps({'error': f'Index update has not completed for qid={qid}, status={status.status}'}), status=400, mimetype='application/json')
            return Response(response=json.dumps({'error': f'Index has not been built for qid={qid}'}), status=400, mimetype='application/json')
        
        args = request.args.to_dict()
        try:
            args = SearchArgs().load(args)
        except ValueError as e:
            return Response(response=json.dumps({'error': str(e)}), status=400, mimetype='application/json')
        
        client = ElvClient.from_configuration_url(config.CONFIG_URL, args['auth'])
        if qid not in searchers:
            index = FaissIndex.from_path(os.path.join(config.INDEX_PATH, qid))
            processor = SimpleProcessor(client, text_encoder)
            ranker = SimpleRanker(index, get_semantic_scorer())
            searcher = SimpleSearcher(qid, client, processor, index, ranker)
            searchers[qid] = searcher
        searcher = searchers[qid]

        if "search_fields" not in args:
            args["search_fields"] = searcher.index.get_fields()

        res = searcher.search(args)
       
        return Response(response=json.dumps(res), status=200, mimetype='application/json')

    @server.route('/q/<qid>/search_update')
    def handle_update(qid: str) -> Response:
        args = request.args
        auth = args.get('auth')
        if not _check_access(qid, auth):
            return Response(response=json.dumps({'error': f'Unauthorized, qid={qid}'}), status=401, mimetype='application/json')

        status = index_builder.get_status(qid)
        if status and status.status == 'running':
            return Response(response=json.dumps({'error': f'Indexing already in progress, qid={qid}, status={status.status}, progress={status.progress}'}), status=400, mimetype='application/json')
        
        threading.Thread(target=_update, args=(qid, auth)).start()
        return Response(response=json.dumps({'lro_handle': qid}), status=200, mimetype='application/json')

    @server.route('/q/<qid>/update_status')
    def handle_status(qid: str) -> Response:
        if not _check_access(qid, request.args.get('auth')):
            return Response(response=json.dumps({'error': f'Unauthorized, qid={qid}'}), status=401, mimetype='application/json')
        status = index_builder.get_status(qid)
        if status:
            return Response(response=json.dumps({'status': status.status, 'progress': status.progress, 'error': status.error}), status=200, mimetype='application/json')
        else:
            return Response(response=json.dumps({'error': 'No index build has been initiated for qid={qid}'}), status=400, mimetype='application/json')
    
    @server.route('/q/<qid>/stop_update')
    def handle_stop(qid: str) -> Response:
        if not _check_access(qid, request.args.get('auth')):
            return Response(response=json.dumps({'error': f'Unauthorized, qid={qid}'}), status=401, mimetype='application/json')
        status = index_builder.stop(qid)
        if status is None:
            return Response(response=json.dumps({'error': f'No index build has been initiated for qid={qid}'}), status=400, mimetype='application/json')
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