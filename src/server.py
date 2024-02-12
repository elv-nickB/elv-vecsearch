from flask import Flask, request, Response
import argparse
import json
import os
import threading
import atexit
import signal
import shutil
from elv_client_py import ElvClient
from flask_cors import CORS
import tempfile
import logging

from src.search import Searcher 
from src.rank import SimpleRanker
from src.update import IndexBuilder
from src.format import SearchArgs
from src.embedding import get_encoder
from src import config
from src.index import FaissIndex
from src.query_understanding import SimpleQueryProcessor

# TODO: ensure thread safety 
def get_server():
    server = Flask(__name__)
    encoder = get_encoder(config.SBERT_MODEL)
    index_builder = IndexBuilder(encoder)

    def _update(qid: str, auth: str) -> None:
        client = ElvClient.from_configuration_url(config.CONFIG_URL, auth)
        tmp_path = tempfile.mkdtemp(dir=config.TMP_PATH)
        index = FaissIndex(tmp_path, config.IndexConstructor)
        index_builder.build(qid, index, client)
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
        index = FaissIndex.from_path(os.path.join(config.INDEX_PATH, qid))
        processor = SimpleQueryProcessor(client, encoder)
        ranker = SimpleRanker(index)
        searcher = Searcher(qid, client, processor, index, ranker)

        if "search_fields" not in args:
            args["search_fields"] = index.get_fields()

        res = searcher.search(args)
       
        return Response(response=json.dumps(res), status=200, mimetype='application/json')

    @server.route('/q/<qid>/search_update')
    def handle_update(qid: str) -> Response:
        args = request.args
        auth = args.get('auth')
        if not _check_access(qid, auth):
            return Response(response=json.dumps({'error': f'Unauthorized, qid={qid}'}), status=401, mimetype='application/json')

        status = index_builder.get_status(qid)
        if status is not None:
            return Response(response=json.dumps({'error': f'Indexing already in progress, qid={qid}, status={status.status}, progress={status.progress}'}), status=400, mimetype='application/json')
        
        threading.Thread(target=_update, args=(qid, auth)).start()
        return Response(response=json.dumps({'lro_handle': qid}), status=200, mimetype='application/json')

    @server.route('/q/<qid>/update_status')
    def handle_status(qid: str) -> Response:
        if not _check_access(qid, request.args.get('auth')):
            return Response(response=json.dumps({'error': f'Unauthorized, qid={qid}'}), status=401, mimetype='application/json')
        status = index_builder.get_status(qid)
        if status is not None:
            return Response(response=json.dumps({'status': status.status, 'progress': status.progress, 'error': status.error}), status=200, mimetype='application/json')
        elif _is_indexed(qid):
            return Response(response=json.dumps({'status': 'complete', 'progress': 1.0, 'error': None}), status=200, mimetype='application/json')
        else:
            return Response(response=json.dumps({'error': 'No index build has not been initiated for qid={qid}'}), status=400, mimetype='application/json')
    
    @server.route('/q/<qid>/stop_update')
    def handle_stop(qid: str) -> Response:
        if not _check_access(qid, request.args.get('auth')):
            return Response(response=json.dumps({'error': f'Unauthorized, qid={qid}'}), status=401, mimetype='application/json')
        status = index_builder.stop(qid)
        if status is None:
            return Response(response=json.dumps({'error': f'No index build has not been initiated for qid={qid}'}), status=400, mimetype='application/json')
        return Response(response=json.dumps({'status': status.status, 'progress': status.progress, 'error': status.error}), status=200, mimetype='application/json')

    # register cleanup on exit
    atexit.register(index_builder.cleanup)
    signal.signal(signal.SIGTERM, index_builder.cleanup)
    signal.signal(signal.SIGINT, index_builder.cleanup)

    CORS(server)
    return server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8085)
    args = parser.parse_args()
    server = get_server()
    server.run(port=args.port)  