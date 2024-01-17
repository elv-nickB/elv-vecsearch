from flask import Flask, request, Response
import argparse
import json
import os
import threading
import atexit
import signal
import shutil
from elv_client_py import ElvClient

from src.search import search 
from src.update import build_index
from src.format import SearchArgs
from src.classes import UpdateStatus
from src import config

def get_server():
    server = Flask(__name__)

    tasks = {}

    def _update(qid: str, auth: str, status: UpdateStatus) -> None:
        try:
            build_index(qid, auth, status=status)
        except Exception as e:
            status.status = 'error'
            status.error = str(e)
            return
        status.status = 'complete'

    def _is_indexed(qid: str) -> bool:
        if qid in os.listdir(config.INDEX_PATH) and not qid in tasks:
            return True
        elif qid in tasks:
            with tasks[qid].lock:
                return tasks[qid].status == 'complete'
        
    def _check_access(qid: str, auth: str) -> bool:
        client = ElvClient.from_configuration_url(config.CONFIG_URL, auth)
        try:
            client.content_object(object_id=qid)
        except Exception as e:
            return False
        return True

    @server.route('/q/<qid>/search')
    def handle_search(qid: str) -> Response:
        if not _check_access(qid, request.args.get('auth')):
            return Response(response=json.dumps({'error': f'Unauthorized, qid={qid}'}), status=401, mimetype='application/json')

        if not _is_indexed(qid):
            return Response(response=json.dumps({'error': f'Index has not been built or is currently being updated for qid={qid}'}), status=400, mimetype='application/json')
        
        args = request.args.to_dict()
        try:
            args = SearchArgs().load(args)
        except ValueError as e:
            return Response(response=json.dumps({'error': str(e)}), status=400, mimetype='application/json')
        
        try:
            res = search(qid, args, args["auth"])
        except ValueError as e:
            return Response(response=json.dumps({'error': str(e)}), status=400, mimetype='application/json')
        
        return Response(response=json.dumps(res), status=200, mimetype='application/json')

    @server.route('/q/<qid>/search_update')
    def search_update(qid: str) -> Response:
        if not _check_access(qid, request.args.get('auth')):
            return Response(response=json.dumps({'error': f'Unauthorized, qid={qid}'}), status=401, mimetype='application/json')

        if qid in tasks:
            return Response(response=json.dumps({'error': 'Indexing already in progress'}), status=400, mimetype='application/json')
        args = request.args
        auth = args.get('auth')
        tasks[qid] = UpdateStatus("running", 0.0)
        threading.Thread(target=_update, args=(qid, auth, tasks[qid])).start()
        return Response(response=json.dumps({'lro_handle': qid}), status=200, mimetype='application/json')

    @server.route('/q/<qid>/update_status')
    def status(qid: str) -> Response:
        if not _check_access(qid, request.args.get('auth')):
            return Response(response=json.dumps({'error': f'Unauthorized, qid={qid}'}), status=401, mimetype='application/json')
        if qid in tasks:
            status = tasks[qid]
            with status.lock:
                res = {'status': status.status, 'progress': status.progress}
                if status.error is not None:
                    res['error'] = status.error
            return Response(response=json.dumps(res), status=200, mimetype='application/json')
        elif _is_indexed(qid):
            return Response(response=json.dumps({'status': 'complete'}), status=200, mimetype='application/json')
        else:
            return Response(response=json.dumps({'error': 'No indexing in progress'}), status=400, mimetype='application/json')
    
    @server.route('/q/<qid>/stop_update')
    def stop(qid: str) -> Response:
        if not _check_access(qid, request.args.get('auth')):
            return Response(response=json.dumps({'error': f'Unauthorized, qid={qid}'}), status=401, mimetype='application/json')
        
        if qid not in tasks:
            return Response(response=json.dumps({'error': 'No indexing in progress'}), status=400, mimetype='application/json')
        with tasks[qid].lock:
            tasks[qid].stop_event.set()
        shutil.rmtree(os.path.join(config.TMP_PATH, qid), ignore_errors=True)
        with tasks[qid].lock:
            tasks[qid].status = 'stopped'
        return Response(response=json.dumps({'status': 'stopping'}), status=200, mimetype='application/json')

    def cleanup():
        for qid in tasks:
            with tasks[qid].lock:
                tasks[qid].exit_signal.set()
            del tasks[qid]
        shutil.rmtree(config.TMP_PATH, ignore_errors=True)

    # register cleanup on exit
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    return server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8085)
    args = parser.parse_args()
    server = get_server()
    server.run(port=args.port)  