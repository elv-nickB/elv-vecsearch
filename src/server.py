from flask import Flask, request, Response
import argparse
import json

from src.search import search 
from src.update import build_index
from src.format import SearchArgs

app = Flask(__name__)

def get_server():
    server = Flask(__name__)

    @server.route('/q/<qid>/search')
    def handle_search(qid: str):
        args = request.args.to_dict()
        args = SearchArgs().load(args)
        res = search(qid, args, args["auth"])
        res = Response(response=json.dumps(res), status=200, mimetype='application/json')
        return res

    @server.route('/q/<qid>/search_update?auth=<auth>')
    def search_update(qid: str, auth: str):
        res = build_index(qid, auth)
        res = Response(response=json.dumps(res), status=200, mimetype='application/json')

    return server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8085)
    args = parser.parse_args()
    server = get_server()
    server.run(port=args.port)  