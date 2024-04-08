from flask import Flask, session, request, jsonify, Response
from flask_cors import CORS
from flask_session import Session  

from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import logging
import threading
import argparse
import os
from typing import List, Tuple
from elv_client_py import ElvClient

from search import config
from search.utils import to_hash
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_server():
    app = Flask(__name__)

    secret = os.getenv('CHAT_SECRET')
    if secret is None:
        raise ValueError("CHAT_SECRET environment variable must be set.")

    # Configure secret key and Flask-Session
    app.config['SECRET_KEY'] = secret
    app.config['SESSION_TYPE'] = 'filesystem'  # For this example, using filesystem
    
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, load_in_8bit=False, device_map="auto")

    lock = threading.Lock()

    # Returns response from the model and status of the response
    def prompt_search(messages: List[dict]) -> Tuple[str, bool]:
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=60)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split('[/INST]')[-1]
        if 'SEARCH' in response:
            return response.replace('!', '.').replace('?', '.').replace('\n','.'), True
        elif 'here are' in response.lower():
            response = 'Sorry I could not get it. Will you please try again. A good format is Show me some clips with ...', False

    def validate_auth_token(qhot: str, auth_token: str) -> bool:
        client = ElvClient.from_configuration_url(config.CONFIG_URL, auth_token)
        try:
            if qhot.startswith("hq__"):
                client.content_object(version_hash=qhot)
            elif qhot.startswith("iq__"):
                client.content_object(object_id=qhot)
            else:
                return False
        except Exception as e:
            return False
    
        return True
    
    @app.route('/q/<qhot>/start_session', methods=['GET'])
    def start_session(qhot: str):
        auth_token = request.args.get('authorization')
        if validate_auth_token(qhot, auth_token):
            client = ElvClient.from_configuration_url(config.CONFIG_URL, auth_token)
            data = pd.read_csv(os.path.join(config.PROJECT_PATH, 'messages.csv'))

            prior_messages = [] 
            for _, row in data.iterrows():
                prompt = row['Prompt']
                query = row['Query']
                    
                prior_messages += [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"SEARCH {query}"},  
                ]

            session['index hash'] = to_hash(qhot, client)
            session['chat_context'] = prior_messages  
            return jsonify({"message": "Session started successfully.", "session_id": session.sid}), 200
        else:
            return jsonify({"error": "Invalid auth token."}), 401
        
    @app.route('/q/<qhot>/message', methods=['GET'])
    def message(qhot: str) -> Response:
        if not validate_auth_token(qhot, request.args.get('authorization')):
            return jsonify({"error": "Invalid auth token."}), 401
    
        if 'chat_context' not in session:
            return jsonify({"error": "Session not started."}), 401
        
        client = ElvClient.from_configuration_url(config.CONFIG_URL, request.args.get('authorization'))
        hash = to_hash(qhot, client)
        if hash != session["index hash"]:
            return jsonify({"error": f"Index hash does not match the current session: given={hash}, session={session["index_hash"]}"}), 401
        
        user_message = request.args.get('message').lower()
        chat_context = session.get('chat_context')

        if user_message == 'exit':
            return terminate()
        
        with lock:
            response, ok = prompt_search(chat_context)

        if ok:
            # add to
            chat_context += [{"role": "user", "content": user_message}]
            chat_context += [{"role": "assistant", "content": response}]

        return jsonify({"response": response}), 200
    
    @app.route('/terminate', methods=['GET', 'POST'])
    def handle_term(qhot: str) -> Response:
        return terminate()
    
    def terminate() -> Response:
        session.clear()
        return jsonify({"message": "Session terminated."}), 200
    
    #def _call_search(qhot: str, query: str) -> Any:
    #    return requests.get(f"{search_endpoint}/q/{qhot}/rep/search", params={"terms": query, "max_total": 10, "display_fields":"all", "authorization": request.args.get('authorization')}).json()

    Session(app)
    CORS(app, supports_credentials=True)
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8083)
    args = parser.parse_args()
    server = get_server()
    server.run(port=args.port)  