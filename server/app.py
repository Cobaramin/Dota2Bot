from flask import Flask, jsonify, request
from ActorCritic import Model

app = Flask(__name__)

m = Model()

@app.route('/creep_control/get_model', methods=['GET'])
def get_model():
    return jsonify(m.get_weights())

@app.route('/creep_control/update_model', methods=['POST'])
def update_model():
    m.update(request.json)
    return jsonify({})

@app.route('/creep_control/dump', methods=['GET'])
def dump():
    m.dump()
    return jsonify({})

@app.route('/creep_control/load', methods=['POST'])
def load():
    m.load(request.json['file'])
    return jsonify({})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
