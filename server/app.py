from flask import Flask, jsonify, request
# from ActorCritic import Model
from DDPG import DDPG
from setting import cf

app = Flask(__name__)

model = DDPG()

@app.route('/creep_control/get_model', methods=['GET'])
def get_model():
    return jsonify(model.get_model())

@app.route('/creep_control/update_model', methods=['POST'])
def update_model():
    model.update(request.json, train_indicator=cf.TRAIN)
    return jsonify({})

@app.route('/creep_control/dump', methods=['GET'])
def dump():
    model.dump()
    return jsonify({})

@app.route('/creep_control/load', methods=['POST'])
def load():
    model.load(request.json['ep'])
    return jsonify({})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
