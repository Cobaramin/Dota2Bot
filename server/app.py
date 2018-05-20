import os
import threading

from flask import Flask, jsonify, request

from DDPG import DDPG
from setting import cf

app = Flask(__name__)

model = DDPG()


# def test_function():

#     reload_file = os.path.dirname(os.path.abspath(__file__)) + '/reload.py'
#     # os.system('python %s --timestamp 1521285961 --ep 151000' % (reload_file))
#     os.system('python %s --timestamp 1523632535 --ep 118000' % (reload_file))
#
#
# t = threading.Thread(target=test_function, args=([]))
# t.start()

print('log_dir: ' + cf.TMP_PATH)


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
    model.load(request.json['ep'], request.json['timestamp'])
    return jsonify({})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
