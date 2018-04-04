import os
import threading

from DDPG import DDPG
from flask import Flask, jsonify, request
from setting import cf

app = Flask(__name__)

model = DDPG()

# Start tensorboard
# def launchTensorBoard():
#     os.system('tensorboard --port=6006 --logdir=' + cf.TMP_PATH)
#     print('.....Starting tensorboard')
#     return

# t = threading.Thread(target=launchTensorBoard, args=([]))
# t.start()
print('log_dir: '+ cf.TMP_PATH)


@app.route('/creep_control/get_model', methods=['GET'])
def get_model():
    return jsonify(model.get_model())


@app.route('/creep_control/update_model', methods=['POST'])
def update_model():
    # Mock data
    # data = {'0': {'s': [1]*11, 'a': [1, 0.1], 'r': 2., 's1': [3]*11, 'done': 0},
    #         '1': {'s': [2]*11, 'a': [-1, -0.8], 'r': 5., 's1': [5]*11, 'done': 1},
    #         'ep': i}
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
