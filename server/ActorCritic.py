import os

# test
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam
from sklearn.datasets.samples_generator import make_circles

from Memory import Memory


class Model:

    def __init__(self):

        # Variable Definition
        self.ep = 0
        self.explore = 5
        self.boot_strap = 0
        self.boot_strap_freq = 5
        self.replace_freq = 10
        self.save_freq = 25
        self.save_path = os.path.dirname(os.path.abspath(__file__)) + '/weights/'

        # Net Variable
        self.net_layers = [2, 4, 1]
        self.num_layers = len(self.net_layers)
        self.learning_rate = 0.01

        # Network
        # self.critic_network = Net(
        #     net_layers=self.net_layers,
        #     learning_rate=self.learning_rate)

        self.policy_network = Net(
            net_layers=self.net_layers,
            learning_rate=self.learning_rate)

        self.memory = Memory()

    def get_model(self):
        # helper function
        def get_json_each_model(weights):
            num_dense = self.num_layers - 1
            data = []
            for i in range(0, 2 * num_dense, 2):
                data.append({
                    'dense': i / 2,
                    'weights': {'shape': weights[i].shape, 'value': weights[i].tolist()},
                    'bias': {'shape': weights[i + 1].shape, 'value': weights[i + 1].tolist()}
                })
            return {'len': num_dense, 'data': data}

        policy_network_weights = self.policy_network.get_weights()
        return {'policy': get_json_each_model(policy_network_weights),
                'replace': (self.ep % self.replace_freq) == 0}

    def update(self, data):
        # Helper function
        def parse_data(data):
            SAR = []
            R = 0
            for i in reversed(range(len(data)-1)):
                s_t = data[i]['s']
                action = data[i]['a']
                R = data[i]['r'] + 0.7 * R
                SAR.append([s_t, action, R])
            SAR = np.array(SAR, dtype=np.float32)
            np.clip(SAR[:,2], -2, 2) # cliping reward
            return zip(SAR[:,0], SAR[:,1], SAR[:,2])

        # Start training
        for d in self.parse_data(data):
            self.memory.insert(d)
            if self.memory.full() and np.random.rand() < 0.1:
                batch = self.memory.get_batch()
                self.policy_network.train(batch)


        # Duplicate Episode
        if data['ep'] == self.ep:
            print("Duplicate ep %d" % self.ep, file=sys.stderr)
            return
        self.ep = data['ep']
        print("Ep: %d" % self.ep, file=sys.stderr)

        for d in self.parse_data(data):
            self.memory.insert(d)

        self.policy_network.train()

    def load(self, file):
        self.policy_network.load_weights(self.save_path + 'aaaa.hdf5')
        print('already loaded weights from file "%s"' % file)

    def dump(self):
        self.policy_network.save_weights(self.save_path + 'aaaa.hdf5')
        print('already saved weights to "%s"' % self.save_path)


class Net:

    def __init__(self, net_layers, learning_rate):
        self.tf_session = K.get_session()  # this creates a new session since one doesn't exist already.
        self.tf_graph = tf.get_default_graph()

        # self.sess = tf.Session()
        self.net_layers = net_layers
        self.learning_rate = learning_rate

        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                self.model = self.new_model()

    def new_model(self):
        # K.set_session(self.sess)

        model = Sequential()
        model.add(Dense(self.net_layers[1], input_dim=self.net_layers[0], activation="relu"))
        for num_node in self.net_layers[2:-1]:
            model.add(Dense(num_node, activation="relu"))
        model.add(Dense(self.net_layers[-1]))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))

        # model = Sequential()
        # model.add(Dense(4, input_shape=(2,), activation='tanh'))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, batch):
        s_t, action, reward = zip(*batch)

        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                self.model.fit(X, y, epochs=20)

    def get_weights(self):
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                weights = self.model.get_weights()
        return weights

    def save_weights(self, path):
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                self.model.save_weights(path)

    def load_weights(self, path):
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                try:
                    self.model.load_weights(path, by_name=True)
                except OSError:
                    print('File does not exist')
